"""Local web UI: upload a MongoDB export file and get the dashboard back.

Usage:
    .venv\\Scripts\\python app.py
    -> open http://127.0.0.1:5000 in your browser

Accepts a file exported from your own MongoDB `tickets` collection:
    mongoexport --collection=tickets --db=<db> --jsonArray --out=tickets.json
    mongoexport --collection=tickets --db=<db> --out=tickets.jsonl   (one JSON doc per line)
    mongodump   --collection=tickets --db=<db> --out=dump/           (dump/<db>/tickets.bson)
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from flask import Flask, jsonify, redirect, render_template_string, request, send_from_directory

from src.config import settings
from src.generate_report import generate_report
from src.pipeline import add_manual_ticket, run_pipeline
from src.rag_qa import RagUnavailable, answer_question

app = Flask(__name__)
DEBUG = True  # single source of truth - also read by the auto-refresh startup guard below
# Was forced off (2026-09-22): Werkzeug's file-watching reloader crashed
# under Python 3.14 (AttributeError inside werkzeug/_reloader.py's
# module-path scan, unrelated to this project's code) as soon as torch/
# sentence-transformers got imported. Fixed by moving this project's .venv
# to Python 3.12 - re-enabled now that the underlying incompatibility is
# gone. If it starts crashing again, that's a sign something reintroduced
# an unsupported-Python-version situation, not a reason to just re-disable
# this without checking why.
USE_RELOADER = True

UPLOAD_DIR = settings.data_dir / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ATTACHMENTS_DIR = settings.data_dir / "attachments"

ALLOWED_EXT = {".json", ".jsonl", ".bson", ".csv"}

DOMAIN_OPTIONS = ["App Support", "Hardware", "Software"]
PRIORITY_OPTIONS = ["Low", "Medium", "High", "Critical"]
# Real category/product-line values from the historical dataset (by
# ticket count) - this is what your real data is actually organized by, so
# selecting the correct one here is what lets a manual ticket match a real
# historical repeat from that same category. See src/classify.py's module
# docstring for the measured impact of getting this wrong.
CATEGORY_OPTIONS = [
    "Easychit", "EasyChit Client", "REMO", "S/W", "Finsta NBFC Angular",
    "Finsta DotNet Clients", "Desktop/System", "Printer", "PROPERTY MANAGEMENT", "Other",
]
DEFAULT_PRODUCT = "Easychit"

_refresh_lock = threading.Lock()


def _latest_report_name() -> str | None:
    reports = sorted(settings.reports_dir.glob("report_*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0].name if reports else None


def _mongo_configured() -> bool:
    return bool(settings.mongo_uri and settings.mongo_db)


def check_for_new_tickets(verbose: bool = False) -> dict:
    """Phase 5: run phases 1-3 (extract -> embed -> cluster) incrementally
    against MongoDB and refresh the dashboard if anything new showed up.
    Used by both the background poller and the manual "Check now" button, so
    there's one code path (and one lock, to keep a poll and a manual click
    from racing each other) for "did anything change, and did we handle it."
    """
    if not _mongo_configured():
        return {"ok": False, "error": "MongoDB is not configured (set MONGO_URI/MONGO_DB in .env)."}

    with _refresh_lock:
        try:
            before = 0
            if settings.clusters_path.exists():
                before = json.loads(settings.clusters_path.read_text(encoding="utf-8"))["kpis"]["tickets_analyzed"]

            result = run_pipeline(source="mongo", incremental=True, verbose=verbose)
            if result is None:
                return {"ok": True, "updated": False, "new_tickets": 0}

            generate_report()
            return {
                "ok": True,
                "updated": True,
                "new_tickets": result["kpis"]["tickets_analyzed"] - before,
                "report_url": f"/reports/{_latest_report_name()}",
            }
        except Exception as exc:
            traceback.print_exc()
            return {"ok": False, "error": str(exc)}


def _auto_refresh_loop(interval: int) -> None:
    print(f"[auto-refresh] polling MongoDB for new tickets every {interval}s")
    while True:
        time.sleep(interval)
        status = check_for_new_tickets(verbose=False)
        if status.get("updated"):
            print(f"[auto-refresh] {status['new_tickets']} new ticket(s) processed, dashboard refreshed.")
        elif not status.get("ok"):
            print(f"[auto-refresh] skipped: {status.get('error')}")


def start_auto_refresh_if_configured() -> None:
    if settings.auto_refresh_interval_seconds <= 0 or not _mongo_configured():
        return
    # Flask's debug reloader (when enabled) runs this module twice (a parent
    # that only watches files, and a child that actually serves requests) -
    # only the child should start the poller, or new tickets would get
    # processed twice. Irrelevant when USE_RELOADER is off (no parent
    # process exists), which is why this checks USE_RELOADER, not DEBUG.
    if USE_RELOADER and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        return
    threading.Thread(target=_auto_refresh_loop, args=(settings.auto_refresh_interval_seconds,), daemon=True).start()

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ticket Repetition Analyzer</title>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Arial, sans-serif; background: #f4f5f7;
         margin: 0; padding: 3rem 1rem; color: #1f2430; }
  .card { max-width: 560px; margin: 0 auto; background: #fff; border-radius: 12px;
          box-shadow: 0 1px 3px rgba(0,0,0,.08), 0 8px 24px rgba(0,0,0,.06); padding: 2.5rem; }
  h1 { font-size: 1.35rem; margin: 0 0 .35rem; }
  p.sub { color: #64748b; margin: 0 0 1.75rem; font-size: .92rem; line-height: 1.5; }
  .drop { border: 2px dashed #cbd5e1; border-radius: 10px; padding: 2rem 1rem; text-align: center;
          cursor: pointer; transition: border-color .15s, background .15s; }
  .drop.drag { border-color: #4f7cff; background: #f0f4ff; }
  .drop input[type=file] { display: none; }
  .drop .fname { margin-top: .5rem; font-size: .85rem; color: #334155; word-break: break-all; }
  .btn { display: block; width: 100%; margin-top: 1.25rem; padding: .75rem; border: none;
         border-radius: 8px; background: #4f7cff; color: #fff; font-size: 1rem; font-weight: 600;
         cursor: pointer; }
  .btn:disabled { background: #a9bbe8; cursor: not-allowed; }
  .btn:hover:not(:disabled) { background: #3d64e0; }
  .row { margin-top: 1rem; font-size: .85rem; color: #475569; }
  .row input { width: 90px; padding: .3rem .4rem; border: 1px solid #cbd5e1; border-radius: 6px; }
  .error { margin-top: 1rem; padding: .75rem 1rem; background: #fef2f2; color: #b91c1c;
           border-radius: 8px; font-size: .88rem; }
  .hint { margin-top: 1.5rem; font-size: .78rem; color: #94a3b8; line-height: 1.5; }
  code { background: #f1f5f9; padding: .1rem .35rem; border-radius: 4px; }
  .working { display: none; margin-top: 1rem; font-size: .88rem; color: #475569; }
  .links { margin-top: 1rem; display: flex; gap: .75rem; flex-wrap: wrap; align-items: center; }
  .links a { font-size: .82rem; color: #4f7cff; text-decoration: none; font-weight: 600; }
  .links a:hover { text-decoration: underline; }
  .refresh-btn { font: inherit; font-size: .82rem; font-weight: 600; color: #4f7cff; background: none;
                 border: 1px solid #cbd5e1; border-radius: 6px; padding: .3rem .7rem; cursor: pointer; }
  .refresh-btn:hover:not(:disabled) { background: #f0f4ff; }
  .refresh-btn:disabled { color: #94a3b8; cursor: not-allowed; }
  .refresh-status { margin-top: .6rem; font-size: .8rem; color: #475569; }
  .refresh-status.err { color: #b91c1c; }
  .auto-note { margin-top: .4rem; font-size: .74rem; color: #94a3b8; }
</style>
</head>
<body>
  <div class="card">
    <h1>Ticket Repetition Analyzer</h1>
    <p class="sub">Upload a file exported from your MongoDB tickets collection. It will be
      embedded, clustered into recurring issues, and rendered as a dashboard.</p>

    <div class="links">
      {% if latest_report %}<a href="/reports/{{ latest_report }}" id="reportLink">View last report</a>{% endif %}
      {% if has_data %}<a href="/ask">Ask AI about the analyzed data &rarr;</a>{% endif %}
      <button class="refresh-btn" id="refreshBtn" type="button">Check for new tickets now</button>
    </div>
    <div class="refresh-status" id="refreshStatus"></div>
    {% if mongo_configured and auto_refresh_seconds %}
      <div class="auto-note">Auto-checking MongoDB for new tickets every {{ auto_refresh_seconds }}s while this app is running.</div>
    {% elif not mongo_configured %}
      <div class="auto-note">Phase 5 automation (MongoDB polling) needs MONGO_URI/MONGO_DB set in .env - the button above will explain this if clicked without it.</div>
    {% endif %}

    {% if error %}<div class="error">{{ error }}</div>{% endif %}

    <form id="f" action="/upload" method="post" enctype="multipart/form-data">
      <label class="drop" id="drop" for="file">
        <div>Click to choose, or drag a file here</div>
        <div class="fname" id="fname">.csv / .json / .jsonl / .bson</div>
        <input type="file" id="file" name="file" accept=".csv,.json,.jsonl,.bson" required>
      </label>
      <div class="row">
        Limit to first
        <input type="number" name="limit" min="1" placeholder="all">
        tickets (optional, for a quick test)
      </div>
      <button class="btn" id="go" type="submit">Analyze</button>
      <div class="working" id="working">Processing... first run downloads the local embedding
        model, this can take a few minutes.</div>
    </form>

    <div class="hint">
      Accepted formats: <code>mongoexport --type=csv</code> (.csv),
      <code>mongoexport --jsonArray</code> (.json),
      <code>mongoexport</code> default (.jsonl), or <code>mongodump</code> (.bson).<br>
      Your file is deleted from this machine right after it's processed.
    </div>
  </div>

<script>
  const drop = document.getElementById('drop');
  const input = document.getElementById('file');
  const fname = document.getElementById('fname');
  const form = document.getElementById('f');
  const go = document.getElementById('go');
  const working = document.getElementById('working');

  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
  drop.addEventListener('drop', e => {
    e.preventDefault();
    drop.classList.remove('drag');
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      fname.textContent = input.files[0].name;
    }
  });
  input.addEventListener('change', () => {
    if (input.files.length) fname.textContent = input.files[0].name;
  });
  form.addEventListener('submit', () => {
    go.disabled = true;
    go.textContent = 'Analyzing...';
    working.style.display = 'block';
  });

  const refreshBtn = document.getElementById('refreshBtn');
  const refreshStatus = document.getElementById('refreshStatus');
  const reportLink = document.getElementById('reportLink');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', async () => {
      refreshBtn.disabled = true;
      refreshBtn.textContent = 'Checking...';
      refreshStatus.className = 'refresh-status';
      refreshStatus.textContent = '';
      try {
        const res = await fetch('/refresh', { method: 'POST' });
        const data = await res.json();
        if (!res.ok || !data.ok) {
          refreshStatus.className = 'refresh-status err';
          refreshStatus.textContent = data.error || 'Something went wrong.';
        } else if (data.updated) {
          refreshStatus.textContent = `${data.new_tickets} new ticket(s) found - dashboard refreshed.`;
          if (reportLink && data.report_url) reportLink.href = data.report_url;
        } else {
          refreshStatus.textContent = 'No new tickets since the last check.';
        }
      } catch (err) {
        refreshStatus.className = 'refresh-status err';
        refreshStatus.textContent = 'Network error: ' + err;
      } finally {
        refreshBtn.disabled = false;
        refreshBtn.textContent = 'Check for new tickets now';
      }
    });
  }
</script>
</body>
</html>
"""


def _render_index(error: str | None = None):
    return render_template_string(
        PAGE,
        error=error,
        latest_report=_latest_report_name(),
        has_data=settings.tickets_path.exists(),
        mongo_configured=_mongo_configured(),
        auto_refresh_seconds=settings.auto_refresh_interval_seconds if settings.auto_refresh_interval_seconds > 0 else None,
    )


@app.get("/")
def index():
    return _render_index()


@app.post("/refresh")
def refresh():
    return jsonify(check_for_new_tickets(verbose=True))


@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return _render_index(error="Please choose a file first."), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return _render_index(error=f"Unsupported file type '{ext}'. Use .json, .jsonl, or .bson."), 400

    limit_raw = (request.form.get("limit") or "").strip()
    limit = int(limit_raw) if limit_raw.isdigit() else None

    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    f.save(dest)
    try:
        run_pipeline(source="file", file_path=str(dest), limit=limit)
        report_path = generate_report()
    except Exception as exc:  # surface the real error instead of a blank 500
        traceback.print_exc()
        return _render_index(error=f"Failed to process file: {exc}"), 500
    finally:
        dest.unlink(missing_ok=True)

    return send_from_directory(settings.reports_dir, Path(report_path).name)


@app.get("/reports/<path:name>")
def view_report(name):
    return send_from_directory(settings.reports_dir, name)


SUBMIT_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Submit a Ticket - Ticket Repetition Analyzer</title>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Arial, sans-serif; background: #f4f5f7;
         margin: 0; padding: 3rem 1rem; color: #1f2430; }
  .card { max-width: 560px; margin: 0 auto; background: #fff; border-radius: 12px;
          box-shadow: 0 1px 3px rgba(0,0,0,.08), 0 8px 24px rgba(0,0,0,.06); padding: 2.5rem; }
  .top { display: flex; align-items: center; justify-content: space-between; margin-bottom: .35rem; }
  .top a { font-size: .85rem; color: #4f7cff; text-decoration: none; font-weight: 600; }
  .top a:hover { text-decoration: underline; }
  h1 { font-size: 1.35rem; margin: 0 0 .35rem; }
  p.sub { color: #64748b; margin: 0 0 1.75rem; font-size: .92rem; line-height: 1.5; }
  label { display: block; font-size: .85rem; font-weight: 600; color: #334155; margin: 1rem 0 .35rem; }
  label:first-of-type { margin-top: 0; }
  .req { color: #dc2626; }
  input[type=text], select, textarea {
    width: 100%; box-sizing: border-box; padding: .6rem .75rem; border: 1px solid #cbd5e1;
    border-radius: 8px; font-size: .9rem; font-family: inherit; background: #fff; color: #1f2430;
  }
  textarea { resize: vertical; min-height: 110px; }
  input[type=file] { width: 100%; font-size: .85rem; }
  .row2 { display: flex; gap: 1rem; }
  .row2 > div { flex: 1; }
  .btn { display: block; width: 100%; margin-top: 1.5rem; padding: .75rem; border: none;
         border-radius: 8px; background: #4f7cff; color: #fff; font-size: 1rem; font-weight: 600;
         cursor: pointer; }
  .btn:disabled { background: #a9bbe8; cursor: not-allowed; }
  .btn:hover:not(:disabled) { background: #3d64e0; }
  .error { margin-top: 1rem; padding: .75rem 1rem; background: #fef2f2; color: #b91c1c;
           border-radius: 8px; font-size: .88rem; }
  .working { display: none; margin-top: 1rem; font-size: .88rem; color: #475569; }
  .hint { margin-top: 1.5rem; font-size: .78rem; color: #94a3b8; line-height: 1.5; }
</style>
</head>
<body>
  <div class="card">
    <div class="top">
      <h1>Submit a Ticket</h1>
      <a href="/">&larr; Back</a>
    </div>
    <p class="sub">Manually add a ticket outside of MongoDB - it's embedded and clustered
      immediately, joining an existing recurring issue if it matches one, so you can see
      phase 5 automation react to a new ticket without waiting on a live feed.</p>

    {% if error %}<div class="error">{{ error }}</div>{% endif %}

    <form id="f" action="/submit" method="post" enctype="multipart/form-data">
      <div class="row2">
        <div>
          <label for="domain">Domain <span class="req">*</span></label>
          <select id="domain" name="domain" required>
            <option value="" disabled {% if not form.domain %}selected{% endif %}>Select domain</option>
            {% for d in domains %}
            <option value="{{ d }}" {% if form.domain == d %}selected{% endif %}>{{ d }}</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label for="category">Category <span class="req">*</span></label>
          <select id="category" name="category" required>
            <option value="" disabled {% if not form.category %}selected{% endif %}>Select category</option>
            {% for c in categories %}
            <option value="{{ c }}" {% if form.category == c %}selected{% endif %}>{{ c }}</option>
            {% endfor %}
          </select>
        </div>
      </div>

      <label for="priority">Priority</label>
      <select id="priority" name="priority">
        {% for p in priorities %}
        <option value="{{ p }}" {% if (form.priority or "Medium") == p %}selected{% endif %}>{{ p }}</option>
        {% endfor %}
      </select>


      <label for="subject">Subject <span class="req">*</span></label>
      <input type="text" id="subject" name="subject" value="{{ form.subject or '' }}" maxlength="200" required>

      <label for="description">Description <span class="req">*</span></label>
      <textarea id="description" name="description" required>{{ form.description or '' }}</textarea>

      <label for="submitted_by">Ticket Submitted By</label>
      <input type="text" id="submitted_by" name="submitted_by" value="{{ form.submitted_by or '' }}"
             placeholder="Name or department (optional)">

      <label for="attachments">Attachments / Screenshots</label>
      <input type="file" id="attachments" name="attachments" multiple>

      <button class="btn" id="go" type="submit">Submit Ticket</button>
      <div class="working" id="working">Submitting... embedding and re-clustering against the
        existing dataset, this can take a moment.</div>
    </form>

    <div class="hint">Stored locally and processed the same way phase 5's incremental
      MongoDB poll handles a new ticket - no MongoDB connection required.</div>
  </div>

<script>
  const form = document.getElementById('f');
  const go = document.getElementById('go');
  const working = document.getElementById('working');
  form.addEventListener('submit', () => {
    go.disabled = true;
    go.textContent = 'Submitting...';
    working.style.display = 'block';
  });
</script>
</body>
</html>
"""


def _render_submit(error: str | None = None, form: dict | None = None):
    return render_template_string(
        SUBMIT_PAGE,
        error=error,
        form=form or {},
        domains=DOMAIN_OPTIONS,
        categories=CATEGORY_OPTIONS,
        priorities=PRIORITY_OPTIONS,
    )


@app.get("/submit")
def submit_page():
    return _render_submit()


@app.post("/submit")
def submit_ticket():
    form = {
        "domain": (request.form.get("domain") or "").strip(),
        "category": (request.form.get("category") or "").strip(),
        "priority": (request.form.get("priority") or "Medium").strip(),
        "subject": (request.form.get("subject") or "").strip(),
        "description": (request.form.get("description") or "").strip(),
        "submitted_by": (request.form.get("submitted_by") or "").strip(),
    }

    if not form["domain"] or not form["category"] or not form["subject"] or not form["description"]:
        return _render_submit(
            error="Domain, Category, Subject, and Description are required.", form=form
        ), 400

    now = datetime.now(timezone.utc)
    ticket_no = f"MANUAL-{now:%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"

    files = [f for f in request.files.getlist("attachments") if f and f.filename]
    if files:
        dest_dir = ATTACHMENTS_DIR / ticket_no
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            f.save(dest_dir / Path(f.filename).name)

    ticket = {
        "ticket_no": ticket_no,
        "subject": form["subject"],
        "description": form["description"],
        "domain": form["domain"],
        # The category actually selected on the form - this is what
        # determines whether the ticket can match a real historical repeat
        # from that same category. See src/classify.py's module docstring
        # for the measured impact of getting this wrong (proven with real
        # data: a mobile-number ticket needed "Easychit", an invoice-date
        # ticket needed "PRECAST" - no single default covers both).
        "category": form["category"],
        "priority": form["priority"].lower(),
        "status": "open",
        "department": form["submitted_by"] or "unspecified",
        "company": None,
        "created_at": now,
        "closed_at": None,
    }

    try:
        add_manual_ticket(ticket)
        report_path = generate_report()
    except Exception as exc:  # surface the real error instead of a blank 500
        traceback.print_exc()
        return _render_submit(error=f"Failed to process ticket: {exc}", form=form), 500

    # Look up how the ticket it just processed was classified, so the
    # dashboard can open with a clear "matched an existing issue" vs.
    # "new, unrepeated issue" confirmation instead of silently refreshing.
    outcome, issue, count = "new", "", 1
    try:
        tickets_index = json.loads(settings.tickets_path.read_text(encoding="utf-8"))
        entry = next((t for t in tickets_index if t["ticket_no"] == ticket_no), None)
        if entry and (entry.get("cluster_size") or 1) >= 2:
            outcome, issue, count = "matched", entry.get("cluster_issue") or "", entry["cluster_size"]
    except Exception:
        traceback.print_exc()

    query = urlencode({"submitted": ticket_no, "outcome": outcome, "issue": issue, "count": count})
    return redirect(f"/reports/{Path(report_path).name}?{query}")


ASK_PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ask AI - Ticket Repetition Analyzer</title>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Arial, sans-serif; background: #f4f5f7;
         margin: 0; padding: 2rem 1rem; color: #1f2430; }
  .wrap { max-width: 720px; margin: 0 auto; }
  .top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; }
  .top a { font-size: .85rem; color: #4f7cff; text-decoration: none; font-weight: 600; }
  .top a:hover { text-decoration: underline; }
  h1 { font-size: 1.2rem; margin: 0; }
  p.sub { color: #64748b; margin: .35rem 0 1.25rem; font-size: .88rem; line-height: 1.5; }
  .card { background: #fff; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.08), 0 8px 24px rgba(0,0,0,.06);
          display: flex; flex-direction: column; height: 65vh; overflow: hidden; }
  .msgs { flex: 1; overflow-y: auto; padding: 1.25rem; display: flex; flex-direction: column; gap: .9rem; }
  .msg { max-width: 85%; padding: .65rem .9rem; border-radius: 10px; font-size: .9rem; line-height: 1.5; white-space: pre-wrap; }
  .msg.user { align-self: flex-end; background: #4f7cff; color: #fff; }
  .msg.bot { align-self: flex-start; background: #f1f5f9; color: #1f2430; }
  .msg.bot.error { background: #fef2f2; color: #b91c1c; }
  .sources { margin-top: .5rem; font-size: .74rem; color: #64748b; }
  .sources span { display: inline-block; background: #e2e8f0; border-radius: 4px; padding: .1rem .4rem; margin: .15rem .2rem 0 0; }
  .empty { margin: auto; text-align: center; color: #94a3b8; font-size: .88rem; padding: 1rem; }
  form { display: flex; gap: .6rem; padding: .9rem; border-top: 1px solid #e2e8f0; }
  input[type=text] { flex: 1; padding: .65rem .8rem; border: 1px solid #cbd5e1; border-radius: 8px; font-size: .9rem; }
  button { padding: .65rem 1.1rem; border: none; border-radius: 8px; background: #4f7cff; color: #fff;
           font-size: .9rem; font-weight: 600; cursor: pointer; }
  button:disabled { background: #a9bbe8; cursor: not-allowed; }
  .examples { margin-top: .8rem; font-size: .8rem; color: #64748b; }
  .examples button { all: unset; cursor: pointer; color: #4f7cff; text-decoration: underline; }
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <h1>Ask AI about your ticket data</h1>
    <a href="/">&larr; Back</a>
  </div>
  <p class="sub">Answers are grounded in the last analyzed dataset (dashboard KPIs/breakdowns
    plus the tickets most relevant to your question) - not general knowledge.</p>
  <div class="card">
    <div class="msgs" id="msgs">
      <div class="empty" id="empty">
        Try: "What are the top 5 repeating issues?" or "How many network tickets repeated
        in the Finance department?"
      </div>
    </div>
    <form id="f">
      <input type="text" id="q" placeholder="Ask a question about the analyzed tickets..." autocomplete="off" required>
      <button id="go" type="submit">Ask</button>
    </form>
  </div>
</div>
<script>
  const msgs = document.getElementById('msgs');
  const empty = document.getElementById('empty');
  const form = document.getElementById('f');
  const input = document.getElementById('q');
  const go = document.getElementById('go');
  let history = [];

  function addMsg(role, text, sources) {
    if (empty) empty.remove();
    const div = document.createElement('div');
    div.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
    div.textContent = text;
    if (sources && sources.length) {
      const s = document.createElement('div');
      s.className = 'sources';
      s.textContent = 'Sources: ';
      sources.forEach(src => {
        const tag = document.createElement('span');
        tag.textContent = src.ticket_no;
        tag.title = src.subject;
        s.appendChild(tag);
      });
      div.appendChild(s);
    }
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    return div;
  }

  function addError(text) {
    if (empty) empty.remove();
    const div = document.createElement('div');
    div.className = 'msg bot error';
    div.textContent = text;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    addMsg('user', question);
    history.push({ role: 'user', content: question });
    input.value = '';
    go.disabled = true;
    go.textContent = 'Thinking...';
    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, history: history.slice(0, -1) }),
      });
      const data = await res.json();
      if (!res.ok) {
        addError(data.error || 'Something went wrong.');
      } else {
        addMsg('bot', data.answer, data.sources);
        history.push({ role: 'assistant', content: data.answer });
      }
    } catch (err) {
      addError('Network error: ' + err);
    } finally {
      go.disabled = false;
      go.textContent = 'Ask';
      input.focus();
    }
  });
</script>
</body>
</html>
"""


@app.get("/ask")
def ask_page():
    return render_template_string(ASK_PAGE)


@app.post("/api/ask")
def api_ask():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    history = payload.get("history") or []
    if not question:
        return jsonify({"error": "Question is required."}), 400
    try:
        result = answer_question(question, history=history)
    except RagUnavailable as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # surface the real error instead of a blank 500
        traceback.print_exc()
        return jsonify({"error": f"Something went wrong: {exc}"}), 500
    return jsonify(result)


if __name__ == "__main__":
    start_auto_refresh_if_configured()
    app.run(debug=DEBUG, port=5000, use_reloader=USE_RELOADER)
