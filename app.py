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
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_from_directory

from src.config import settings
from src.generate_report import generate_report
from src.pipeline import run_pipeline
from src.rag_qa import RagUnavailable, answer_question

app = Flask(__name__)
DEBUG = True  # single source of truth - also read by the auto-refresh startup guard below

UPLOAD_DIR = settings.data_dir / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".json", ".jsonl", ".bson", ".csv"}

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
    # Flask's debug reloader runs this module twice (a parent that only
    # watches files, and a child that actually serves requests) - only the
    # child should start the poller, or new tickets would get processed twice.
    if DEBUG and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
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
    app.run(debug=DEBUG, port=5000)
