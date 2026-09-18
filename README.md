# Ticket Repetition Analyzer

Finds which support tickets are the same recurring issue reported again and
again in different words, and renders a static HTML dashboard showing how
many tickets point at each one. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for
the full architecture and reasoning.

## Setup

```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

Leave `.env` untouched to run against the bundled synthetic sample dataset,
or fill in `MONGO_URI` / `MONGO_DB` / `MONGO_COLLECTION` to point at your
real `tickets` collection.

## Run

### Option A — Upload a file from the browser

```powershell
.venv\Scripts\python app.py
```

Open http://127.0.0.1:5000, click the upload box, and pick a file exported
from your own MongoDB `tickets` collection:

```powershell
mongoexport --db=<db> --collection=tickets --jsonArray --out=tickets.json   # or:
mongoexport --db=<db> --collection=tickets --out=tickets.jsonl              # or:
mongodump   --db=<db> --collection=tickets --out=dump\                      # -> dump\<db>\tickets.bson
```

Click **Analyze** — the dashboard opens in the same tab when it's done. The
uploaded file is deleted from disk right after processing.

From either the upload page or the dashboard, click **Ask AI about this
data** to open a chat where you can ask natural-language questions like
*"What are the top 5 repeating issues?"* or *"How many network tickets
repeated in the Finance department?"* — answers are grounded in the last
analyzed dataset (see [Ask AI setup](#ask-ai-setup) below).

### Option B — Command line, live MongoDB connection or bundled sample

```powershell
.venv\Scripts\python run.py                  # auto: Mongo if configured, else sample data
.venv\Scripts\python run.py --source sample  # force the sample dataset
.venv\Scripts\python run.py --source mongo   # force MongoDB
.venv\Scripts\python run.py --limit 500      # quick test on the first 500 tickets
.venv\Scripts\python run.py --incremental    # phase 5: only pull/embed tickets newer than last run (MongoDB only)
```

This writes `data/clusters.json` (the computed clusters) and
`reports/report_<timestamp>.html` (the dashboard) - open the HTML file in
any browser.

To regenerate the synthetic sample dataset itself:

```powershell
.venv\Scripts\python scripts\generate_sample_data.py
```

## Ask AI setup

The dashboard's duplicate detection runs entirely offline (local embeddings),
but the natural-language **Ask AI** Q&A layer needs a cloud LLM call, since it
has to reason over retrieved tickets in free-form English. To enable it:

1. Get an API key from [console.anthropic.com](https://console.anthropic.com).
2. In `.env`, set:
   ```
   LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=sk-ant-...
   ```
3. Restart `app.py`. Run the pipeline at least once first (upload a file, or
   `python run.py`) — Ask AI answers from the last analyzed dataset.

Without a key, the rest of the app (upload, clustering, dashboard) still
works fully offline; `/ask` just shows a message asking you to configure it.

## Keeping the dashboard current (Phase 5)

From the upload page, click **+ Submit a new ticket** to add one ticket by
hand (domain, priority, subject, description, who submitted it, optional
attachments) - it's embedded and re-clustered against the existing dataset
immediately, no MongoDB connection required, which is the fastest way to see
phase 5's incremental logic react to a "new" ticket.

If you're pointed at a live MongoDB collection (`MONGO_URI`/`MONGO_DB` set),
the app can also pick up new tickets automatically:

- With `app.py` running, it polls MongoDB every `AUTO_REFRESH_INTERVAL_SECONDS`
  (`.env`, default 60s) and re-clusters + refreshes the report the moment new
  tickets show up. There's also a **Check for new tickets now** button on the
  upload page for an immediate check.
- Or run `python run.py --incremental` from a scheduled task/cron job if you
  don't want to keep `app.py` running continuously.

Either path only reads and embeds tickets newer than the last check - it
never re-embeds the whole collection, so it stays fast as the dataset grows.

## Tuning

- **`SIMILARITY_THRESHOLD`** (in `.env`, default `0.82`) controls how close
  two tickets' embeddings must be to count as "the same issue." Lower it if
  obvious duplicates are landing in separate clusters; raise it if unrelated
  tickets are getting merged. Check `data/clusters.json` after a run to see
  which way it needs to move.
- **`LLM_PROVIDER=anthropic`** (plus `ANTHROPIC_API_KEY`) turns on one Claude
  call per recurring cluster to generate a clean issue name from the sample
  subject lines, instead of just reusing the shortest one verbatim. Optional
  - the pipeline works fully offline without it.

## Project layout

```
app.py                 local web UI: upload a MongoDB export file, get the dashboard back
src/
  extract.py          MongoDB (or sample-file / uploaded-file) reader, schema normalization
  preprocess.py        text cleaning, embedding-input construction
  embed.py             local sentence-transformers embeddings
  vector_store.py      minimal on-disk vector store (numpy, no server)
  cluster.py           cosine-similarity + union-find grouping
  label.py             cluster naming (extractive, or optional LLM)
  pipeline.py           orchestrates the above, computes KPIs/trend
  generate_report.py    renders data/clusters.json into the static HTML report
  rag_qa.py             Ask AI: retrieval + Claude-backed Q&A over the analyzed data
templates/report_template.html   the dashboard shell the report is built from
scripts/generate_sample_data.py  synthetic dataset generator (demo/testing)
```
