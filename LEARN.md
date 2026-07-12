# LEARN.md — Understand Every Line of Sentinel

This is a from-zero course on everything Sentinel touches. It assumes you can
read basic Python and nothing else. Each section explains a concept in plain
language, shows **exactly where it lives in this codebase**, and recommends a
video (search the title on YouTube — channels are named so you find the right
one).

**How to use this file:** read Part 1 first, then jump around. The fastest way
to learn is to keep `python run_demo.py` running in one terminal and poke the
code while the dashboard reacts.

---

## Part 1 — The big picture: why models need a control room

Training a model is like tuning a guitar in your bedroom. Production is the
concert: temperature changes, strings stretch, and the tuning silently goes
off. Nothing "crashes" — it just sounds worse.

Concretely, a model learns patterns from a **snapshot** of the world (the
training set). When the live world stops matching that snapshot, the model's
answers degrade. This mismatch is called **drift**, and it has three flavors:

1. **Data drift (covariate drift)** — the *inputs* change shape. Example: your
   customers' average age was 40 at training time; now it's 28. Detectable
   **without knowing any right answers** — you just compare distributions.
2. **Concept drift** — the *relationship* between inputs and outputs changes.
   Same customers, but fraudsters invented a new trick. Detectable only by
   watching the model's error rate move.
3. **Label drift** — the mix of outcomes changes (fraud rate doubles).

The cruel part: in production you usually can't compute accuracy at prediction
time because the true answer (did the loan default?) arrives weeks later.
That's why data drift matters so much — it's the only early-warning signal
that needs zero labels. This entire codebase is organized around that fact.

📺 *"MLOps explained" — Google Cloud Tech*; *"Data drift vs concept drift"*
(several good short explainers exist — watch one of each).

Where in the code: the three flavors map to
`DriftDetector.detect_covariate_drift`, `ModelMonitor._update_concept_drift`,
and `DriftDetector.detect_label_drift` in [sentinel/core/drift.py](sentinel/core/drift.py)
and [sentinel/core/monitor.py](sentinel/core/monitor.py).

---

## Part 2 — The statistics (the part interviewers love)

### 2.1 The p-value, honestly explained

Every statistical test in Sentinel asks the same question: *"could this
difference be a coincidence?"* The **p-value** is the probability of seeing a
difference at least this large **if nothing actually changed**. Tiny p-value
(< 0.05) = "almost certainly not a coincidence" = drift.

One trap that actually bit this project (see git history): when nothing has
changed, p-values are **uniformly random between 0 and 1** — they don't sit
near 1. So `1 − p` is NOT a "drift score"; on average it's 0.5 even for
identical data. Sentinel v0.1 treated it like a score and cried wolf
constantly. The fix: real tests make the drifted/not-drifted decision;
the blended score is only for dashboards. That story is told by the code in
`DriftDetector._analyse_feature` ([drift.py](sentinel/core/drift.py) — note
`is_drifted` uses `p_val < self.ks_threshold` or `psi > self.psi_threshold`,
never the blended score).

📺 *"p-values clearly explained" — StatQuest with Josh Starmer* (the single
best stats channel on the internet; watch his hypothesis-testing video too).

### 2.2 Kolmogorov–Smirnov (KS) test — for continuous features

Take 1,000 training values of `income` and 500 production values. Sort each,
draw their **cumulative curves** (x = income, y = fraction of samples ≤ x).
The KS statistic is the **biggest vertical gap** between the two curves.
Big gap → different distributions. scipy does the math:
`stats.ks_2samp(reference, production)` in `DriftDetector._ks_test`.

### 2.3 Chi-square test — for categorical features

`country_code` has values like {0, 1, 2} — cumulative curves make no sense.
Instead, count how many samples fall in each category, and compare observed
vs expected counts. Sentinel auto-picks this test when a feature has ≤ 10
unique values (`n_unique <= 10` in `_analyse_feature`), and scales the
production counts so both sides total the same (the test requires it).

### 2.4 PSI — the "how big is the shift" number

The Population Stability Index comes from credit-risk modeling. Cut the value
range into 10 bins; for each bin compare the *fraction* of baseline vs
production samples: `PSI = Σ (prod% − base%) × ln(prod% / base%)`.
Industry rule of thumb, worth saying in an interview: **< 0.1 stable ·
0.1–0.2 keep watching · > 0.2 investigate now**. Unlike KS, PSI doesn't get
oversensitive with huge samples — that's why Sentinel uses both.
Code: `_compute_psi` in [drift.py](sentinel/core/drift.py).

### 2.5 Jensen–Shannon divergence — a third opinion

A symmetric, always-finite cousin of KL divergence, bounded [0, 1]. It reads
the *entire shape* of the two histograms rather than one worst gap.
Code: `_jensen_shannon_divergence`.

### 2.6 ADWIN — catching change in a stream

Data drift compares snapshots. ADWIN watches a **stream** (here: the model's
error rate — 0 for right, 1 for wrong). It keeps a window of recent values and
repeatedly asks: "is there a split point where the average before differs
from the average after by more than chance allows?" "More than chance allows"
is the **Hoeffding bound** — a formula for the biggest gap two same-source
averages could plausibly show. Exceed it → change detected → drop the stale
half of the window. Reference: Bifet & Gavaldà, 2007.
Code: class `ADWIN` in [drift.py](sentinel/core/drift.py) — the bound is the
`epsilon_cut` line.

### 2.7 Page-Hinkley — the tripwire

A 1950s quality-control classic: accumulate how far each new value sits above
a running mean; if that cumulative sum climbs more than λ=50 above its
lowest-ever point, the mean has shifted upward. O(1) per observation.
Code: class `PageHinkley`.

Both stream detectors are fed live — every labeled prediction — in
`ModelMonitor._update_concept_drift` ([monitor.py](sentinel/core/monitor.py)).

📺 *"Kullback-Leibler divergence" and "Covariate shift" — StatQuest*;
for streaming drift, search *"concept drift detection ADWIN explained"*.

---

## Part 3 — Python concepts this codebase uses

Each of these is a 10-minute lesson using code you already have open:

- **Dataclasses** — `@dataclass` writes `__init__`/`__repr__` for you.
  `PredictionRecord` ([storage/backend.py](sentinel/storage/backend.py)),
  `DriftReport`, `Alert` (note `frozen=True` = immutable — an alert is a
  historical fact, nobody should edit it).
  📺 *"Python dataclasses" — mCoding or ArjanCodes.*
- **Abstract base classes** — `StorageBackend(ABC)` declares six
  `@abstractmethod`s; `InMemoryStorage` and `SQLiteStorage` implement them.
  Callers depend on the interface, so a Postgres backend is a drop-in later.
  This is the "D" in SOLID (dependency inversion).
  📺 *"Protocols and ABCs in Python" — ArjanCodes.*
- **Threading & locks** — the demo logs predictions from a background thread
  *while* FastAPI serves reads. `InMemoryStorage` wraps every mutation in
  `threading.RLock`; `AlertManager` locks its history list. Remove a lock and
  run the demo to see why they exist.
  📺 *"Python threading tutorial" — Corey Schafer.*
- **Enums** — `AlertSeverity(str, Enum)`: a fixed vocabulary that
  JSON-serializes as a plain string because it also inherits `str`.
- **`__getattr__` delegation** — the sklearn wrappers forward unknown
  attributes (`classes_`, `feature_importances_`) to the wrapped model:
  [integrations/sklearn.py](sentinel/integrations/sklearn.py). This is the
  proxy pattern in five lines.
- **Lazy / optional imports** — `prometheus_client` and `sqlalchemy` are
  imported inside `try` or inside `__init__` so the package still works
  without them ([core/metrics.py](sentinel/core/metrics.py) top,
  `SQLiteStorage.__init__`). Graceful degradation.
- **Type hints** — every signature; they're documentation that tools check.
  📺 *"Python 101: type hints" — mCoding.*

---

## Part 4 — The web layer

- **HTTP + REST in one paragraph:** the dashboard server is a program that
  answers URLs. `GET /api/drift` = "give me the drift report as JSON".
  `POST /api/log` = "here's data, store it". Status 200 = OK. That's most of
  REST.
- **FastAPI** — each endpoint is just a decorated async function in
  [dashboard/app.py](sentinel/dashboard/app.py). Pydantic's `PredictionLog`
  model auto-validates the POST body (send garbage and you get a clean 422).
  Free bonus: interactive docs at `/docs` — open it, it impresses people.
  📺 *"FastAPI course for beginners" — freeCodeCamp*, or
  *"FastAPI in 100 seconds" — Fireship* for the trailer version.
- **WebSockets** — a phone call instead of repeated text messages: the browser
  connects once to `/ws` and the server can push. When a prediction is logged
  via the API, `app.py` broadcasts `{"event": "prediction_logged"}` and the UI
  refreshes instantly. Look at `connectWS()` in
  [templates/index.html](sentinel/dashboard/templates/index.html) — note the
  3-second auto-reconnect loop.
  📺 *"WebSockets in 100 seconds" — Fireship.*
- **ASGI / uvicorn** — FastAPI describes endpoints; uvicorn is the engine that
  actually listens on the port and speaks HTTP.
- **CORS** — a browser safety rule about which sites may call your API.
  Sentinel currently allows all origins (`allow_origins=["*"]`) — fine for a
  demo, one line to tighten for real deployments.

The front-end has **no framework** — ~300 lines of vanilla JS: `fetch()` four
endpoints on timers, redraw Chart.js charts, flip CSS classes for the status
board. Read `pollSnapshot()` → `renderStatus()` top to bottom; that's the
whole architecture.

---

## Part 5 — Observability: Prometheus & Grafana

Prometheus flips monitoring around: instead of your app *sending* metrics
somewhere, it just exposes a text page (`/metrics` — open it, it's readable)
and Prometheus **scrapes** it every 15 s, storing every value as a time
series. Three metric types cover almost everything:

| Type | Behavior | Sentinel example |
|---|---|---|
| Counter | only goes up | `sentinel_predictions_total` |
| Gauge | goes up and down | `sentinel_drift_score` |
| Histogram | buckets a distribution | `sentinel_prediction_latency_seconds` |

Subtle detail worth knowing: [core/metrics.py](sentinel/core/metrics.py)
creates a **separate registry per collector** — the library throws
"duplicate metric" errors if two monitors register the same name globally.
Grafana is just the pretty face: it queries Prometheus with PromQL
(`rate(sentinel_predictions_total[1m])` = predictions per second) and draws.

📺 *"Prometheus explained" and "Grafana tutorial" — TechWorld with Nana.*

---

## Part 6 — Storage

Two backends, one interface:

- **InMemoryStorage** — a Python list + `{id: index}` dict behind a lock.
  Zero setup, capped at 50,000 records, gone when the process dies. Perfect
  default for demos and tests.
- **SQLiteStorage** — real persistence via **SQLAlchemy Core** (the SQL
  toolkit *without* the ORM — you build queries like
  `select(table).where(...)` and it emits SQL). Features are stored as JSON
  text columns; `model_name` and `timestamp` are indexed because every query
  filters on them.

One production-grade detail: SQLite drops timezone info on read, so
`_row_to_record` restores UTC-awareness — mixing naive and aware datetimes is
a classic Python crash.

📺 *"SQLAlchemy Core tutorial" — search it; also "SQL indexes explained".*

---

## Part 7 — The CLI and packaging

`setup.py` has `entry_points={"console_scripts": ["sentinel=sentinel.cli.commands:cli"]}`
— that one line is why typing `sentinel` in a terminal works after install.
The CLI is **Click** (each `@cli.command()` function becomes a subcommand,
`@click.option` handles flags) rendered with **Rich** (tables, colors).

Design choice worth quoting: the CLI doesn't compute anything — it calls the
dashboard's REST API (`SENTINEL_API` env var). One source of truth; the CLI
works against any running Sentinel, local or remote.

📺 *"Build a CLI with Click" — anthonywritescode / ArjanCodes both have one.*

---

## Part 8 — Docker & Compose

- **Image** ([docker/Dockerfile](docker/Dockerfile)): start from
  `python:3.11-slim`, copy requirements first (so Docker caches the slow
  pip-install layer), copy code, run the demo. Rebuilds after a code edit are
  seconds, not minutes — that layer-ordering trick is the thing to remember.
- **Compose** ([docker/docker-compose.yml](docker/docker-compose.yml)): three
  containers on one private network. The magic: containers reach each other
  **by service name** — Prometheus's config says `targets: ["sentinel:8080"]`
  and Docker's internal DNS resolves `sentinel`. No IPs anywhere.

📺 *"Docker in 100 seconds" — Fireship*, then
*"Docker Compose tutorial" — TechWorld with Nana.*

---

## Part 9 — CI: the robot that checks your homework

[.github/workflows/ci.yml](.github/workflows/ci.yml): on every push, GitHub
spins up Ubuntu **twice** (Python 3.10 and 3.11 — a version matrix), installs
`requirements-dev.txt`, runs `flake8` (style/lint) and `pytest` with coverage.
A red ✗ on a commit means it broke somewhere you didn't test locally.

The tests themselves teach a pattern: **seeded randomness**
(`np.random.default_rng(42)`) so "random" data is identical every run —
deterministic tests never flake. See [tests/test_drift.py](tests/test_drift.py):
same-distribution data must NOT drift, shifted data MUST.

📺 *"GitHub Actions tutorial" — Fireship or TechWorld with Nana.*

---

## Part 10 — Guided code-reading order (~90 minutes total)

1. `run_demo.py` (5 min) — the whole system assembled in 90 lines.
2. `sentinel/core/monitor.py` → `log_prediction()` (15 min) — the hot path;
   trace one prediction end to end.
3. `sentinel/core/drift.py` → `_analyse_feature()` (20 min) — the statistical
   heart. Then skim `ADWIN.add_element`.
4. `sentinel/core/alerts.py` → `AlertManager.evaluate_rules()` (10 min) —
   rules, cooldowns, fan-out to channels.
5. `sentinel/dashboard/app.py` (10 min) — every endpoint is ~10 lines.
6. `templates/index.html` → the `<script>` block (15 min) — how the control
   room stays live.
7. `sentinel/storage/backend.py` (10 min) — one interface, two backends.
8. `tests/` (5 min) — how each layer proves it works.

## Part 11 — Make it yours: exercises

Easy → hard. Each one is interview ammunition ("then I extended it to…"):

1. Add a `GET /api/version` endpoint. (10 min — touches FastAPI.)
2. Add a Discord alert channel (Discord webhooks are nearly Slack-shaped —
   subclass `BaseAlertChannel`).
3. Add a `sentinel report summary` CLI command hitting `/api/snapshot`.
4. Persist alerts into the storage backend so history survives restarts.
5. Add a `--fast` flag to `run_demo.py` (drift at 15 s) for demos.
6. New Prometheus metric: `sentinel_concept_drift` gauge (0/1).
7. Add AUC to classification metrics when `predict_proba` is available.
8. Multi-model dashboard: `set_monitor` registers into a dict; endpoints take
   `?model=`.
9. Write a `PostgresStorage` backend (SQLAlchemy makes it ~30 lines).
10. Drift on *embeddings*: monitor a sentence-transformer's output
    distribution for a text classifier — that's how the big platforms do
    unstructured drift.

---

*If you work through this file, you don't just "have a project" — you can
defend every design decision in it. That's the difference interviewers hear.*
