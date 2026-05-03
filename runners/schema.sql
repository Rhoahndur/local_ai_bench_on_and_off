-- Canonical schema for runs.sqlite (eval results + memory layer).
-- Idempotent: safe to re-run via `python -m runners.db`.

CREATE TABLE IF NOT EXISTS eval_runs (
  run_id          TEXT,
  parent_run_id   TEXT,
  timestamp       TEXT,
  model           TEXT,
  tier            TEXT,
  domain          TEXT,
  track           TEXT,
  prompt_id       TEXT,
  ctx_size        INTEGER,
  tokens_in       INTEGER,
  tokens_out      INTEGER,
  cold_load_ms    REAL,
  ttft_ms         REAL,
  latency_ms      REAL,
  gen_duration_ms REAL,
  tok_per_sec     REAL,
  peak_ram_mb     REAL,
  output          TEXT,
  score           REAL,
  scoring_method  TEXT,
  judge_model     TEXT,
  error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_eval_runs_run_id   ON eval_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_runs_model    ON eval_runs(model);
CREATE INDEX IF NOT EXISTS idx_eval_runs_prompt   ON eval_runs(prompt_id);
CREATE INDEX IF NOT EXISTS idx_eval_runs_ts       ON eval_runs(timestamp);

CREATE TABLE IF NOT EXISTS perf_sustained (
  model                    TEXT,
  ts                       TEXT,
  run_index                INTEGER,
  ctx_size                 INTEGER,
  tokens_out               INTEGER,
  gen_duration_ms          REAL,
  tok_per_sec              REAL,
  throttle_pct_vs_first    REAL,
  peak_ram_mb              REAL
);

CREATE TABLE IF NOT EXISTS perf_dimensional (
  model           TEXT,
  ts              TEXT,
  ctx_size        INTEGER,
  tokens_in       INTEGER,
  tokens_out      INTEGER,
  cold_load_ms    REAL,
  ttft_ms         REAL,
  gen_duration_ms REAL,
  latency_ms      REAL,
  tok_per_sec     REAL,
  peak_ram_mb     REAL
);
CREATE INDEX IF NOT EXISTS idx_perf_dim_model ON perf_dimensional(model);
CREATE INDEX IF NOT EXISTS idx_perf_sus_model ON perf_sustained(model);

-- Memory layer (used by swap_model.py)
CREATE TABLE IF NOT EXISTS sessions (
  session_id    TEXT PRIMARY KEY,
  created_at    TEXT,
  active_model  TEXT
);

CREATE TABLE IF NOT EXISTS turns (
  turn_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  TEXT,
  role        TEXT,
  content     TEXT,
  tokens      INTEGER,
  ts          TEXT
);

CREATE TABLE IF NOT EXISTS summaries (
  session_id     TEXT,
  up_to_turn     INTEGER,
  summary        TEXT,
  embedding      BLOB,
  ts             TEXT
);

CREATE TABLE IF NOT EXISTS memory_objects (
  id             TEXT PRIMARY KEY,
  type           TEXT,
  session_id     TEXT,
  date           TEXT,
  project        TEXT,
  summary        TEXT,
  evidence_json  TEXT,
  recheck        TEXT,
  embedding      BLOB
);

-- Note: vec_memory virtual table is created lazily by runners/db.py:ensure_vec_table()
-- once the sqlite-vec extension is loaded. It is not part of this schema because the
-- extension is optional for non-memory workflows.
