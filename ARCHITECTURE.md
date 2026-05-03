# local-model-bench — Architecture (engineering handoff)

Three diagrams, each targeting a different concern. Mermaid renders natively on GitHub, GitLab, VS Code (with extension), Notion, Obsidian, Mermaid Live Editor, etc.

---

## 1. System Architecture (component view)

The static picture: what exists, what owns what, who talks to whom.

```mermaid
graph TB
    subgraph UI["User Interface"]
        CLI["CLI commands<br/>run_bench / compare / promote / swap_model"]
        DASH["Dashboards<br/>leaderboard.md, model_cards/"]
    end

    subgraph ORCH["Orchestration (Python)"]
        RB[run_bench.py]
        CMP[compare.py]
        PRM[promote.py]
        SWP[swap_model.py]
    end

    subgraph RUN["Eval Runners"]
        RO["run_ollama.py<br/>streaming caller + metrics"]
        MP["measure_perf.py<br/>cold load, throughput, throttle"]
        AL["agent_loop.py<br/>Track B: tool-use"]
        DR["deep_reasoning.py<br/>Track C: multi-stage"]
        JO["judge_outputs.py<br/>rubric scoring"]
    end

    subgraph CFG["Configs (YAML)"]
        MOD[models.yaml]
        PF[promptfooconfig.yaml]
        EV["evals/<br/>capability_prompts<br/>code_tasks<br/>injection_tasks<br/>vision_tasks<br/>rag_tasks"]
    end

    subgraph OLLAMA["Ollama Runtime (localhost:11434)"]
        OL["HTTP API<br/>/api/generate, /api/embed"]
    end

    subgraph TIERA["Tier A — 7-8B baseline"]
        M1["llama3.1:8b-instruct-q4_K_M<br/>4.9GB · 128K ctx"]
        M2["qwen2.5-coder:7b<br/>4.7GB · 32K ctx"]
        M3["phi3.5:3.8b-mini-instruct-q4_K_M<br/>2.4GB · 128K ctx"]
    end

    subgraph TIERB["Tier B — 3-4B small + new"]
        M4["gemma3:4b<br/>3.3GB · 128K · vision"]
        M5["qwen3:4b<br/>2.5GB · 256K · thinking"]
        M6["phi4-mini:3.8b<br/>2.5GB · 128K"]
    end

    subgraph EMB["Embeddings"]
        E1["nomic-embed-text<br/>270MB"]
    end

    subgraph STORE["Storage"]
        DB[("SQLite<br/>runs.sqlite")]
        VEC[("sqlite-vec<br/>vector index")]
        FS["filesystem<br/>artifacts/, memory/notes/"]
    end

    CLI --> RB & CMP & PRM & SWP
    RB --> RO & MP & AL & DR
    RO --> JO
    MOD --> RB
    EV --> RO & AL & DR
    PF --> RO

    RO -->|HTTP| OL
    MP -->|HTTP| OL
    AL -->|HTTP| OL
    DR -->|HTTP| OL
    JO -->|HTTP| OL
    SWP -->|HTTP| OL

    OL --> M1 & M2 & M3 & M4 & M5 & M6
    OL --> E1

    RO --> DB
    MP --> DB
    AL --> DB
    DR --> DB
    JO --> DB
    E1 --> VEC
    SWP --> FS

    DB --> CMP & PRM
    VEC --> SWP
    CMP --> DASH
    PRM --> DASH

    classDef tierA fill:#e1f5ff,stroke:#0277bd
    classDef tierB fill:#fff4e1,stroke:#f57c00
    classDef storage fill:#f3e5f5,stroke:#7b1fa2
    class M1,M2,M3 tierA
    class M4,M5,M6 tierB
    class DB,VEC,FS storage
```

### Key invariants for the engineer

- **Runners are stateless.** All state lives in SQLite + filesystem. Any runner can be killed and restarted; the harness picks up from the next un-run row.
- **Ollama is the only model boundary.** Swapping to MLX or llama.cpp later means changing one HTTP client, not the harness.
- **Configs drive everything.** Adding a model = one entry in `models.yaml`. Adding an eval = one YAML file under `evals/`. No code changes.
- **Two database concerns, one file.** `runs.sqlite` holds both eval results and memory-layer data. `sqlite-vec` is an extension loaded into the same DB, not a separate store.
- **`OLLAMA_KEEP_ALIVE=0` is required.** Without it, sequential model calls thrash memory because Ollama keeps the previous model resident for 5 minutes by default.

---

## 2. Single eval-run lifecycle (sequence)

What actually happens when you run `python run_bench.py --models all --suite full`. Engineers debugging "why did this score low" should follow this path.

```mermaid
sequenceDiagram
    participant U as User/CLI
    participant RB as run_bench.py
    participant RO as run_ollama.py
    participant OL as Ollama API
    participant M as Model under test
    participant JO as judge_outputs.py
    participant J as Judge model<br/>(phi4-mini)
    participant DB as SQLite

    U->>RB: run_bench --models all --suite full
    RB->>RB: load models.yaml + evals/*.yaml
    RB->>DB: insert run_id, started_at

    loop for each (model, prompt) pair
        RB->>RO: call(model, prompt, ctx_size)
        RO->>OL: POST /api/generate (stream=true, keep_alive=0)
        OL->>M: load weights if not resident
        M-->>OL: stream tokens
        OL-->>RO: SSE chunks
        RO->>RO: track TTFT, tok/s, peak RAM (psutil)
        RO->>DB: insert raw output + perf metrics

        alt deterministic task (code, JSON, math)
            RO->>RO: exact-match / unit-test / parse check
            RO->>DB: write score
        else open-ended task
            RO->>JO: grade(prompt, rubric, output)
            JO->>OL: POST /api/generate (judge model)
            OL-->>JO: 0-5 score
            JO->>DB: write score
        end
    end

    RB->>DB: aggregate per-model tier
    RB->>U: write dashboards/leaderboard.md
```

### Key invariants

- **One row per inference.** Even within a multi-stage task (deep reasoning), each stage is its own row with a `parent_run_id`. This makes failure isolation possible.
- **Streaming is non-optional for metrics.** TTFT can only be measured if you read the response stream incrementally. Don't refactor `run_ollama.py` to use `stream=false`.
- **Judge calls are sequential, not parallel.** The judge model can be loaded simultaneously with small models (Phi-4-mini ~2.5GB) but conflicts with 8B models on this 16GB machine. Batch all 8B-under-test calls first, then run the judge pass.
- **Failures are logged, not raised.** A model crashing on prompt N must not abort prompts N+1..M. Wrap each call in try/except and write `score=null, error="..."` to the row.

---

## 3. Model-swap with memory retention

The differentiating capability: continue work on a project after switching models. This is also the part most likely to subtly break — engineers should pay attention to the embedding compatibility issue noted below.

```mermaid
flowchart TB
    START[/"User: swap_model.py --session abc --to qwen3:4b"/]

    subgraph LOAD["Load prior context"]
        Q1["query SQLite:<br/>SELECT turns WHERE session_id=abc"]
        Q2["query SQLite:<br/>SELECT summaries WHERE session_id=abc<br/>ORDER BY up_to_turn DESC LIMIT 1"]
        Q3["sqlite-vec query:<br/>top-K similar memory_objects<br/>by recent-turn embedding"]
    end

    subgraph BUILD["Build context preamble"]
        B1["Recent summary<br/>(last rolling summary)"]
        B2["Earlier highlights<br/>(top-K vector matches)"]
        B3["Structured facts<br/>(decisions, constraints, entities)"]
        B4["Compose system prompt:<br/>[CONTEXT FROM PRIOR WORK]<br/>...<br/>[/CONTEXT]"]
    end

    subgraph ACTIVATE["Switch active model"]
        A1["UPDATE sessions<br/>SET active_model='qwen3:4b'"]
        A2["Warm new model:<br/>POST /api/generate with preamble<br/>(discard output, just prime KV)"]
    end

    subgraph CONTINUE["Continue work"]
        C1["User issues next turn"]
        C2["run_ollama with new model<br/>+ preamble + new turn"]
        C3["INSERT new turn"]
        C4{"Turn count<br/>since last summary<br/>>= N?"}
        C5["Generate new rolling summary<br/>(active model summarizes recent N turns)"]
        C6["Embed summary via nomic-embed-text"]
        C7["INSERT into summaries + vectors"]
    end

    START --> LOAD
    Q1 --> B3
    Q2 --> B1
    Q3 --> B2
    B1 & B2 & B3 --> B4
    B4 --> A1 --> A2
    A2 --> C1 --> C2 --> C3 --> C4
    C4 -->|yes| C5 --> C6 --> C7 --> C1
    C4 -->|no| C1

    classDef warn fill:#ffebee,stroke:#c62828
    class A2 warn
```

### Key invariants and gotchas

- **Embedding model must be stable across swaps.** All vectors in `sqlite-vec` were generated with `nomic-embed-text`. If you swap the embedding model, you must re-embed all summaries — vectors from different embedding models are not comparable. This is the most common silent bug.
- **The new model never sees raw prior turns** — only the compressed preamble. This is intentional (cheaper, model-agnostic) but means the new model has no fine-grained recall. If you need it, query SQLite directly per turn.
- **Warm-up call is real, not cosmetic.** Models cold-start slow on M-series; the throwaway call gets the weights loaded and the KV cache primed before the user feels latency.
- **Summaries are lossy.** Don't depend on them for facts that need to survive verbatim — those go in `memory_objects` as structured records (decisions, constraints, entities). Two-track memory.
- **Session IDs are externally generated.** The harness doesn't auto-create sessions. Caller (CLI or wrapper script) supplies the ID.

---

## 4. SQLite schema (canonical)

The diagrams above reference these tables. Engineer should treat this as the source of truth.

```sql
-- Eval results
CREATE TABLE eval_runs (
  run_id TEXT,
  parent_run_id TEXT,           -- for multi-stage tasks
  timestamp TEXT,
  model TEXT,
  tier TEXT,
  domain TEXT,                   -- 'math', 'code', 'json', 'vision', etc.
  track TEXT,                    -- 'hitl', 'agent', 'deep_reasoning'
  prompt_id TEXT,
  ctx_size INTEGER,
  tokens_in INTEGER,
  tokens_out INTEGER,
  cold_load_ms INTEGER,
  ttft_ms INTEGER,
  latency_ms INTEGER,
  tok_per_sec REAL,
  peak_ram_mb INTEGER,
  output TEXT,                   -- raw model output
  score REAL,                    -- 0-1 normalized or 0-5 rubric/5
  scoring_method TEXT,           -- 'exact', 'unit_test', 'rubric', 'parse'
  judge_model TEXT,              -- if rubric
  error TEXT
);

-- Sustained throughput tests
CREATE TABLE perf_sustained (
  model TEXT,
  ts TEXT,
  run_index INTEGER,             -- 1..5
  tok_per_sec REAL,
  throttle_pct_vs_first REAL
);

-- Memory layer
CREATE TABLE sessions (
  session_id TEXT PRIMARY KEY,
  created_at TEXT,
  active_model TEXT
);

CREATE TABLE turns (
  turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  role TEXT,
  content TEXT,
  tokens INTEGER,
  ts TEXT
);

CREATE TABLE summaries (
  session_id TEXT,
  up_to_turn INTEGER,
  summary TEXT,
  embedding BLOB,
  ts TEXT
);

CREATE TABLE memory_objects (
  id TEXT PRIMARY KEY,
  type TEXT,                     -- 'decision' | 'fact' | 'constraint' | 'entity'
  session_id TEXT,
  date TEXT,
  project TEXT,
  summary TEXT,
  evidence_json TEXT,
  recheck TEXT,
  embedding BLOB
);

-- sqlite-vec virtual table (loaded via .load extension)
CREATE VIRTUAL TABLE vec_memory USING vec0(
  embedding float[768]           -- nomic-embed-text dimension
);
```

---

## 5. Where things plug in

For an engineer asked "where do I add X":

| Adding | Where it goes |
|---|---|
| New model | `models.yaml` only; no code change |
| New eval domain | `evals/<domain>.yaml` + scoring helper if non-standard |
| New scoring method | `runners/judge_outputs.py` — extend the `grade()` dispatcher |
| New track (4th mode) | New runner in `runners/`, register in `run_bench.py` |
| New backend (e.g. MLX) | `runners/run_ollama.py` becomes `run_backend.py` with a strategy interface; rest of stack unchanged |
| New dashboard view | `dashboards/` — read from SQLite, write markdown |
| New memory object type | `memory_objects.type` is already free-text; just start writing new types |

The boundary discipline is: **runners own model interaction, configs own what to test, SQLite owns everything else.**
