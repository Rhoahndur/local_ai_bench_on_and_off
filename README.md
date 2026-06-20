# Local Model Bench

Role-based benchmarking harness for local LLMs running on Apple Silicon. Six Ollama models, five capability axes (HITL, agent loop, prompt injection, vision, sustained perf) plus a memory-layer / model-swap demo, measured on a 16 GB M3 Air.

The thesis: **local model selection should be role-based, not leaderboard-based.** A 7B code specialist beats an 8B "general baseline" on agent tasks. The same baseline is the *only* model in the set with perfect injection resistance. The smallest 3.8B compact is the surprise all-rounder. The right answer changes per use case.

See `dashboards/leaderboard.md` for the static matrix, or `streamlit run app.py` for the interactive view (overview / per-model / hardware-aware recommender).

## Models tested

Defined in `models.yaml`. Toggle `enabled` to control which runs.

| Model | Tier | Hypothesis | Disk | Ctx |
|---|---|---|---|---|
| `llama3.1:8b-instruct-q4_K_M` | A | general baseline | 4.9 GB | 128k |
| `qwen2.5-coder:7b` | A | code specialist | 4.7 GB | 32k |
| `phi3.5:3.8b-mini-instruct-q4_K_M` | A | compact reasoning, prev-gen | 2.4 GB | 128k |
| `gemma3:4b` | B | vision / multilingual | 3.3 GB | 128k |
| `qwen3:4b` | B | agent + thinking | 2.5 GB | 256k |
| `phi4-mini:3.8b` | B | compact reasoning, new-gen | 2.5 GB | 128k |

Plus `nomic-embed-text` (~270 MB) for the memory-layer embeddings.

## What's measured

| Track | What it tests | Runner |
|---|---|---|
| Capability probe | 5-prompt smoke: JSON output / math / code / strict-format / hallucination | `runners.run_bench` |
| Dimensional perf | tok/s + peak RAM at 1k / 4k / 8k context | `runners.measure_perf` |
| Sustained throttle | same prompt 5× with no cooldown — fanless thermal behaviour | `runners.measure_perf` |
| Agent loop (Track B) | mock filesystem tools, prompt-based JSON tool calling | `runners.agent_loop` |
| Prompt injection (§11) | 5 attack patterns scored by `not_contains` | `runners.run_bench --prompts-yaml evals/injection_tasks.yaml` |
| Vision | bar-chart QA, OCR, color recognition (gemma3 only) | `runners.vision_eval` |
| Memory / model swap | sessions, structured memory_objects, vector search via `sqlite-vec`, `[CONTEXT]` preamble | `swap_model.py` |

All results land in a single `results/runs.sqlite` (rows per inference for evals, separate tables for perf and memory). The leaderboard generator and Streamlit dashboard read from there.

## Setup

Requires Ollama 0.6+ (0.5.13+ for `phi4-mini`) and Python 3.10+.

```bash
# Ollama
brew install ollama
OLLAMA_KEEP_ALIVE=0 ollama serve &      # KEEP_ALIVE=0 → unload between calls

# Models — ~21 GB total. Stage them; don't blast all 7 in parallel on a tight SSD.
ollama pull qwen3:4b
ollama pull phi4-mini:3.8b
ollama pull nomic-embed-text
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
ollama pull gemma3:4b
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull qwen2.5-coder:7b

# Python
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Initialize the canonical schema
.venv/bin/python -m runners.db
```

## Quickstart

Bank a result in under two minutes:

```bash
# Capability smoke on the smallest fast model
.venv/bin/python -m runners.run_bench --only-model phi4-mini:3.8b

# Generate the markdown leaderboard from whatever's in the DB
.venv/bin/python -m runners.leaderboard
cat dashboards/leaderboard.md

# Or open the interactive dashboard
.venv/bin/streamlit run app.py     # http://localhost:8501
```

Full bench (~45 min on M3 Air with all models pulled):

```bash
.venv/bin/python -m runners.run_bench                                          # HITL probe (all enabled)
.venv/bin/python -m runners.measure_perf --all                                 # dim + sustained perf
.venv/bin/python -m runners.agent_loop --all                                   # mock-FS agent loop
.venv/bin/python -m runners.run_bench --prompts-yaml evals/injection_tasks.yaml --tag injection
.venv/bin/python -m runners.vision_eval                                        # gemma3 multimodal
.venv/bin/python -m runners.leaderboard                                        # regenerate leaderboard.md
```

## Memory layer / model swap

`swap_model.py` is a subcommand CLI on top of `runners/memory.py`. Sessions, turn log, structured memory_objects (decision/fact/constraint/entity), rolling summaries via the active model, vector retrieval through `sqlite-vec`. The headline operation is `swap`: it builds a compressed `[CONTEXT FROM PRIOR WORK]` preamble from the latest summary plus top-K vector-retrieved highlights plus all decisions/constraints, updates the session's active model, and warms the new model with `keep_alive=600` so the next user turn is fast.

```bash
# Seed a demo session with today's findings, then swap phi4-mini → qwen3
.venv/bin/python swap_model.py demo

# Or drive it manually
.venv/bin/python swap_model.py start --session abc --model phi4-mini:3.8b
.venv/bin/python swap_model.py turn --session abc --role user --content "fix the auth bug"
.venv/bin/python swap_model.py note --type decision --summary "use JWT not session cookies" --session abc
.venv/bin/python swap_model.py summarize --session abc --model phi4-mini:3.8b
.venv/bin/python swap_model.py search "auth" --k 3
.venv/bin/python swap_model.py swap --session abc --to qwen3:4b
```

The embedding model is fixed (`nomic-embed-text`, 768-dim) — see ARCHITECTURE.md §3 for the cross-model embedding-stability gotcha.

## Headline findings (M3 Air, 2026-05-03)

| | Pick | Why |
|---|---|---|
| Best all-rounder | `phi4-mini:3.8b` | 5/5 capability, Great agent loop in 4 steps, 4/5 injection resistance, lowest peak RAM at every ctx size |
| Best agent loop | `qwen2.5-coder:7b` | optimal 3-step solution — specialist beats same-size generalist |
| Best safety | `llama3.1:8b` | only model with 5/5 perfect injection resistance — slow (9 tok/s sustained) but the only safe choice for hostile inputs |
| Best multimodal | `gemma3:4b` | 3/3 on vision tasks at 22-38 tok/s; nothing else can attempt them |
| Don't pick | `phi3.5:3.8b` | `phi4-mini:3.8b` strictly dominates at the same parameter count |

Other findings worth keeping handy:
- **`qwen3:4b` thinking modes are not equivalent** — same JSON prompt scores 1.00 with default (Ollama strips `<think>`) and 0.00 with `think:false` (model inlines reasoning into the visible output).
- **Thermal throttling is order-dependent** on a fanless M3 — the first model in a sweep gets a cold-system bonus the last one doesn't. `phi4-mini` peaked at 30 tok/s mid-sustained then dropped to 14.
- **`phi3.5` hallucinated agent output** — invented a TODO line that wasn't in the source notes. Worst agent failure mode (passes JSON validity, signals done).

## Architecture

`ARCHITECTURE.md` has Mermaid diagrams for the component view, the eval-run lifecycle, and the model-swap memory flow, plus the canonical SQLite schema and a "where do I add X" table.

## Project layout

```
local_ai_bench_on_and_off/
├── ARCHITECTURE.md          # design diagrams + invariants
├── README.md                # this file
├── app.py                   # Streamlit dashboard (Overview / Per-Model / Recommender)
├── swap_model.py            # memory layer CLI
├── models.yaml              # model registry (toggle `enabled`)
├── requirements.txt
├── evals/
│   ├── capability_prompts.yaml      # HITL probe
│   ├── injection_tasks.yaml         # 5 attack patterns
│   └── charts/                      # generated by vision_eval (.gitignored)
├── runners/
│   ├── run_ollama.py                # streaming caller + metrics
│   ├── run_bench.py                 # capability probe orchestrator
│   ├── measure_perf.py              # dimensional + sustained perf
│   ├── agent_loop.py                # Track B mock-FS agent
│   ├── vision_eval.py               # gemma3 multimodal probe
│   ├── memory.py                    # sessions / vec search / swap
│   ├── leaderboard.py               # → dashboards/leaderboard.md
│   ├── score.py                     # deterministic scorers
│   ├── db.py                        # SQLite + sqlite-vec helpers
│   └── schema.sql                   # canonical schema (idempotent)
├── dashboards/
│   └── leaderboard.md               # static export
└── results/
    └── runs.sqlite                  # all rows (gitignored)
```

## Re-running weekly

Schema is `CREATE TABLE IF NOT EXISTS`, models toggle via YAML, runners append to `eval_runs` with timestamps and `run_id`. Pin the full-bench commands in `launchd`/`cron` and the leaderboard regenerates itself. To add a new model, append to `models.yaml` and re-run; no code change.

## License

MIT. See [LICENSE](LICENSE).
