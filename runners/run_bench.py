"""Minimum-viable bench runner.

Loads models.yaml + a prompts YAML, runs each enabled model across all prompts,
scores deterministically, writes one row per inference to runs.sqlite.

Usage:
    python -m runners.run_bench                          # all enabled, all prompts
    python -m runners.run_bench --limit 1                # smoke: first prompt only
    python -m runners.run_bench --only-model qwen3:4b    # single model
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runners.run_ollama import call_model
from runners.db import connect, init_db, insert_eval_run
from runners.score import score_prompt


def load_models(path: Path, *, only_enabled: bool = True) -> list[dict]:
    data = yaml.safe_load(path.read_text())
    models = data.get("models", [])
    return [m for m in models if m.get("enabled")] if only_enabled else models


def load_prompts(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text())
    return data.get("prompts", [])


def run_once(model_cfg: dict, prompt: dict, ctx_size: int, think: bool | None = None) -> dict:
    """Run one (model, prompt) pair. `think` overrides thinking mode for capable models."""
    # Per-model default in models.yaml takes effect only when CLI think=None
    effective_think = think if think is not None else model_cfg.get("think_default")
    result = call_model(
        model_cfg["id"], prompt["prompt"], ctx_size=ctx_size, think=effective_think
    )
    score, method = score_prompt(result.output, prompt["scoring"])
    return {
        "model": model_cfg["id"],
        "tier": model_cfg.get("tier"),
        "domain": prompt.get("domain"),
        "track": prompt.get("track"),
        "prompt_id": prompt["id"],
        "ctx_size": ctx_size,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "cold_load_ms": result.cold_load_ms,
        "ttft_ms": result.ttft_ms,
        "latency_ms": result.latency_ms,
        "gen_duration_ms": result.gen_duration_ms,
        "tok_per_sec": result.tok_per_sec,
        "peak_ram_mb": result.peak_ram_mb,
        "output": result.output,
        "score": score,
        "scoring_method": method,
        "error": result.error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-yaml", default=str(ROOT / "models.yaml"))
    parser.add_argument(
        "--prompts-yaml", default=str(ROOT / "evals" / "capability_prompts.yaml")
    )
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0, help="cap prompts (smoke mode)")
    parser.add_argument("--only-model", default=None, help="run a single model id")
    parser.add_argument(
        "--think",
        choices=["on", "off", "default"],
        default="default",
        help="thinking mode override for capable models (qwen3); 'default' uses per-model setting",
    )
    parser.add_argument(
        "--tag", default=None,
        help="optional label written to scoring_method as suffix (e.g. 'think-off') for grouping",
    )
    args = parser.parse_args()
    think_arg = {"on": True, "off": False, "default": None}[args.think]

    init_db()
    run_id = str(uuid.uuid4())[:8]

    models = load_models(Path(args.models_yaml))
    prompts = load_prompts(Path(args.prompts_yaml))

    if args.only_model:
        models = [m for m in models if m["id"] == args.only_model]
    if args.limit:
        prompts = prompts[: args.limit]

    if not models:
        print("No enabled models. Set `enabled: true` in models.yaml.", file=sys.stderr)
        return 1
    if not prompts:
        print("No prompts loaded.", file=sys.stderr)
        return 1

    print(f"run_id={run_id}  models={[m['id'] for m in models]}  prompts={len(prompts)}")

    conn = connect()
    total = passed = failed = 0
    t_start = time.perf_counter()

    for m in models:
        for p in prompts:
            total += 1
            ts = datetime.now(timezone.utc).isoformat()
            row = run_once(m, p, ctx_size=args.ctx, think=think_arg)
            row["run_id"] = run_id
            row["timestamp"] = ts
            if args.tag:
                row["scoring_method"] = f"{row['scoring_method']}|{args.tag}"
            try:
                insert_eval_run(conn, **row)
            except Exception as e:
                print(f"db insert failed: {e}", file=sys.stderr)
            score_str = f"{row['score']:.2f}" if row["error"] is None else "-"
            tps = f"{row['tok_per_sec']:.1f}" if row["tok_per_sec"] else "?"
            err = f" ERR({row['error'][:50]})" if row["error"] else ""
            print(
                f"  [{m['id']:35}] {p['id']:30} score={score_str:>5} tok/s={tps:>5}{err}"
            )
            if row["error"] is None and row["score"] >= 0.5:
                passed += 1
            else:
                failed += 1

    elapsed = time.perf_counter() - t_start
    print(
        f"\nrun_id={run_id} total={total} passed={passed} failed={failed} elapsed={elapsed:.1f}s"
    )
    print(f"results in: {ROOT / 'results' / 'runs.sqlite'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
