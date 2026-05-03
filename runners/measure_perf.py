"""Performance profiling: dimensional sweep + sustained throttle test.

Per PLAN.md §9:
  Pass 1 — dimensional: ~num_predict-token generation at 1k/4k/8k/16k context
           sizes. One call each. Captures cold_load, TTFT, gen tok/s, peak RAM.
  Pass 2 — sustained:    same prompt N times in a row at fixed ctx, NO COOLDOWN.
           Throttle = (first_tok_per_sec - last_tok_per_sec) / first * 100.

Tier classification (PLAN §9):
    Great       sustained > 25 tok/s AND throttle < 10%
    Usable      10-25 tok/s
    Agent-risky 5-10 tok/s
    Bad fit     < 5 tok/s

This is a pure throughput test, so think=False is the default — qwen3's hidden
thinking tokens would otherwise dominate every cell with reasoning overhead
unrelated to generation speed.

Usage:
    python -m runners.measure_perf --model phi4-mini:3.8b --mode both
    python -m runners.measure_perf --all --mode dimensional --quick
    python -m runners.measure_perf --all --mode sustained
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runners.run_ollama import call_model
from runners.db import connect, init_db


GEN_INSTRUCTION = (
    "INSTRUCTION: Write a detailed technical walkthrough of how a token-bucket "
    "rate limiter works. Cover the data structures, the tick logic, the refill "
    "logic, edge cases (clock skew, refill bursts, concurrent access), and "
    "include Python pseudocode. Be specific and concrete; aim for ~1500 words."
)

FILLER_PARA = (
    "The system orchestrates a flow of events through a structured pipeline. "
    "Each event carries a payload and metadata which guide downstream stages. "
    "When a stage completes, it emits a result that feeds the next consumer. "
    "Errors are bubbled up with provenance so debugging remains tractable. "
    "Logging captures decisions and timing, which together enable retrospection.\n\n"
)


def make_context(target_tokens: int, instruction: str = GEN_INSTRUCTION) -> str:
    """Build a prompt of approximately target_tokens (heuristic: 1 token ~ 4 chars).

    Ground truth comes from Ollama's prompt_eval_count, which is recorded with
    each row. The heuristic gets us into the right neighborhood, not exact.
    """
    target_chars = max(0, target_tokens * 4 - len(instruction) - 100)
    if target_chars == 0:
        return instruction
    times = target_chars // len(FILLER_PARA) + 1
    filler = (FILLER_PARA * times)[:target_chars]
    return f"{filler}\n\n{instruction}"


def tier_for(tok_per_sec: float, throttle_pct: float = 0.0) -> str:
    if tok_per_sec > 25 and throttle_pct < 10:
        return "Great"
    if tok_per_sec >= 10:
        return "Usable"
    if tok_per_sec >= 5:
        return "Agent-risky"
    return "Bad fit"


def load_active_models(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text())
    return [m["id"] for m in data.get("models", []) if m.get("enabled")]


def run_dimensional(
    model: str,
    ctx_sizes: list[int],
    num_predict: int,
    conn,
    think: bool | None,
) -> list[dict]:
    rows: list[dict] = []
    for ctx in ctx_sizes:
        prompt = make_context(ctx)
        # Reserve buffer beyond input for the generation tail.
        request_ctx = ctx + max(2 * num_predict, 1024)
        print(f"  [{model}] dimensional ctx={ctx}  ", end="", flush=True)
        result = call_model(
            model,
            prompt,
            ctx_size=request_ctx,
            num_predict=num_predict,
            think=think,
        )
        ts = datetime.now(timezone.utc).isoformat()
        row = {
            "model": model,
            "ts": ts,
            "ctx_size": ctx,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "cold_load_ms": result.cold_load_ms,
            "ttft_ms": result.ttft_ms,
            "gen_duration_ms": result.gen_duration_ms,
            "latency_ms": result.latency_ms,
            "tok_per_sec": result.tok_per_sec,
            "peak_ram_mb": result.peak_ram_mb,
        }
        conn.execute(
            "INSERT INTO perf_dimensional "
            "(model,ts,ctx_size,tokens_in,tokens_out,cold_load_ms,ttft_ms,"
            "gen_duration_ms,latency_ms,tok_per_sec,peak_ram_mb) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                row["model"], row["ts"], row["ctx_size"], row["tokens_in"],
                row["tokens_out"], row["cold_load_ms"], row["ttft_ms"],
                row["gen_duration_ms"], row["latency_ms"], row["tok_per_sec"],
                row["peak_ram_mb"],
            ],
        )
        conn.commit()
        rows.append(row)
        err = f" ERR({result.error[:40]})" if result.error else ""
        print(
            f"toks_in={result.tokens_in:>6} toks_out={result.tokens_out:>4} "
            f"tok/s={result.tok_per_sec:>5.1f} ttft={result.ttft_ms or 0:>5.0f}ms "
            f"peak_ram={result.peak_ram_mb or 0:>6.0f}MB{err}"
        )
    return rows


def run_sustained(
    model: str,
    ctx: int,
    num_predict: int,
    runs: int,
    conn,
    think: bool | None,
) -> list[dict]:
    prompt = make_context(ctx)
    request_ctx = ctx + max(2 * num_predict, 1024)
    rows: list[dict] = []
    first_tps = None
    print(f"  [{model}] sustained ctx={ctx} runs={runs} num_predict={num_predict} (no cooldown)")
    for i in range(1, runs + 1):
        result = call_model(
            model,
            prompt,
            ctx_size=request_ctx,
            num_predict=num_predict,
            think=think,
        )
        ts = datetime.now(timezone.utc).isoformat()
        if first_tps is None and result.tok_per_sec > 0:
            first_tps = result.tok_per_sec
        throttle = (
            (first_tps - result.tok_per_sec) / first_tps * 100
            if first_tps and first_tps > 0
            else 0.0
        )
        row = {
            "model": model,
            "ts": ts,
            "run_index": i,
            "ctx_size": ctx,
            "tokens_out": result.tokens_out,
            "gen_duration_ms": result.gen_duration_ms,
            "tok_per_sec": result.tok_per_sec,
            "throttle_pct_vs_first": throttle,
            "peak_ram_mb": result.peak_ram_mb,
        }
        conn.execute(
            "INSERT INTO perf_sustained "
            "(model,ts,run_index,ctx_size,tokens_out,gen_duration_ms,tok_per_sec,"
            "throttle_pct_vs_first,peak_ram_mb) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [
                row["model"], row["ts"], row["run_index"], row["ctx_size"],
                row["tokens_out"], row["gen_duration_ms"], row["tok_per_sec"],
                row["throttle_pct_vs_first"], row["peak_ram_mb"],
            ],
        )
        conn.commit()
        rows.append(row)
        err = f" ERR({result.error[:40]})" if result.error else ""
        print(
            f"    run {i}/{runs}: tok/s={result.tok_per_sec:>5.1f} "
            f"throttle={throttle:>5.1f}% peak_ram={result.peak_ram_mb or 0:>6.0f}MB{err}"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="single model id")
    parser.add_argument("--all", action="store_true", help="run on all enabled models")
    parser.add_argument("--mode", choices=["dimensional", "sustained", "both"], default="both")
    parser.add_argument("--quick", action="store_true", help="abbreviated sweep for iteration speed")
    parser.add_argument("--ctx-sizes", default="1024,4096,8192,16384", help="dimensional ctx sizes")
    parser.add_argument("--sustained-ctx", type=int, default=4096)
    parser.add_argument("--sustained-runs", type=int, default=5)
    parser.add_argument("--num-predict", type=int, default=2000, help="target output tokens")
    parser.add_argument(
        "--think",
        choices=["on", "off", "default"],
        default="off",
        help="thinking mode override (default: off — perf is pure throughput)",
    )
    args = parser.parse_args()

    if args.quick:
        args.ctx_sizes = "1024,4096"
        args.sustained_runs = 3
        args.num_predict = 200

    think_arg: bool | None
    think_arg = {"on": True, "off": False, "default": None}[args.think]

    init_db()
    conn = connect()

    if args.model:
        models = [args.model]
    elif args.all:
        models = load_active_models(ROOT / "models.yaml")
    else:
        parser.error("specify --model or --all")
        return 2

    ctx_sizes = [int(x) for x in args.ctx_sizes.split(",") if x.strip()]
    print(
        f"models={models} mode={args.mode} ctx_sizes={ctx_sizes} "
        f"num_predict={args.num_predict} think={args.think}"
    )

    summary: list[tuple] = []
    for m in models:
        if args.mode in ("dimensional", "both"):
            run_dimensional(m, ctx_sizes, args.num_predict, conn, think_arg)
        if args.mode in ("sustained", "both"):
            sustained_rows = run_sustained(
                m, args.sustained_ctx, args.num_predict,
                args.sustained_runs, conn, think_arg
            )
            valid = [r for r in sustained_rows if r["tok_per_sec"] > 0]
            if valid:
                first_tps = valid[0]["tok_per_sec"]
                last_tps = valid[-1]["tok_per_sec"]
                throttle = valid[-1]["throttle_pct_vs_first"]
                summary.append((m, first_tps, last_tps, throttle, tier_for(last_tps, throttle)))

    if summary:
        print("\n=== Sustained tier verdict ===")
        print(f"  {'model':35} {'first':>8} {'last':>8} {'throttle':>10} {'tier':>14}")
        for m, f, l, t, tier in summary:
            print(f"  {m:35} {f:>8.1f} {l:>8.1f} {t:>9.1f}% {tier:>14}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
