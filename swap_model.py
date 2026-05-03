"""CLI front-end for the memory / context-swap layer.

Per PLAN.md §12 the headline operation is:
    swap_model.py --session abc --to qwen3:4b

This script exposes that plus the supporting CRUD operations so a session can be
populated, summarized, searched, and swapped from the shell.

Subcommands:
    start     create a session
    turn      append a user/assistant turn
    note      add a structured memory_object (decision/fact/constraint/entity)
    summarize generate a rolling summary using the active model
    search    vector search over summaries + memory_objects
    preamble  print the [CONTEXT] block that would be handed to a swapped model
    swap      change the session's active model and warm it with the preamble
    demo      seed a scripted session about local-model-bench, then swap

Examples:
    python swap_model.py demo
    python swap_model.py swap --session demo-2026-05-03 --to qwen3:4b
    python swap_model.py search "thinking mode" --k 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from runners.memory import MemoryStore
from runners.db import init_db


def cmd_start(mem: MemoryStore, args) -> int:
    sid = mem.start_session(args.session, active_model=args.model)
    print(sid)
    return 0


def cmd_turn(mem: MemoryStore, args) -> int:
    tid = mem.add_turn(args.session, args.role, args.content)
    print(f"turn_id={tid}")
    return 0


def cmd_note(mem: MemoryStore, args) -> int:
    evidence = json.loads(args.evidence) if args.evidence else None
    mid = mem.add_memory_object(
        args.type,
        args.summary,
        session_id=args.session,
        project=args.project,
        evidence=evidence,
        recheck=args.recheck,
    )
    print(mid)
    return 0


def cmd_summarize(mem: MemoryStore, args) -> int:
    summary = mem.make_rolling_summary(args.session, args.model)
    if summary is None:
        print("(no new turns to summarize)", file=sys.stderr)
        return 1
    print(summary)
    return 0


def cmd_search(mem: MemoryStore, args) -> int:
    for h in mem.search(args.query, k=args.k):
        head = h.text.replace("\n", " ")[:200]
        print(f"[{h.distance:.3f}] {h.ref_type}/{h.ref_id}: {head}")
    return 0


def cmd_preamble(mem: MemoryStore, args) -> int:
    print(mem.build_preamble(args.session, k_similar=args.k))
    return 0


def _fmt_ms(v):
    return f"{v:.0f}ms" if v is not None else "-"


def cmd_swap(mem: MemoryStore, args) -> int:
    result = mem.swap_active_model(
        args.session, args.to, warm=not args.no_warm,
        keep_alive_seconds=args.keep_alive,
    )
    print(result["preamble"])
    print()
    if result["warm"]:
        warm = result["warm"]
        print("--- WARM-UP CALL ---")
        print(f"output     : {(warm['output'] or '').strip()[:400] or '(empty — model thinking consumed budget)'}")
        print(f"ttft       : {_fmt_ms(warm['ttft_ms'])}")
        print(f"latency    : {_fmt_ms(warm['latency_ms'])}")
        print(f"cold_load  : {_fmt_ms(warm['cold_load_ms'])}")
        print(f"tokens_out : {warm['tokens_out']}")
        if warm["error"]:
            print(f"error      : {warm['error']}")
    return 0


def cmd_demo(mem: MemoryStore, args) -> int:
    """Populate a scripted session with today's findings, then swap to a new model."""
    sid = "demo-2026-05-03"
    print(f"=== seeding session {sid} on phi4-mini:3.8b ===")
    mem.start_session(sid, active_model="phi4-mini:3.8b")

    convo = [
        ("user",      "We're building local-model-bench on a 16GB M3 Air. PLAN.md §12 calls for a memory layer that survives model swaps."),
        ("assistant", "Read PLAN + ARCHITECTURE. Plan: install Ollama, scaffold harness, smoke 5 prompts × 2 models, then perf sweep."),
        ("user",      "Stage the pulls, don't overload disk."),
        ("assistant", "Pulled qwen3:4b, phi4-mini:3.8b, nomic-embed-text first (~5.3GB). Smoke 9/10 passed."),
        ("user",      "Run the dimensional + sustained perf next."),
        ("assistant", "measure_perf.py done. 3 ctx sizes × 5 sustained × 3 models. All Usable; none Great. Phi4-mini hit 26 tok/s mid-run before throttling to 14."),
        ("user",      "Why was qwen3 so slow on the smoke?"),
        ("assistant", "Default thinking mode. Ollama strips <think> blocks but eval_count includes them — visible/eval ratio was 0.01 vs phi4-mini's 1.0. think:false inlines reasoning into the visible output instead, which breaks JSON parsing."),
    ]
    for role, content in convo:
        mem.add_turn(sid, role, content)

    notes = [
        ("decision",   "Initial 3 models pulled: qwen3:4b, phi4-mini:3.8b, nomic-embed-text. Remaining 4 trickle in during dead time.", ["smoke run 32c6257e"]),
        ("decision",   "measure_perf.py defaults think=False — perf sweep is throughput-pure, not capability-pure.", []),
        ("constraint", "M3 Air 16GB unified RAM, ~40GB free SSD, fanless. Stage downloads. Expect thermal throttling on sustained loads.", []),
        ("fact",       "qwen3:4b has three thinking modes: default (Ollama strips <think>), think:false (inline reasoning, breaks parsers), and /no_think directive in prompt (true silence).", ["B comparison run a17ef647 vs cc284a11"]),
        ("fact",       "Sustained perf on M3 Air: phi4-mini cold ~30 tok/s, throttle to 14 by run 5. Phi3.5 cold ~27, sustained 13-14. Qwen3 (think off) 20 cold, sped up over runs (file cache).", ["perf sweep bkyi9n3vi"]),
        ("entity",     "Headline 6×3 matrix (PLAN §10): rows = models, cols = HITL/Agent/DeepReasoning. Each cell gets a tier (Great/Usable/Agent-risky/Bad fit).", []),
    ]
    for typ, summary, evidence in notes:
        mid = mem.add_memory_object(typ, summary, session_id=sid, project="local-model-bench", evidence=evidence)
        print(f"  + {typ:11} {mid} {summary[:80]}")

    print("\n=== generating rolling summary via phi4-mini:3.8b ===")
    summary = mem.make_rolling_summary(sid, "phi4-mini:3.8b")
    if summary:
        print(summary)
    else:
        print("(no new turns)")

    print("\n=== swap: phi4-mini:3.8b -> qwen3:4b ===")
    result = mem.swap_active_model(sid, "qwen3:4b", warm=True, keep_alive_seconds=600)
    print(result["preamble"])
    if result["warm"]:
        warm = result["warm"]
        print("\n--- qwen3:4b warm-up ack ---")
        print((warm["output"] or "").strip()[:600] or "(empty — thinking consumed budget)")
        print(
            f"\nttft={_fmt_ms(warm['ttft_ms'])} "
            f"latency={_fmt_ms(warm['latency_ms'])} "
            f"cold_load={_fmt_ms(warm['cold_load_ms'])} "
            f"tokens_out={warm['tokens_out']}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("start", help="create a session")
    s.add_argument("--session"); s.add_argument("--model", required=True)

    s = sub.add_parser("turn", help="append a turn")
    s.add_argument("--session", required=True)
    s.add_argument("--role", required=True, choices=["user", "assistant", "system"])
    s.add_argument("--content", required=True)

    s = sub.add_parser("note", help="add a memory_object")
    s.add_argument("--type", required=True, choices=["decision", "fact", "constraint", "entity"])
    s.add_argument("--summary", required=True)
    s.add_argument("--session"); s.add_argument("--project")
    s.add_argument("--evidence", help="JSON list of evidence references")
    s.add_argument("--recheck")

    s = sub.add_parser("summarize", help="generate a rolling summary")
    s.add_argument("--session", required=True); s.add_argument("--model", required=True)

    s = sub.add_parser("search", help="vector search")
    s.add_argument("query"); s.add_argument("--k", type=int, default=5)

    s = sub.add_parser("preamble", help="print the [CONTEXT] block for a session")
    s.add_argument("--session", required=True); s.add_argument("--k", type=int, default=4)

    s = sub.add_parser("swap", help="swap active model, build preamble, warm new model")
    s.add_argument("--session", required=True); s.add_argument("--to", required=True)
    s.add_argument("--no-warm", action="store_true")
    s.add_argument("--keep-alive", type=int, default=600)

    sub.add_parser("demo", help="seed scripted session and demo a swap")
    return p


HANDLERS = {
    "start": cmd_start,
    "turn": cmd_turn,
    "note": cmd_note,
    "summarize": cmd_summarize,
    "search": cmd_search,
    "preamble": cmd_preamble,
    "swap": cmd_swap,
    "demo": cmd_demo,
}


def main() -> int:
    args = build_parser().parse_args()
    init_db()
    mem = MemoryStore()
    if not mem.has_vec:
        print(
            "warn: sqlite-vec extension not loaded — vector search disabled "
            "(memory still works, just no similarity retrieval)",
            file=sys.stderr,
        )
    return HANDLERS[args.cmd](mem, args)


if __name__ == "__main__":
    sys.exit(main())
