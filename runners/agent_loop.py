"""Agent loop runner — Track B from PLAN.md §10.

Multi-step task with mock filesystem tools. The task is the canonical example:
"Read notes.txt, find TODO items, sort by priority, write to todos.txt."

We use **prompt-based** tool calling (not Ollama's native /api/chat tools API)
so the harness compares all six models on equal footing — phi3.5 and gemma3
don't claim native tool support, but every model can be asked to emit JSON.
That uniformity is the whole point: "tool_call_valid_pct" becomes a real
capability metric, not a feature flag.

Per-run metrics written to eval_runs (track='agent'):
  - score              0..1 (1.0 = todos.txt has all 3 TODOs in P1→P3 order)
  - tool_call_valid_pct  % of assistant turns that parsed as a JSON tool call
  - steps              total rounds to done or step_limit
  - done_signaled      did the model use the `done` tool, or did we run out

Usage:
    python -m runners.agent_loop --model phi4-mini:3.8b
    python -m runners.agent_loop --all
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runners.run_ollama import call_model
from runners.db import connect, init_db, insert_eval_run


SYSTEM_PROMPT = """You are an agent that solves tasks by calling tools.

Available tools:
- read_file:   returns file contents.    args: {"path": "<filename>"}
- write_file:  writes content to a file. args: {"path": "<filename>", "content": "<text>"}
- list_files:  returns list of paths.    args: {}
- done:        signal task complete.     args: {"summary": "<short>"}

Rules:
1. Each turn, reply with EXACTLY ONE JSON object on a single line. No prose.
2. Format: {"tool": "<name>", "args": {<args>}}
3. Wait for the TOOL RESULT before making the next call.
4. When the task is complete, call `done`.
"""

TASKS: dict[str, dict] = {
    "todos_from_notes": {
        "fs": {
            "notes.txt": (
                "- buy milk\n"
                "- TODO: fix the auth bug (P1)\n"
                "- TODO: update the readme (P3)\n"
                "- meeting with Alice tomorrow\n"
                "- TODO: review PR #42 (P2)\n"
                "- water plants\n"
            ),
        },
        "task": (
            "Read notes.txt, extract every TODO item, sort them by priority "
            "(P1 first, then P2, then P3), and write them to todos.txt as "
            "a numbered list (one per line, format: \"1. <text>\")."
        ),
        "scorer": "_score_todos",
    },
}


def _score_todos(fs: dict[str, str]) -> tuple[float, str]:
    """Did todos.txt end up with all 3 TODOs in P1→P2→P3 order?"""
    text = fs.get("todos.txt", "")
    if not text:
        return 0.0, "todos.txt missing or empty"
    lower = text.lower()
    flags = {
        "auth (P1)": "auth" in lower or "fix" in lower,
        "pr 42 (P2)": "42" in lower or "review" in lower,
        "readme (P3)": "readme" in lower or "update" in lower,
    }
    if not all(flags.values()):
        missing = [k for k, v in flags.items() if not v]
        return 0.3, f"missing: {', '.join(missing)}"
    # Locate first occurrence of each marker; check ordering.
    def _pos(*needles):
        positions = [lower.find(n) for n in needles if lower.find(n) >= 0]
        return min(positions) if positions else len(lower) + 1
    p_auth = _pos("auth", "fix")
    p_pr = _pos("42", "review")
    p_readme = _pos("readme", "update")
    if p_auth < p_pr < p_readme:
        return 1.0, "all 3 TODOs present, P1→P2→P3 order"
    return 0.7, f"all 3 present but order wrong (auth={p_auth} pr={p_pr} readme={p_readme})"


SCORERS = {"_score_todos": _score_todos}


def _parse_tool_call(text: str) -> tuple[dict | None, str]:
    """Extract first balanced JSON object from text.

    Robust to leading reasoning prelude (qwen3 think:false) and code fences.
    Returns (obj_or_None, error_message).
    """
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines)
    start = s.find("{")
    if start < 0:
        return None, "no '{' in output"
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(s[start : i + 1])
                    if not isinstance(obj, dict):
                        return None, "parsed JSON is not an object"
                    return obj, ""
                except json.JSONDecodeError as e:
                    return None, f"json decode: {e}"
    return None, "unbalanced braces"


def _execute_tool(call: dict, fs: dict[str, str]) -> dict:
    tool = call.get("tool")
    args = call.get("args") or {}
    if tool == "read_file":
        path = args.get("path", "")
        if path in fs:
            return {"result": fs[path]}
        return {"error": f"file not found: {path}", "available": list(fs.keys())}
    if tool == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)
        fs[path] = content
        return {"result": f"wrote {len(content)} chars to {path}"}
    if tool == "list_files":
        return {"result": list(fs.keys())}
    if tool == "done":
        return {"result": "task ended", "_done": True}
    return {"error": f"unknown tool: {tool!r}", "valid_tools": ["read_file", "write_file", "list_files", "done"]}


def _build_prompt(history: list[tuple[str, str]]) -> str:
    parts = []
    for role, content in history:
        if role == "system":
            parts.append(content.rstrip())
        elif role == "user":
            parts.append(f"\nUSER: {content}")
        elif role == "assistant":
            parts.append(f"\nASSISTANT: {content}")
    parts.append("\nASSISTANT: ")
    return "".join(parts)


def run_agent_task(
    model: str,
    task_id: str,
    *,
    step_limit: int = 10,
    ctx_size: int = 8192,
    verbose: bool = True,
) -> dict:
    task = TASKS[task_id]
    fs = dict(task["fs"])
    scorer = SCORERS[task["scorer"]]

    history: list[tuple[str, str]] = [
        ("system", SYSTEM_PROMPT),
        ("user", f"TASK: {task['task']}\n\nMake your first tool call now."),
    ]
    valid_calls = 0
    invalid_calls = 0
    steps = 0
    done_signaled = False
    transcript: list[dict] = []

    while steps < step_limit:
        steps += 1
        prompt = _build_prompt(history)
        result = call_model(
            model, prompt, ctx_size=ctx_size, num_predict=400, think=False
        )
        if result.error:
            transcript.append({"step": steps, "error": result.error})
            return {
                "model": model, "task_id": task_id, "steps": steps,
                "valid_calls": valid_calls, "invalid_calls": invalid_calls,
                "tool_call_valid_pct": 0.0,
                "score": 0.0, "score_msg": f"call error: {result.error}",
                "task_complete": 0, "done_signaled": 0,
                "fs_final": fs, "transcript": transcript,
                "error": result.error,
            }

        output = (result.output or "").strip()
        history.append(("assistant", output))

        call, parse_err = _parse_tool_call(output)
        if call is None:
            invalid_calls += 1
            if verbose:
                print(f"  [{steps}] INVALID: {parse_err} | output[:80]={output[:80]!r}")
            transcript.append({"step": steps, "raw": output[:300], "parse_error": parse_err})
            history.append(("user", f"ERROR: failed to parse tool call ({parse_err}). Reply with valid JSON only."))
            continue

        valid_calls += 1
        tool_result = _execute_tool(call, fs)
        if verbose:
            tool_str = call.get("tool", "?")
            args_str = json.dumps(call.get("args") or {})[:80]
            res_summary = "(done)" if tool_result.get("_done") else (
                f"err={tool_result['error']}" if "error" in tool_result else
                f"ok ({len(json.dumps(tool_result.get('result',''))):d} chars)"
            )
            print(f"  [{steps}] {tool_str}({args_str}) -> {res_summary}")
        transcript.append({"step": steps, "call": call, "tool_result": tool_result})

        if tool_result.get("_done"):
            done_signaled = True
            break
        # Strip _done flag before sending back
        tool_result_to_send = {k: v for k, v in tool_result.items() if not k.startswith("_")}
        history.append(("user", f"TOOL RESULT: {json.dumps(tool_result_to_send)}\n\nMake your next tool call."))

    score, msg = scorer(fs)
    total_attempts = valid_calls + invalid_calls
    return {
        "model": model, "task_id": task_id, "steps": steps,
        "valid_calls": valid_calls, "invalid_calls": invalid_calls,
        "tool_call_valid_pct": (valid_calls / max(1, total_attempts)) * 100,
        "score": score, "score_msg": msg,
        "task_complete": int(score >= 1.0),
        "done_signaled": int(done_signaled),
        "fs_final": fs, "transcript": transcript,
        "error": None,
    }


def load_active_models(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text())
    return [m["id"] for m in data.get("models", []) if m.get("enabled")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="single model id")
    parser.add_argument("--all", action="store_true", help="all enabled models")
    parser.add_argument("--task", default="todos_from_notes")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--no-write-db", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.model:
        models = [args.model]
    elif args.all:
        models = load_active_models(ROOT / "models.yaml")
    else:
        parser.error("specify --model or --all")
        return 2

    init_db()
    conn = connect() if not args.no_write_db else None
    run_id = str(uuid.uuid4())[:8]

    summaries = []
    for m in models:
        print(f"\n=== {m} | {args.task} ===")
        result = run_agent_task(
            m, args.task,
            step_limit=args.steps, ctx_size=args.ctx, verbose=not args.quiet,
        )
        print(
            f"  steps={result['steps']} "
            f"valid_calls={result['valid_calls']}/{result['valid_calls']+result['invalid_calls']} "
            f"({result['tool_call_valid_pct']:.0f}%) "
            f"done={result['done_signaled']} "
            f"score={result['score']:.2f} ({result['score_msg']})"
        )
        if "todos.txt" in result["fs_final"]:
            preview = result["fs_final"]["todos.txt"]
            print(f"  todos.txt:\n{preview[:400]}")
        summaries.append((m, result))
        if conn is not None:
            insert_eval_run(
                conn,
                run_id=run_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                model=m,
                track="agent",
                domain="tool_use",
                prompt_id=args.task,
                score=result["score"],
                scoring_method=(
                    f"agent|valid_pct={result['tool_call_valid_pct']:.0f}"
                    f"|steps={result['steps']}|done={result['done_signaled']}"
                ),
                output=json.dumps({
                    "fs_final": result["fs_final"],
                    "msg": result["score_msg"],
                })[:8000],
                error=result["error"],
            )

    if len(summaries) > 1:
        print("\n=== summary ===")
        print(f"  {'model':35} {'score':>5} {'valid%':>7} {'steps':>5} {'done':>4}")
        for m, r in summaries:
            print(
                f"  {m:35} {r['score']:>5.2f} "
                f"{r['tool_call_valid_pct']:>6.0f}% {r['steps']:>5} "
                f"{r['done_signaled']:>4}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
