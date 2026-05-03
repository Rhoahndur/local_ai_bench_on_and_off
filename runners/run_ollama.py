"""Streaming Ollama caller with TTFT, throughput, and peak-RAM metrics.

ARCHITECTURE.md §2 invariant: streaming is non-optional. TTFT can only be
measured by reading the response stream incrementally. Do not refactor to
stream=False.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional

import requests

OLLAMA_GENERATE = "http://localhost:11434/api/generate"
OLLAMA_EMBED = "http://localhost:11434/api/embed"


@dataclass
class CallResult:
    model: str
    output: str
    tokens_in: int
    tokens_out: int
    ttft_ms: Optional[float]
    latency_ms: float           # wall-clock total (includes cold load)
    gen_duration_ms: Optional[float]  # Ollama eval_duration, generation-only
    tok_per_sec: float          # tokens_out / gen_duration (generation throughput)
    cold_load_ms: Optional[float]
    peak_ram_mb: Optional[float]
    error: Optional[str] = None

    def to_row(self) -> dict:
        return asdict(self)


def _sample_peak_ram(stop_event: threading.Event, sink: dict, interval: float = 0.1) -> None:
    """Poll RSS of any process whose name contains 'ollama' and record the peak.

    Runs in a background thread for the duration of one call. We poll instead of
    reading a pre/post delta because the runtime forks helper processes whose
    RSS we'd otherwise miss.
    """
    try:
        import psutil
    except ImportError:
        sink["peak_ram_mb"] = None
        return

    peak = 0.0
    while not stop_event.is_set():
        try:
            for proc in psutil.process_iter(["name", "memory_info"]):
                name = (proc.info.get("name") or "").lower()
                if "ollama" in name:
                    rss_mb = proc.info["memory_info"].rss / (1024 * 1024)
                    if rss_mb > peak:
                        peak = rss_mb
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        stop_event.wait(interval)
    sink["peak_ram_mb"] = peak if peak > 0 else None


def call_model(
    model: str,
    prompt: str,
    ctx_size: int = 4096,
    image_path: Optional[str] = None,
    timeout: float = 600.0,
    num_predict: Optional[int] = None,
    think: Optional[bool] = None,
    keep_alive: int = 0,
) -> CallResult:
    """Stream one completion.

    Args:
        num_predict: cap output tokens (passed to Ollama options.num_predict).
        think: optional override for thinking-mode-capable models (e.g. qwen3).
            False inlines reasoning into visible output (does NOT silence qwen3 —
            use a `/no_think` directive in the prompt for that).
        keep_alive: seconds to keep model resident after this call. Default 0
            (unload immediately, consistent with eval-harness defaults). Set
            higher (e.g. 600) for memory-layer warm-up calls.
    """
    options: dict = {"num_ctx": ctx_size}
    if num_predict is not None:
        options["num_predict"] = num_predict
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "options": options,
        "keep_alive": keep_alive,
        "stream": True,
    }
    if think is not None:
        payload["think"] = think
    if image_path:
        with open(image_path, "rb") as f:
            payload["images"] = [base64.b64encode(f.read()).decode("utf-8")]

    stop_event = threading.Event()
    sampler_sink: dict = {}
    sampler_thread = threading.Thread(
        target=_sample_peak_ram, args=(stop_event, sampler_sink), daemon=True
    )
    sampler_thread.start()

    t0 = time.perf_counter()
    ttft_ms: Optional[float] = None
    output_parts: list[str] = []
    tokens_in = tokens_out = 0
    load_duration_ms: Optional[float] = None
    gen_duration_ms: Optional[float] = None
    error: Optional[str] = None

    try:
        with requests.post(OLLAMA_GENERATE, json=payload, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw:
                    continue
                chunk = json.loads(raw)
                if "error" in chunk:
                    error = chunk["error"]
                    break
                if ttft_ms is None and chunk.get("response"):
                    ttft_ms = (time.perf_counter() - t0) * 1000
                output_parts.append(chunk.get("response", ""))
                if chunk.get("done"):
                    tokens_in = chunk.get("prompt_eval_count") or 0
                    tokens_out = chunk.get("eval_count") or 0
                    if "load_duration" in chunk:
                        load_duration_ms = chunk["load_duration"] / 1_000_000  # ns -> ms
                    if "eval_duration" in chunk:
                        gen_duration_ms = chunk["eval_duration"] / 1_000_000   # ns -> ms
    except requests.RequestException as e:
        error = f"request_error: {e}"
    finally:
        stop_event.set()
        sampler_thread.join(timeout=1.0)

    latency_ms = (time.perf_counter() - t0) * 1000
    if gen_duration_ms and gen_duration_ms > 0 and tokens_out > 0:
        tok_per_sec = tokens_out / (gen_duration_ms / 1000)
    else:
        tok_per_sec = 0.0

    return CallResult(
        model=model,
        output="".join(output_parts),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        gen_duration_ms=gen_duration_ms,
        tok_per_sec=tok_per_sec,
        cold_load_ms=load_duration_ms,
        peak_ram_mb=sampler_sink.get("peak_ram_mb"),
        error=error,
    )


def embed(text: str, model: str = "nomic-embed-text", timeout: float = 120.0) -> list[float]:
    """Return one embedding vector via /api/embed."""
    resp = requests.post(
        OLLAMA_EMBED, json={"model": model, "input": text}, timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings") or []
    if not embeddings:
        raise RuntimeError(f"no embedding returned: {data}")
    return embeddings[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the Ollama caller.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Say hi in 5 words.")
    parser.add_argument("--ctx", type=int, default=4096)
    args = parser.parse_args()

    result = call_model(args.model, args.prompt, ctx_size=args.ctx)
    print(json.dumps(result.to_row(), indent=2))
