"""Vision eval — gemma3:4b is the only multimodal model in our stack.

Generates 3 synthetic test images and asks gemma3 about them:
  1. Bar-chart QA: "highest value?"  (numerical reasoning over chart)
  2. OCR: "what text is shown?"      (text in image)
  3. Color/shape description          (basic visual recognition)

Writes one row per (model, task) to eval_runs with track='vision'.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runners.run_ollama import call_model
from runners.db import connect, init_db, insert_eval_run
from runners.score import score_prompt

IMG_DIR = ROOT / "evals" / "charts"


def generate_images() -> dict[str, Path]:
    """Create 3 test images. Returns {task_id: image_path}."""
    from PIL import Image, ImageDraw

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 1. Bar chart with labeled values. Highest = 30.
    img = Image.new("RGB", (480, 320), "white")
    d = ImageDraw.Draw(img)
    bars = [("Q1", 10, (60, 120, 200)), ("Q2", 25, (60, 180, 90)),
            ("Q3", 15, (220, 140, 60)), ("Q4", 30, (210, 70, 70))]
    base_y = 290
    chart_height = 220
    for i, (label, val, color) in enumerate(bars):
        x = 70 + i * 95
        h = int(val / 30 * chart_height)
        d.rectangle([x, base_y - h, x + 70, base_y], fill=color)
        d.text((x + 25, base_y + 5), label, fill="black")
        d.text((x + 22, base_y - h - 16), str(val), fill="black")
    d.text((140, 10), "Quarterly Revenue (millions)", fill="black")
    p = IMG_DIR / "bar_chart.png"
    img.save(p)
    paths["chart_max"] = p

    # 2. OCR text — distinct phrase.
    img = Image.new("RGB", (480, 120), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 45), "Local Bench 2026", fill="black")
    p = IMG_DIR / "ocr_text.png"
    img.save(p)
    paths["ocr_basic"] = p

    # 3. Red circle on yellow background.
    img = Image.new("RGB", (240, 240), "yellow")
    d = ImageDraw.Draw(img)
    d.ellipse([60, 60, 180, 180], fill="red")
    p = IMG_DIR / "shape.png"
    img.save(p)
    paths["shape_color"] = p

    return paths


VISION_TASKS = [
    {
        "id": "chart_max",
        "domain": "vision_chart",
        "track": "vision",
        "prompt": (
            "Look at this bar chart. What is the highest value shown? "
            "Reply with just the number, no other text."
        ),
        "scoring": {"method": "contains", "expected": "30"},
    },
    {
        "id": "ocr_basic",
        "domain": "vision_ocr",
        "track": "vision",
        "prompt": (
            "Read the text shown in this image. Reply with just the text, "
            "no quotes or extra words."
        ),
        "scoring": {
            "method": "contains_any_ci",
            "expected": ["local bench 2026", "local bench"],
        },
    },
    {
        "id": "shape_color",
        "domain": "vision_color",
        "track": "vision",
        "prompt": (
            "What color is the shape in this image, and what color is the "
            "background? Answer with two words separated by a comma: "
            "shape_color, background_color."
        ),
        "scoring": {
            "method": "contains_any_ci",
            "expected": ["red", "yellow"],
        },
    },
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--no-write-db", action="store_true")
    args = parser.parse_args()

    print(f"generating test images in {IMG_DIR}…")
    paths = generate_images()
    for tid, p in paths.items():
        print(f"  {tid}: {p}")

    init_db()
    conn = connect() if not args.no_write_db else None
    run_id = str(uuid.uuid4())[:8]

    print(f"\nrunning {args.model} on {len(VISION_TASKS)} tasks (run_id={run_id})")
    summary = []
    for task in VISION_TASKS:
        img_path = paths[task["id"]]
        result = call_model(
            args.model,
            task["prompt"],
            ctx_size=4096,
            num_predict=200,
            image_path=str(img_path),
        )
        score, method = score_prompt(result.output, task["scoring"])
        summary.append((task["id"], score, result))
        err = f" ERR({result.error[:40]})" if result.error else ""
        print(
            f"  [{task['id']:14}] score={score:.2f} tok/s={result.tok_per_sec:>5.1f} "
            f"output={result.output.strip()[:120]!r}{err}"
        )

        if conn is not None:
            insert_eval_run(
                conn,
                run_id=run_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                model=args.model,
                domain=task["domain"],
                track=task["track"],
                prompt_id=task["id"],
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                cold_load_ms=result.cold_load_ms,
                ttft_ms=result.ttft_ms,
                latency_ms=result.latency_ms,
                gen_duration_ms=result.gen_duration_ms,
                tok_per_sec=result.tok_per_sec,
                peak_ram_mb=result.peak_ram_mb,
                output=result.output,
                score=score,
                scoring_method=method + "|vision",
                error=result.error,
            )

    n_pass = sum(1 for _, s, _ in summary if s >= 0.5)
    print(f"\n{args.model}: {n_pass}/{len(summary)} vision tasks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
