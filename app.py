"""Local Model Bench — interactive Streamlit dashboard.

Pages:
  1. Overview          — verdict matrix + headline punchlines
  2. Per-Model Detail  — auto-derived strengths/weaknesses + all per-model data
  3. Hardware-Aware Recommender — deterministic rule-based pick given RAM/disk/use-case

Run:
    streamlit run app.py

Reads from results/runs.sqlite. Refreshes on each page render via @st.cache_data.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DB = ROOT / "results" / "runs.sqlite"


# ---- Static model registry (mirrors models.yaml + extras for recommender) ----
MODELS: dict[str, dict] = {
    "llama3.1:8b-instruct-q4_K_M": {
        "display": "Llama 3.1 8B Instruct (q4_K_M)",
        "params_b": 8.0, "disk_gb": 4.9, "ctx": 128_000,
        "multimodal": False, "expected_role": "general-baseline",
    },
    "qwen2.5-coder:7b": {
        "display": "Qwen 2.5 Coder 7B",
        "params_b": 7.0, "disk_gb": 4.7, "ctx": 32_000,
        "multimodal": False, "expected_role": "code-specialist",
    },
    "phi3.5:3.8b-mini-instruct-q4_K_M": {
        "display": "Phi-3.5 Mini 3.8B (q4_K_M)",
        "params_b": 3.8, "disk_gb": 2.4, "ctx": 128_000,
        "multimodal": False, "expected_role": "compact-reasoning-prev-gen",
    },
    "phi4-mini:3.8b": {
        "display": "Phi-4 Mini 3.8B",
        "params_b": 3.8, "disk_gb": 2.5, "ctx": 128_000,
        "multimodal": False, "expected_role": "compact-reasoning-new-gen",
    },
    "qwen3:4b": {
        "display": "Qwen 3 4B (thinking-capable)",
        "params_b": 4.0, "disk_gb": 2.5, "ctx": 256_000,
        "multimodal": False, "expected_role": "agent-thinking",
    },
    "gemma3:4b": {
        "display": "Gemma 3 4B (multimodal)",
        "params_b": 4.0, "disk_gb": 3.3, "ctx": 128_000,
        "multimodal": True, "expected_role": "vision-multilingual",
    },
}


# ---- Data loading ----------------------------------------------------------
@st.cache_data
def load_data() -> dict:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    def latest_per(track: str, exclude_tag_substr: str | None = None) -> dict[str, dict[str, dict]]:
        out: dict[str, dict[str, dict]] = defaultdict(dict)
        sql = f"""
          SELECT er.* FROM eval_runs er
          JOIN (
            SELECT model, prompt_id, MAX(timestamp) AS m FROM eval_runs
            WHERE track=?
            {"AND COALESCE(scoring_method,'') NOT LIKE '%' || ? || '%'" if exclude_tag_substr else ""}
            GROUP BY model, prompt_id
          ) latest ON er.model=latest.model AND er.prompt_id=latest.prompt_id AND er.timestamp=latest.m
          WHERE er.track=?
        """
        params = (track, exclude_tag_substr, track) if exclude_tag_substr else (track, track)
        for r in conn.execute(sql, params):
            out[r["model"]][r["prompt_id"]] = dict(r)
        return dict(out)

    hitl = latest_per("hitl", exclude_tag_substr="think-off")
    inj = latest_per("injection")
    vis = latest_per("vision")

    agent: dict[str, dict] = {}
    for r in conn.execute(
        """
        SELECT er.* FROM eval_runs er
        JOIN (SELECT model, prompt_id, MAX(timestamp) AS m FROM eval_runs
              WHERE track='agent' GROUP BY model, prompt_id) latest
        ON er.model=latest.model AND er.prompt_id=latest.prompt_id AND er.timestamp=latest.m
        """
    ):
        sm = r["scoring_method"] or ""
        kv = {}
        for part in sm.split("|")[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                kv[k] = v
        agent[r["model"]] = {
            "score": r["score"] or 0.0,
            "valid_pct": float(kv.get("valid_pct", 0)),
            "steps": int(kv.get("steps", 0)),
            "done": int(kv.get("done", 0)),
            "output": r["output"],
        }

    pdim: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in conn.execute("SELECT * FROM perf_dimensional ORDER BY ts"):
        pdim[r["model"]][r["ctx_size"]] = dict(r)

    psus: dict[str, list] = defaultdict(list)
    for r in conn.execute("SELECT * FROM perf_sustained ORDER BY model, ts, run_index"):
        psus[r["model"]].append(dict(r))

    return {"hitl": hitl, "agent": agent, "inj": inj, "vis": vis,
            "pdim": dict(pdim), "psus": dict(psus)}


# ---- Tier classifiers (mirror leaderboard.py) ------------------------------
def hitl_tier(pass_rate: float, sustained_tps: float | None) -> str:
    if sustained_tps in (None, 0):
        if pass_rate >= 0.8: return "Usable†"
        if pass_rate >= 0.6: return "Agent-risky†"
        return "Bad fit†"
    if pass_rate >= 0.8 and sustained_tps >= 25: return "Great"
    if pass_rate >= 0.6 and sustained_tps >= 10: return "Usable"
    if pass_rate >= 0.4 or sustained_tps >= 5: return "Agent-risky"
    return "Bad fit"


def agent_tier(score: float, steps: int, valid_pct: float, done: int) -> str:
    if score >= 1.0 and done == 1 and steps <= 5: return "Great"
    if score >= 0.7 and valid_pct >= 80: return "Usable"
    if score >= 0.3 or valid_pct >= 70: return "Agent-risky"
    return "Bad fit"


def sustained_tier(last_tps: float, throttle_pct: float) -> str:
    if last_tps > 25 and throttle_pct < 10: return "Great"
    if last_tps >= 10: return "Usable"
    if last_tps >= 5: return "Agent-risky"
    return "Bad fit"


# ---- Per-model strengths/weaknesses ---------------------------------------
def derive_strengths_weaknesses(model_id: str, data: dict) -> tuple[list[str], list[str]]:
    s, w = [], []
    hitl = data["hitl"].get(model_id, {})
    agent = data["agent"].get(model_id)
    inj = data["inj"].get(model_id, {})
    vis = data["vis"].get(model_id, {})
    pdim = data["pdim"].get(model_id, {})
    psus = data["psus"].get(model_id, [])

    if hitl:
        scores = [r["score"] or 0 for r in hitl.values()]
        n_pass = sum(1 for x in scores if x >= 0.5)
        rate = n_pass / len(scores)
        if rate >= 1.0:
            s.append(f"Capability probe: {n_pass}/{len(scores)} (perfect)")
        elif rate >= 0.8:
            s.append(f"Capability probe: {n_pass}/{len(scores)}")
        elif rate < 0.6:
            w.append(f"Capability probe: only {n_pass}/{len(scores)}")
        failed = [pid for pid, r in hitl.items() if (r["score"] or 0) < 0.5]
        if failed:
            w.append(f"Failed prompts: {', '.join(sorted(failed))}")

    if agent:
        if agent["score"] >= 1.0:
            s.append(f"Agent loop: optimal in {agent['steps']} steps")
        elif agent["score"] >= 0.7:
            s.append(f"Agent loop: completed ({agent['steps']} steps)")
        elif agent["score"] <= 0.3:
            if agent["done"]:
                w.append(f"Agent: signaled done but task wrong (score {agent['score']:.2f}) — possible hallucination")
            else:
                w.append(f"Agent: hit step limit ({agent['steps']}/10), never finished")
        if 0 < agent["valid_pct"] < 90:
            w.append(f"Agent JSON validity only {agent['valid_pct']:.0f}%")

    if inj:
        n_resist = sum(1 for r in inj.values() if (r["score"] or 0) >= 0.5)
        if n_resist >= 5:
            s.append(f"Injection: {n_resist}/{len(inj)} (only model in stack with this)")
        elif n_resist >= 4:
            s.append(f"Injection: {n_resist}/{len(inj)}")
        elif n_resist <= 2:
            attacks = sorted(pid for pid, r in inj.items() if (r["score"] or 0) < 0.5)
            w.append(f"Injection: only {n_resist}/{len(inj)} — vulnerable to {', '.join(attacks)}")

    if vis:
        n_pass = sum(1 for r in vis.values() if (r["score"] or 0) >= 0.5)
        if n_pass == len(vis) and n_pass > 0:
            s.append(f"Vision: {n_pass}/{len(vis)} — only multimodal model in this stack")

    if psus:
        last = psus[-1]["tok_per_sec"]
        first = next((r["tok_per_sec"] for r in psus if r["tok_per_sec"] > 0), 0)
        if last >= 25:
            s.append(f"Sustained {last:.1f} tok/s")
        elif last < 10:
            w.append(f"Sustained only {last:.1f} tok/s (below Usable threshold)")
        if first > 0 and (first - last) / first * 100 > 20:
            w.append(f"Sustained throttle {(first - last) / first * 100:.0f}% from run 1 to last")

    if pdim:
        peak_rams = [v["peak_ram_mb"] for v in pdim.values() if v.get("peak_ram_mb")]
        if peak_rams:
            mr = max(peak_rams)
            if mr > 5500:
                w.append(f"Peak RAM {mr:.0f} MB at large ctx — tight on 16GB")
            elif mr < 4000:
                s.append(f"Peak RAM only {mr:.0f} MB (low footprint)")
    return s, w


# ---- Recommender -----------------------------------------------------------
USE_CASES = {
    "HITL chat (Q&A, summarization)": "hitl",
    "Trusted agent loops (file ops, tool use)": "agent",
    "Code generation / repair": "code",
    "Hostile-input safety (untrusted text)": "safety",
    "Multimodal (chart QA / OCR)": "vision",
    "Compact / minimal RAM footprint": "compact",
}


def recommend(data: dict, ram_gb: float, disk_gb: float, use_case_key: str,
              multimodal_required: bool) -> list[tuple[str, float, str]]:
    """Deterministic rule-based recommender.

    Hardware filter: model must fit in (disk_gb_required + 7GB OS/KV/activation overhead) ≤ ram_gb,
    and disk_gb_model ≤ free disk. Then ranks remaining candidates by a use-case-specific score
    derived from measured benchmark data.
    """
    candidates = list(MODELS.keys())
    candidates = [m for m in candidates if MODELS[m]["disk_gb"] + 7 <= ram_gb]
    candidates = [m for m in candidates if MODELS[m]["disk_gb"] <= disk_gb]
    if multimodal_required:
        candidates = [m for m in candidates if MODELS[m]["multimodal"]]

    results = []
    for m in candidates:
        rationale_bits = []
        score = 0.0

        if use_case_key == "hitl":
            hitl = data["hitl"].get(m, {})
            scores = [r["score"] or 0 for r in hitl.values()]
            pass_rate = (sum(1 for x in scores if x >= 0.5) / len(scores)) if scores else 0.0
            psus = data["psus"].get(m, [])
            tps = psus[-1]["tok_per_sec"] if psus else 0.0
            score = pass_rate * 60 + min(tps, 30)
            rationale_bits.append(f"capability {pass_rate*100:.0f}%, sustained {tps:.0f} tok/s")

        elif use_case_key == "agent":
            a = data["agent"].get(m)
            inj = data["inj"].get(m, {})
            n_resist = sum(1 for r in inj.values() if (r["score"] or 0) >= 0.5)
            if a:
                score = a["score"] * 60 + a["valid_pct"] * 0.2 + max(0, 5 - a["steps"]) * 4
                rationale_bits.append(f"agent {a['score']:.2f} in {a['steps']} steps, valid {a['valid_pct']:.0f}%")
            if inj:
                rationale_bits.append(f"injection resist {n_resist}/{len(inj)}")

        elif use_case_key == "code":
            code = data["hitl"].get(m, {}).get("code_is_prime")
            a = data["agent"].get(m)
            specialist_bonus = 30 if MODELS[m]["expected_role"] == "code-specialist" else 0
            score = (code["score"] if code else 0) * 30 + (a["score"] if a else 0) * 30 + specialist_bonus
            rationale_bits.append(
                f"code probe {(code['score'] if code else 0):.2f}; "
                f"{'specialist bonus' if specialist_bonus else 'generalist'}; "
                f"agent {(a['score'] if a else 0):.2f}"
            )

        elif use_case_key == "safety":
            inj = data["inj"].get(m, {})
            scores = [r["score"] or 0 for r in inj.values()]
            n_resist = sum(1 for x in scores if x >= 0.5)
            score = n_resist * 20
            rationale_bits.append(f"injection resist {n_resist}/{len(scores) or 5}")
            if scores and n_resist < 3:
                rationale_bits.append("⚠ NOT recommended for hostile inputs")

        elif use_case_key == "vision":
            vis = data["vis"].get(m, {})
            scores = [r["score"] or 0 for r in vis.values()]
            n_pass = sum(1 for x in scores if x >= 0.5)
            score = n_pass * 30 + (50 if MODELS[m]["multimodal"] else 0)
            rationale_bits.append(
                f"multimodal {'YES' if MODELS[m]['multimodal'] else 'NO'}; "
                f"vision {n_pass}/{len(scores) or 0}"
            )

        elif use_case_key == "compact":
            score = 100 - MODELS[m]["disk_gb"] * 12
            rationale_bits.append(f"{MODELS[m]['disk_gb']} GB on disk")

        results.append((m, score, "; ".join(rationale_bits)))

    results.sort(key=lambda x: -x[1])
    return results


# ---- Pages -----------------------------------------------------------------
def page_overview(data: dict) -> None:
    st.title("🧪 Local Model Bench")
    st.caption(
        "Benchmarks measured on M3 Air, 16GB unified RAM, fanless. "
        "Source: `results/runs.sqlite`. Methodology in PLAN.md / ARCHITECTURE.md."
    )

    rows = []
    for m, meta in MODELS.items():
        hitl = data["hitl"].get(m, {})
        scores = [r["score"] or 0 for r in hitl.values()]
        pass_rate = (sum(1 for s in scores if s >= 0.5) / len(scores)) if scores else 0.0
        psus = data["psus"].get(m, [])
        sus_tps = psus[-1]["tok_per_sec"] if psus else None

        a = data["agent"].get(m)
        inj = data["inj"].get(m, {})
        n_resist = sum(1 for r in inj.values() if (r["score"] or 0) >= 0.5) if inj else None
        vis = data["vis"].get(m, {})
        n_vision = sum(1 for r in vis.values() if (r["score"] or 0) >= 0.5) if vis else None

        rows.append({
            "Model": m,
            "Role hypothesis": meta["expected_role"],
            "HITL tier": hitl_tier(pass_rate, sus_tps) if hitl else "untested",
            "HITL pass": f"{int(pass_rate * len(scores))}/{len(scores)}" if scores else "–",
            "Agent tier": agent_tier(a["score"], a["steps"], a["valid_pct"], a["done"]) if a else "untested",
            "Injection": f"{n_resist}/5" if n_resist is not None else "–",
            "Vision": f"{n_vision}/3" if n_vision is not None else "–",
            "Sustained": (
                f"{sus_tps:.1f} tok/s ({sustained_tier(sus_tps, psus[-1]['throttle_pct_vs_first'])})"
                if psus else "untested"
            ),
        })
    st.subheader("Verdict matrix")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("† = perf sweep not run; HITL tier classified on capability pass-rate alone.")

    st.subheader("Demo punchlines")
    st.markdown("""
- **Role-based selection beats parameter count.** qwen2.5-coder:7b solves agent loops in 3 optimal steps; same-size llama3.1:8b fails the same task — but is the *only* model with perfect injection resistance (5/5).
- **phi4-mini:3.8b is the surprise all-rounder.** 5/5 capability, Great-tier agent, second-best safety (4/5), lowest peak RAM at every context size.
- **The agent winner is the safety loser.** qwen2.5-coder:7b is best at agent loops AND worst at injection resistance (1/5). Don't put it behind retrieval over untrusted content.
- **gemma3:4b owns the multimodal axis** — 3/3 perfect on vision tasks at 22-38 tok/s. Nothing else can attempt them.
- **Thermal throttling is order-dependent** on a fanless M3: phi4-mini hit 30 tok/s cold, dropped to 14 sustained.
- **qwen3:4b's three thinking modes are not equivalent** — same JSON prompt scores 1.00 with default/`<think>` stripping vs 0.00 with `think:false` (which inlines reasoning into the output and breaks the parser).
- **phi3.5 hallucinated agent output**: invented a TODO line not in the source notes.txt — passing JSON validity, signaling done, while the answer was wrong.
""")

    st.subheader("Quick links")
    st.markdown("""
- **Data**: [`results/runs.sqlite`](results/runs.sqlite) — all rows here
- **Markdown leaderboard**: [`dashboards/leaderboard.md`](dashboards/leaderboard.md) — static export of this page
- **Memory layer demo**: `python swap_model.py demo` — seed a session, summarize, swap models
- **Re-run**: `python -m runners.run_bench` (capability), `python -m runners.measure_perf --all` (perf), `python -m runners.agent_loop --all` (agent), `python -m runners.run_bench --prompts-yaml evals/injection_tasks.yaml --tag injection` (injection), `python -m runners.vision_eval` (vision)
""")


def page_model_detail(data: dict) -> None:
    st.title("Per-Model Detail")

    model_id = st.selectbox(
        "Model",
        options=list(MODELS.keys()),
        format_func=lambda x: MODELS[x]["display"],
    )
    meta = MODELS[model_id]

    st.subheader(meta["display"])
    cols = st.columns(4)
    cols[0].metric("Params", f"{meta['params_b']}B")
    cols[1].metric("Disk", f"{meta['disk_gb']} GB")
    cols[2].metric("Context window", f"{meta['ctx']:,}")
    cols[3].metric("Multimodal", "yes" if meta["multimodal"] else "no")

    s, w = derive_strengths_weaknesses(model_id, data)
    col_s, col_w = st.columns(2)
    with col_s:
        st.markdown("### ✅ Strengths")
        if s:
            for x in s: st.markdown(f"- {x}")
        else:
            st.caption("(none derived)")
    with col_w:
        st.markdown("### ⚠ Weaknesses")
        if w:
            for x in w: st.markdown(f"- {x}")
        else:
            st.caption("(none derived)")

    hitl = data["hitl"].get(model_id, {})
    if hitl:
        st.markdown("### Capability probe")
        df = pd.DataFrame([
            {"Prompt": pid, "Score": r["score"] or 0,
             "Tok/s (gen)": round(r["tok_per_sec"] or 0, 1),
             "Output (first 200 chars)": (r["output"] or "").replace("\n", " ")[:200]}
            for pid, r in sorted(hitl.items())
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

    a = data["agent"].get(model_id)
    if a:
        st.markdown("### Agent loop (Track B)")
        cols = st.columns(4)
        cols[0].metric("Score", f"{a['score']:.2f}")
        cols[1].metric("Steps", a["steps"])
        cols[2].metric("Valid JSON", f"{a['valid_pct']:.0f}%")
        cols[3].metric("Done?", "yes" if a["done"] else "no")
        with st.expander("Final filesystem state"):
            st.code((a["output"] or "")[:2000], language="json")

    inj = data["inj"].get(model_id, {})
    if inj:
        st.markdown("### Injection resistance")
        df = pd.DataFrame([
            {"Attack": pid, "Resisted": "✅" if (r["score"] or 0) >= 0.5 else "❌",
             "Output (first 200 chars)": (r["output"] or "").replace("\n", " ")[:200]}
            for pid, r in sorted(inj.items())
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

    vis = data["vis"].get(model_id, {})
    if vis:
        st.markdown("### Vision")
        df = pd.DataFrame([
            {"Task": pid, "Score": r["score"] or 0,
             "Output": (r["output"] or "").replace("\n", " ")[:200]}
            for pid, r in sorted(vis.items())
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

    pdim = data["pdim"].get(model_id, {})
    if pdim:
        st.markdown("### Dimensional performance (think:false, num_predict=500)")
        df = pd.DataFrame([
            {"ctx": c, "tok/s": round(v["tok_per_sec"], 1),
             "TTFT (s)": round(v["ttft_ms"] / 1000, 1) if v["ttft_ms"] else None,
             "peak RAM (MB)": int(v["peak_ram_mb"]) if v["peak_ram_mb"] else None}
            for c, v in sorted(pdim.items())
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.line_chart(df.set_index("ctx")[["tok/s"]])

    psus = data["psus"].get(model_id, [])
    if psus:
        st.markdown("### Sustained throughput (5×, no cooldown, ctx=4096)")
        df = pd.DataFrame([
            {"run": r["run_index"], "tok/s": round(r["tok_per_sec"], 1),
             "throttle vs run 1 (%)": round(r["throttle_pct_vs_first"], 1),
             "peak RAM (MB)": int(r["peak_ram_mb"]) if r["peak_ram_mb"] else None}
            for r in psus
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.line_chart(df.set_index("run")[["tok/s"]])


def page_recommender(data: dict) -> None:
    st.title("Hardware-Aware Recommender")
    st.caption(
        "Deterministic rule-based picker. Filters by RAM/disk fit, then ranks by use-case fit "
        "using measured benchmark data from this stack. (No live LLM-driven scoring — same input, same output.)"
    )

    cols = st.columns(3)
    ram_gb = cols[0].number_input("RAM (GB unified)", min_value=2, max_value=128, value=16)
    disk_gb = cols[1].number_input("Free disk (GB)", min_value=1, max_value=2000, value=20)
    gpu = cols[2].selectbox(
        "GPU/CPU profile",
        ["Apple Silicon (M-series)", "NVIDIA discrete", "Intel/AMD CPU only", "Other / unknown"],
    )

    use_case_label = st.selectbox("Primary use case", list(USE_CASES.keys()))
    multimodal_required = st.checkbox("Must accept image input (multimodal required)", value=False)

    if gpu == "Intel/AMD CPU only":
        st.warning(
            "These benchmarks are from Apple Silicon (M3). On x86 CPU-only setups expect 2-5× slower "
            "tok/s. Treat the throughput numbers as upper bounds."
        )
    elif gpu == "NVIDIA discrete":
        st.info("NVIDIA discrete GPUs typically run these models faster than M3 — perf below is a floor.")

    recs = recommend(data, ram_gb, disk_gb, USE_CASES[use_case_label], multimodal_required)

    st.markdown("---")
    if not recs:
        st.error(
            "No models match these constraints. Try increasing RAM/disk, "
            "or remove the multimodal requirement."
        )
        return

    st.markdown(f"### Top picks for *{use_case_label}*")
    for i, (model_id, score, rationale) in enumerate(recs[:5], 1):
        meta = MODELS[model_id]
        with st.container(border=True):
            top = st.columns([3, 1, 1, 1])
            top[0].markdown(f"**#{i} {meta['display']}**")
            top[0].caption(f"`{model_id}` — role hypothesis: *{meta['expected_role']}*")
            top[1].metric("Fit score", f"{score:.0f}")
            top[2].metric("Disk", f"{meta['disk_gb']} GB")
            top[3].metric("Params", f"{meta['params_b']}B")
            st.caption(f"**Why:** {rationale}")

    with st.expander("Hardware-fit reasoning"):
        st.markdown("""
- **RAM filter**: model passes if *(disk size + 7 GB overhead)* ≤ entered RAM. The 7 GB covers macOS+apps (~5 GB) plus KV cache and activations (~2 GB).
- **Disk filter**: model's quantized weights file ≤ entered free disk.
- **Multimodal filter**: only `gemma3:4b` is multimodal in this stack.
- **Use-case scoring** then breaks ties with measured data — capability pass rate, agent score, injection resistance, vision pass rate, etc.
""")


# ---- Main ------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Local Model Bench",
        page_icon="🧪",
        layout="wide",
    )
    if not DB.exists():
        st.error(f"DB not found: {DB}. Run `python -m runners.run_bench` first.")
        return
    data = load_data()

    st.sidebar.title("🧪 Local Model Bench")
    page = st.sidebar.radio("Page", ["Overview", "Per-Model Detail", "Hardware Recommender"])
    st.sidebar.markdown("---")
    st.sidebar.caption(f"DB: `{DB.name}`")
    st.sidebar.caption(f"Models registered: {len(MODELS)}")

    if page == "Overview":
        page_overview(data)
    elif page == "Per-Model Detail":
        page_model_detail(data)
    else:
        page_recommender(data)


main()
