"""Memory / context-swap layer.

Implements the 4-layer design from PLAN.md §12 and ARCHITECTURE.md §3:
  1. Raw artifacts (filesystem)              — handled outside this module
  2. Event log (SQLite eval_runs)            — handled by run_bench / measure_perf
  3. Compressed summaries (rolling, embedded)— this module
  4. Vector memory (sqlite-vec)              — this module

Two memory types share the vec_memory virtual table via ref_type/ref_id columns:
- 'summary'        : rolling per-session summaries
- 'memory_object'  : structured facts (decision/constraint/entity/fact)

Embeddings are produced by nomic-embed-text (768 dim). PLAN.md and ARCHITECTURE.md
both flag the cross-model embedding-stability gotcha: do not change the embedder
without re-embedding everything.
"""
from __future__ import annotations

import json
import struct
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

from runners.run_ollama import call_model, embed
from runners.db import connect, init_db, ensure_vec_table

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
SUMMARY_EVERY_N_TURNS = 6


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _vec_to_blob(vec: list[float]) -> bytes:
    """Pack a Python float list into 4-byte little-endian floats."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _vec_from_blob(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


@dataclass
class MemoryHit:
    ref_type: str
    ref_id: str
    distance: float
    text: str
    extra: dict


class MemoryStore:
    def __init__(self, conn: Optional[sqlite3.Connection] = None):
        if conn is None:
            init_db()
            conn = connect()
        self.conn = conn
        self.has_vec = ensure_vec_table(conn, dim=EMBED_DIM)

    # ---- sessions / turns -------------------------------------------------

    def start_session(self, session_id: Optional[str] = None, active_model: str = "") -> str:
        sid = session_id or f"sess-{uuid.uuid4().hex[:8]}"
        self.conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, created_at, active_model) VALUES (?,?,?)",
            (sid, _now(), active_model),
        )
        self.conn.commit()
        return sid

    def get_session(self, session_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT session_id, created_at, active_model FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        tokens: Optional[int] = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO turns (session_id, role, content, tokens, ts) VALUES (?,?,?,?,?)",
            (session_id, role, content, tokens, _now()),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_turns(self, session_id: str, limit: Optional[int] = None) -> list[dict]:
        sql = "SELECT turn_id, role, content, tokens, ts FROM turns WHERE session_id=? ORDER BY turn_id"
        rows = self.conn.execute(sql, (session_id,)).fetchall()
        rows = [dict(r) for r in rows]
        if limit:
            rows = rows[-limit:]
        return rows

    def turns_since_last_summary(self, session_id: str) -> int:
        last = self.conn.execute(
            "SELECT MAX(up_to_turn) AS up FROM summaries WHERE session_id=?",
            (session_id,),
        ).fetchone()
        last_turn = (last["up"] if last and last["up"] is not None else 0)
        latest = self.conn.execute(
            "SELECT MAX(turn_id) AS m FROM turns WHERE session_id=?",
            (session_id,),
        ).fetchone()
        latest_id = latest["m"] if latest and latest["m"] is not None else 0
        return max(0, latest_id - last_turn)

    # ---- memory_objects ---------------------------------------------------

    def add_memory_object(
        self,
        type: str,                      # decision | fact | constraint | entity
        summary: str,
        *,
        session_id: Optional[str] = None,
        project: Optional[str] = None,
        evidence: Optional[list] = None,
        recheck: Optional[str] = None,
        date: Optional[str] = None,
    ) -> str:
        mid = f"mo-{uuid.uuid4().hex[:8]}"
        emb = embed(summary, model=EMBED_MODEL)
        if len(emb) != EMBED_DIM:
            raise RuntimeError(f"embedding dim mismatch: got {len(emb)}, want {EMBED_DIM}")
        self.conn.execute(
            "INSERT INTO memory_objects (id,type,session_id,date,project,summary,evidence_json,recheck,embedding) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                mid, type, session_id,
                date or _now()[:10], project, summary,
                json.dumps(evidence or []),
                recheck,
                _vec_to_blob(emb),
            ),
        )
        if self.has_vec:
            self.conn.execute(
                "INSERT INTO vec_memory (embedding, ref_type, ref_id) VALUES (?,?,?)",
                (_vec_to_blob(emb), "memory_object", mid),
            )
        self.conn.commit()
        return mid

    def list_memory_objects(self, type: Optional[str] = None) -> list[dict]:
        if type:
            rows = self.conn.execute(
                "SELECT id,type,session_id,date,project,summary,evidence_json,recheck "
                "FROM memory_objects WHERE type=? ORDER BY date",
                (type,),
            )
        else:
            rows = self.conn.execute(
                "SELECT id,type,session_id,date,project,summary,evidence_json,recheck "
                "FROM memory_objects ORDER BY date"
            )
        return [dict(r) for r in rows]

    # ---- rolling summaries ------------------------------------------------

    def make_rolling_summary(
        self,
        session_id: str,
        summarizer_model: str,
        max_input_chars: int = 12000,
    ) -> Optional[str]:
        """Summarize all turns since the last summary using the given model.

        Returns the generated summary text, or None if there are no new turns.
        """
        turns = self.get_turns(session_id)
        if not turns:
            return None
        last_summary_row = self.conn.execute(
            "SELECT summary, up_to_turn FROM summaries WHERE session_id=? "
            "ORDER BY up_to_turn DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        last_up_to = last_summary_row["up_to_turn"] if last_summary_row else 0
        new_turns = [t for t in turns if t["turn_id"] > last_up_to]
        if not new_turns:
            return None

        prior = last_summary_row["summary"] if last_summary_row else ""
        transcript = "\n".join(
            f"[{t['role']}] {t['content']}" for t in new_turns
        )[-max_input_chars:]
        prompt = (
            "You are a session summarizer. Produce a concise (~150 words) summary "
            "of the new turns below, integrating any prior summary. Preserve concrete "
            "decisions, file paths, model names, and numeric findings verbatim.\n\n"
            f"Prior summary:\n{prior}\n\n"
            f"New turns:\n{transcript}\n\n"
            "Updated summary:"
        )
        result = call_model(summarizer_model, prompt, ctx_size=8192, num_predict=400)
        if result.error or not result.output.strip():
            return None
        summary_text = result.output.strip()

        emb = embed(summary_text, model=EMBED_MODEL)
        new_up_to = new_turns[-1]["turn_id"]
        self.conn.execute(
            "INSERT INTO summaries (session_id, up_to_turn, summary, embedding, ts) VALUES (?,?,?,?,?)",
            (session_id, new_up_to, summary_text, _vec_to_blob(emb), _now()),
        )
        if self.has_vec:
            self.conn.execute(
                "INSERT INTO vec_memory (embedding, ref_type, ref_id) VALUES (?,?,?)",
                (_vec_to_blob(emb), "summary", f"{session_id}:{new_up_to}"),
            )
        self.conn.commit()
        return summary_text

    def latest_summary(self, session_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT summary, up_to_turn, ts FROM summaries WHERE session_id=? "
            "ORDER BY up_to_turn DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None

    # ---- vector search ----------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[MemoryHit]:
        """Vector search over both summaries and memory_objects."""
        if not self.has_vec:
            return []
        qvec = embed(query, model=EMBED_MODEL)
        rows = self.conn.execute(
            "SELECT rowid, ref_type, ref_id, distance FROM vec_memory "
            "WHERE embedding MATCH ? AND k=? ORDER BY distance",
            (_vec_to_blob(qvec), k),
        ).fetchall()
        hits: list[MemoryHit] = []
        for r in rows:
            text, extra = self._resolve_ref(r["ref_type"], r["ref_id"])
            if text:
                hits.append(
                    MemoryHit(
                        ref_type=r["ref_type"],
                        ref_id=r["ref_id"],
                        distance=r["distance"],
                        text=text,
                        extra=extra,
                    )
                )
        return hits

    def _resolve_ref(self, ref_type: str, ref_id: str) -> tuple[str, dict]:
        if ref_type == "summary":
            session_id, _, up_to = ref_id.partition(":")
            row = self.conn.execute(
                "SELECT summary, ts FROM summaries WHERE session_id=? AND up_to_turn=?",
                (session_id, int(up_to or 0)),
            ).fetchone()
            return (row["summary"], {"ts": row["ts"], "session_id": session_id}) if row else ("", {})
        if ref_type == "memory_object":
            row = self.conn.execute(
                "SELECT type, summary, date, project, evidence_json FROM memory_objects WHERE id=?",
                (ref_id,),
            ).fetchone()
            if not row:
                return "", {}
            return row["summary"], {
                "type": row["type"],
                "date": row["date"],
                "project": row["project"],
                "evidence": json.loads(row["evidence_json"] or "[]"),
            }
        return "", {}

    # ---- preamble + swap --------------------------------------------------

    def build_preamble(
        self,
        session_id: str,
        *,
        k_similar: int = 4,
        include_recent_turns: int = 0,
    ) -> str:
        """Compose a [CONTEXT FROM PRIOR WORK] block for the next active model."""
        latest = self.latest_summary(session_id)
        recent_summary = latest["summary"] if latest else "(no prior summary)"

        # Highlights via vector similarity to the recent summary.
        highlights: list[MemoryHit] = []
        if self.has_vec and latest:
            highlights = [
                h for h in self.search(latest["summary"], k=k_similar + 4)
                if not (h.ref_type == "summary" and h.ref_id == f"{session_id}:{latest['up_to_turn']}")
            ][:k_similar]

        # Structured facts: list all decisions and constraints.
        decisions = self.list_memory_objects(type="decision")
        constraints = self.list_memory_objects(type="constraint")

        parts = ["[CONTEXT FROM PRIOR WORK]"]
        parts.append(f"Recent summary:\n{recent_summary}")
        if highlights:
            parts.append("Earlier highlights (vector-retrieved):")
            for h in highlights:
                parts.append(f"  - ({h.ref_type}) {h.text[:240]}")
        if decisions:
            parts.append("Key decisions:")
            for d in decisions:
                parts.append(f"  - {d['summary']}")
        if constraints:
            parts.append("Constraints:")
            for c in constraints:
                parts.append(f"  - {c['summary']}")
        if include_recent_turns > 0:
            tail = self.get_turns(session_id, limit=include_recent_turns)
            if tail:
                parts.append("Most recent turns:")
                for t in tail:
                    parts.append(f"  [{t['role']}] {t['content'][:300]}")
        parts.append("[/CONTEXT]")
        parts.append("You are continuing this work. Acknowledge briefly, then await the next request.")
        return "\n".join(parts)

    def swap_active_model(
        self,
        session_id: str,
        new_model: str,
        *,
        warm: bool = True,
        keep_alive_seconds: int = 600,
    ) -> dict:
        """Update active_model, build preamble, optionally warm the new model.

        Returns dict with: preamble, warm_result (CallResult-as-dict or None).
        """
        if not self.get_session(session_id):
            raise ValueError(f"unknown session {session_id!r}")
        self.conn.execute(
            "UPDATE sessions SET active_model=? WHERE session_id=?",
            (new_model, session_id),
        )
        self.conn.commit()
        preamble = self.build_preamble(session_id)
        warm_dict = None
        if warm:
            # Single throwaway call to load weights and prime KV cache. Keep
            # model resident so the next user turn doesn't pay cold-load again.
            # think=False is critical here: thinking models (qwen3) would
            # otherwise consume the num_predict budget on hidden reasoning and
            # the warm-up would emit no visible token at all.
            result = call_model(
                new_model,
                preamble + "\n\nReply with one short sentence acknowledging prior context has loaded.",
                ctx_size=8192,
                num_predict=128,
                think=False,
                keep_alive=keep_alive_seconds,
            )
            warm_dict = {
                "output": result.output,
                "ttft_ms": result.ttft_ms,
                "latency_ms": result.latency_ms,
                "cold_load_ms": result.cold_load_ms,
                "tokens_out": result.tokens_out,
                "error": result.error,
            }
        return {"preamble": preamble, "warm": warm_dict}
