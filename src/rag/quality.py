# src/rag/quality.py
from __future__ import annotations
import os, re, json, hashlib
from typing import Any, Dict, List, Tuple
import streamlit as st
from src.config import settings, QUALITY_REPORT_PATH

# LlamaIndex 문서 모델
try:
    # 0.12.x
    from llama_index.core.schema import Document
except Exception:  # 안전 폴백
    from llama_index.core import Document  # type: ignore

_ws_re = re.compile(r"[ \t\f\v]+")

def _clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = _ws_re.sub(" ", s)
    return s.strip()

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def get_opt() -> Dict[str, Any]:
    return {
        "chunk_size": int(st.session_state.get("opt_chunk_size", settings.CHUNK_SIZE)),
        "chunk_overlap": int(st.session_state.get("opt_chunk_overlap", settings.CHUNK_OVERLAP)),
        "min_chars": int(st.session_state.get("opt_min_chars", settings.MIN_CHARS_PER_DOC)),
        "dedup": bool(st.session_state.get("opt_dedup", settings.DEDUP_BY_TEXT_HASH)),
        "skip_low_text": bool(st.session_state.get("opt_skip_low_text", settings.SKIP_LOW_TEXT_DOCS)),
        "pre_summarize": bool(st.session_state.get("opt_pre_summarize", settings.PRE_SUMMARIZE_DOCS)),
    }

def maybe_summarize_docs(docs: List[Any]) -> None:
    """옵션이 켜지면 문서 요약을 metadata['doc_summary']에 저장(실패 무시)."""
    if not docs or not get_opt()["pre_summarize"]:
        return
    try:
        from llama_index.core import Settings
        for d in docs:
            md = getattr(d, "metadata", {}) or {}
            if "doc_summary" in md:
                continue
            text = (getattr(d, "text", "") or "")[:4000]
            if not text:
                continue
            prompt = (
                "다음 문서를 교사 시각에서 5줄 이내 핵심 bullet로 요약하라.\n"
                "교재 단원/개념/예문/핵심 규칙을 간단히 표시하라.\n\n"
                f"[문서 내용]\n{text}"
            )
            try:
                resp = Settings.llm.complete(prompt)
                summary = getattr(resp, "text", None) or str(resp)
                md["doc_summary"] = summary.strip()
                # 새 Document로 복제해서 metadata를 반영
                d_idx = docs.index(d)
                docs[d_idx] = Document(text=getattr(d, "text", ""), metadata=md)
            except Exception:
                pass
    except Exception:
        pass

def _clone_with_text_and_meta(d: Any, new_text: str, new_meta: Dict[str, Any]) -> Document:
    """원본 d에서 텍스트/메타를 반영한 새 Document 생성(직접 대입 금지)."""
    try:
        # 파일명/기타 메타를 최대한 보존
        md = dict(getattr(d, "metadata", {}) or {})
        md.update(new_meta)
    except Exception:
        md = dict(new_meta)
    return Document(text=new_text, metadata=md)

def preprocess_docs(docs: List[Any], seen_hashes: set, min_chars: int, dedup: bool) -> Tuple[List[Any], Dict[str, Any]]:
    """텍스트 정리/저품질 필터/중복제거 후 유효 문서만 반환 + 통계."""
    kept: List[Any] = []
    stats = {"input_docs": len(docs), "kept": 0, "skipped_low_text": 0, "skipped_dup": 0, "total_chars": 0}
    for d in docs:
        raw = getattr(d, "text", "") or ""
        t = _clean_text(raw)
        if len(t) < min_chars:
            stats["skipped_low_text"] += 1
            continue
        h = _sha1(t)
        if dedup and h in seen_hashes:
            stats["skipped_dup"] += 1
            continue

        md = dict(getattr(d, "metadata", {}) or {})
        md["text_hash"] = h

        nd = _clone_with_text_and_meta(d, t, md)
        kept.append(nd)
        seen_hashes.add(h)
        stats["kept"] += 1
        stats["total_chars"] += len(t)
    return kept, stats

def load_quality_report(path: str = QUALITY_REPORT_PATH) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"summary": {}, "files": {}}

def save_quality_report(data: Dict[str, Any], path: str = QUALITY_REPORT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
