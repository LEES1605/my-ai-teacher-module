# ===== [01] IMPORTS ==========================================================
from __future__ import annotations

import os
from typing import Callable, Mapping, Dict, List, Any, Set

import streamlit as st

from .storage import _make_storage_context
from .checkpoint import (
    _load_checkpoint,
    _mark_done,
    save_checkpoint_copy_to_persist,
)
from .quality import (
    get_opt,
    preprocess_docs,
    maybe_summarize_docs,
    load_quality_report,
    save_quality_report,
)

# ===== [02] TYPE HINTS =======================================================
UpdatePct = Callable[[int, str | None], None]
UpdateMsg = Callable[[str], None]
Manifest = Dict[str, Dict[str, Any]]

# ===== [03] SAFE IMPORT (RUNTIME) ============================================
def _safe_import_llama():
    try:
        from llama_index.core import VectorStoreIndex, load_index_from_storage
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.readers.google import GoogleDriveReader
        return VectorStoreIndex, load_index_from_storage, SentenceSplitter, GoogleDriveReader
    except Exception as e:
        st.error("llama_index 모듈 로드 중 문제가 발생했습니다. requirements 버전을 확인하세요.")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()
        raise  # for type-checkers

# ===== [04] MAIN ENTRY =======================================================
def build_index_with_checkpoint(
    update_pct: UpdatePct,
    update_msg: UpdateMsg,
    gdrive_folder_id: str,
    gcp_creds: Mapping[str, Any],
    persist_dir: str,
    remote_manifest: Manifest,
    should_stop: Callable[[], bool] | None = None,
):
    """
    파일ID 단위 처리 → 각 파일 완주 후 저장 & 체크포인트 기록.
    전처리/중복제거/문서요약 → SentenceSplitter 청킹 → 인덱스 누적.
    중지 버튼(should_stop=True) 감지 시 '현재 파일까지' 저장 후 안전 종료.
    """
    VectorStoreIndex, load_index_from_storage, SentenceSplitter, GoogleDriveReader = _safe_import_llama()

    # ----- [04.1] STOP SWITCH -------------------------------------------------
    if should_stop is None:
        should_stop = lambda: False  # noqa: E731

    # ----- [04.2] OPTIONS -----------------------------------------------------
    opt = get_opt()

    # ----- [04.3] LOADER & CHECKPOINT ----------------------------------------
    update_pct(15, "Drive 리더 초기화")
    loader = GoogleDriveReader(service_account_key=gcp_creds)

    # persist_dir에 남아있을 수 있는 checkpoint.json도 자동 로드
    cp: Dict[str, Dict[str, Any]] = _load_checkpoint(also_from_persist_dir=persist_dir)

# ----- [04.4] TODO LIST (변경분만 처리) -----------------------------------
todo_ids: List[str] = []
for fid, manifest_meta in remote_manifest.items():
    done = cp.get(fid)
    if done and done.get("md5") and manifest_meta.get("md5") and done["md5"] == manifest_meta["md5"]:
        continue
    todo_ids.append(fid)

total = len(remote_manifest)
pending = len(todo_ids)
done_cnt = total - pending
update_pct(30, f"문서 목록 불러오는 중 • 전체 {total}개, 이번에 처리 {pending}개")

    # ----- [04.5] STORAGE CONTEXT & PRELOAD ----------------------------------
    os.makedirs(persist_dir, exist_ok=True)
    storage_context = _make_storage_context(persist_dir)
    try:
        _ = load_index_from_storage(storage_context)
    except Exception:
        # 저장된 인덱스가 없거나, 일부 파일만 있을 수 있음 → 통과
        pass

    # ===== [05] QUALITY REPORT & STATE =======================================
    qrep = load_quality_report()
    qrep.setdefault("summary", {}).setdefault("total_docs", total)
    for k in ("processed_docs", "kept_docs", "skipped_low_text", "skipped_dup", "total_chars"):
        qrep["summary"].setdefault(k, 0)
    qrep.setdefault("files", {})

    # 해시 중복 제거용 집합 — 명시적 타입 (mypy)
    seen_hashes: Set[str] = set()

    # ===== [06] EARLY RETURN (NO PENDING) ====================================
    if pending == 0:
        update_pct(95, "변경 없음 → 저장본 그대로 사용")
        return load_index_from_storage(storage_context)

    # ===== [07] CHUNKER INITIALIZE ===========================================
    splitter = SentenceSplitter(
        chunk_size=int(opt["chunk_size"]),
        chunk_overlap=int(opt["chunk_overlap"]),
    )

# ===== [08] MAIN LOOP =====================================================
for i, fid in enumerate(todo_ids, start=1):
    # ----- [08.1] STOP CHECK ---------------------------------------------
    if should_stop():
        update_msg("🛑 중지 요청 감지 — 현재까지 저장 후 종료합니다.")
        break

    # ----- [08.2] LOAD SINGLE FILE ---------------------------------------
    fmeta: Dict[str, Any] = remote_manifest.get(fid, {})
    fname = str(fmeta.get("name") or fid)
    update_msg(f"전처리 • {fname} ({done_cnt + i}/{total})")

    try:
        # 일부 버전은 file_ids 파라미터를 지원하지 않아 TypeError가 날 수 있음
        docs = loader.load_data(file_ids=[fid])
    except TypeError:
        st.error(
            "GoogleDriveReader 버전이 오래되어 file_ids 옵션을 지원하지 않습니다. "
            "requirements 업데이트가 필요합니다."
        )
        st.stop()

    # ----- [08.3] PREPROCESS & DEDUP -------------------------------------
    kept, stats = preprocess_docs(
        docs,
        seen_hashes,
        min_chars=int(opt["min_chars"]),
        dedup=bool(opt["dedup"]),
    )
    maybe_summarize_docs(kept)

    # ----- [08.4] QUALITY REPORT UPDATE ----------------------------------
    from_name = fmeta.get("name")
    qrep.setdefault("files", {})[fid] = {
        "name": from_name or fid,
        "md5": fmeta.get("md5"),
        "modifiedTime": fmeta.get("modifiedTime"),
        "kept": stats["kept"],
        "skipped_low_text": stats["skipped_low_text"],
        "skipped_dup": stats["skipped_dup"],
        "total_chars": stats["total_chars"],
    }
    qs = qrep.setdefault("summary", {})
    qs["processed_docs"] = int(qs.get("processed_docs", 0)) + 1
    qs["kept_docs"] = int(qs.get("kept_docs", 0)) + int(stats["kept"])
    qs["skipped_low_text"] = int(qs.get("skipped_low_text", 0)) + int(stats["skipped_low_text"])
    qs["skipped_dup"] = int(qs.get("skipped_dup", 0)) + int(stats["skipped_dup"])
    qs["total_chars"] = int(qs.get("total_chars", 0)) + int(stats["total_chars"])
    save_quality_report(qrep)

    # ----- [08.5] SKIP CASE (NO KEPT) ------------------------------------
    if int(stats["kept"]) == 0:
        _mark_done(cp, fid, fmeta)                          # 완료 체크
        save_checkpoint_copy_to_persist(cp, persist_dir)    # persist_dir에도 동기화
        pct = 30 + int((i / max(1, pending)) * 60)
        update_pct(pct, f"건너뜀 • {fname} (저품질/중복)")
        if should_stop():
            update_msg("🛑 중지 요청 감지 — 현재까지 저장 후 종료합니다.")
            break
        continue

    # ----- [08.6] INDEX APPEND ------------------------------------------
    update_msg(f"인덱스 생성 • {fname} ({done_cnt + i}/{total})")
    try:
        VectorStoreIndex.from_documents(
            kept,
            storage_context=storage_context,
            show_progress=False,
            transformations=[splitter],
        )
        storage_context.persist(persist_dir=persist_dir)     # 부분 저장
        _mark_done(cp, fid, fmeta)                           # 파일 완료 기록
        save_checkpoint_copy_to_persist(cp, persist_dir)     # persist_dir에도 동기화
    except Exception as e:
        st.error(f"인덱스 생성 중 오류: {fname}")
        with st.expander("자세한 오류 보기"):
            st.exception(e)
        st.stop()

    # ----- [08.7] PROGRESS UPDATE ---------------------------------------
    pct = 30 + int((i / max(1, pending)) * 60)
    update_pct(pct, f"완료 • {fname}")

    # ----- [08.8] STOP CHECK (AFTER SAVE) --------------------------------
    if should_stop():
        update_msg("🛑 중지 요청 감지 — 현재 파일까지 저장 완료, 종료합니다.")
        break


    # ===== [09] FINALIZE & RETURN ============================================
    update_pct(95, "두뇌 저장 중")
    try:
        storage_context.persist(persist_dir=persist_dir)
    except Exception as e:
        st.error("인덱스 저장 중 오류가 발생했습니다.")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

    update_pct(100, "완료")
    return load_index_from_storage(storage_context)
