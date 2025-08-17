# src/rag/index_build.py
from __future__ import annotations
import os
from typing import Callable, Mapping, Dict, List, Any

import streamlit as st

from .storage import _make_storage_context
from .checkpoint import _load_checkpoint, _mark_done
from .quality import get_opt, preprocess_docs, maybe_summarize_docs, load_quality_report, save_quality_report

def build_index_with_checkpoint(
    update_pct: Callable[[int, str | None], None],
    update_msg: Callable[[str], None],
    gdrive_folder_id: str,
    gcp_creds: Mapping[str, Any],
    persist_dir: str,
    remote_manifest: Dict[str, Dict],
    should_stop: Callable[[], bool] | None = None,
):
    """
    파일ID 단위 처리 → 각 파일 완주 후 저장 & 체크포인트 기록.
    전처리/중복제거/문서요약 → SentenceSplitter 청킹 → 인덱스 누적.
    중지 버튼(should_stop=True) 감지 시 '현재 파일까지' 저장 후 안전 종료.
    """
    from llama_index.core import VectorStoreIndex, load_index_from_storage
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.readers.google import GoogleDriveReader

    if should_stop is None:
        should_stop = lambda: False

    opt = get_opt()

    update_pct(15, "Drive 리더 초기화")
    loader = GoogleDriveReader(service_account_key=gcp_creds)

    cp = _load_checkpoint()
    todo_ids: List[str] = []
    for fid, meta in remote_manifest.items():
        done = cp.get(fid)
        if done and done.get("md5") and meta.get("md5") and done["md5"] == meta["md5"]:
            continue
        todo_ids.append(fid)

    total = len(remote_manifest)
    pending = len(todo_ids)
    done_cnt = total - pending
    update_pct(30, f"문서 목록 불러오는 중 • 전체 {total}개, 이번에 처리 {pending}개")

    os.makedirs(persist_dir, exist_ok=True)
    storage_context = _make_storage_context(persist_dir)
    try:
        _ = load_index_from_storage(storage_context)
    except Exception:
        pass

    # 품질 리포트
    qrep = load_quality_report()
    qrep.setdefault("summary", {}).setdefault("total_docs", total)
    for k in ("processed_docs","kept_docs","skipped_low_text","skipped_dup","total_chars"):
        qrep["summary"].setdefault(k, 0)
    qrep.setdefault("files", {})
    seen_hashes = set(h for h in [])  # 초기화(파일별로 누적)

    if pending == 0:
        update_pct(95, "변경 없음 → 저장본 그대로 사용")
        return load_index_from_storage(storage_context)

    splitter = SentenceSplitter(chunk_size=opt["chunk_size"], chunk_overlap=opt["chunk_overlap"])

    for i, fid in enumerate(todo_ids, start=1):
        if should_stop():
            update_msg("🛑 중지 요청 감지 — 현재까지 저장 후 종료합니다.")
            break

        meta = remote_manifest.get(fid, {})
        fname = meta.get("name") or fid
        update_msg(f"전처리 • {fname} ({done_cnt + i}/{total})")

        # 1) 파일 로드
        try:
            docs = loader.load_data(file_ids=[fid])
        except TypeError:
            st.error("GoogleDriveReader 버전이 오래되어 file_ids 옵션을 지원하지 않습니다. requirements 업데이트 필요.")
            st.stop()

        # 2) 전처리/중복제거
        kept, stats = preprocess_docs(
            docs, seen_hashes,
            min_chars=opt["min_chars"], dedup=opt["dedup"]
        )
        maybe_summarize_docs(kept)

        # 3) 품질 리포트 갱신(파일 단위)
        qrep.setdefault("files", {})[fid] = {
            "name": fname,
            "md5": meta.get("md5"),
            "modifiedTime": meta.get("modifiedTime"),
            "kept": stats["kept"],
            "skipped_low_text": stats["skipped_low_text"],
            "skipped_dup": stats["skipped_dup"],
            "total_chars": stats["total_chars"],
        }
        qs = qrep.setdefault("summary", {})
        qs["processed_docs"] = qs.get("processed_docs", 0) + 1
        qs["kept_docs"] = qs.get("kept_docs", 0) + stats["kept"]
        qs["skipped_low_text"] = qs.get("skipped_low_text", 0) + stats["skipped_low_text"]
        qs["skipped_dup"] = qs.get("skipped_dup", 0) + stats["skipped_dup"]
        qs["total_chars"] = qs.get("total_chars", 0) + stats["total_chars"]
        save_quality_report(qrep)

        if stats["kept"] == 0:
            _mark_done(cp, fid, meta)  # 완료 체크만
            pct = 30 + int((i / max(1, pending)) * 60)
            update_pct(pct, f"건너뜀 • {fname} (저품질/중복)")
            if should_stop():
                update_msg("🛑 중지 요청 감지 — 현재까지 저장 후 종료합니다.")
                break
            continue

        # 4) 인덱스에 누적 추가
        update_msg(f"인덱스 생성 • {fname} ({done_cnt + i}/{total})")
        try:
            VectorStoreIndex.from_documents(
                kept, storage_context=storage_context, show_progress=False,
                transformations=[splitter]
            )
            storage_context.persist(persist_dir=persist_dir)  # 부분 저장
            _mark_done(cp, fid, meta)                         # 파일 완료 기록
        except Exception as e:
            st.error(f"인덱스 생성 중 오류: {fname}")
            with st.expander("자세한 오류 보기"):
                st.exception(e)
            st.stop()

        pct = 30 + int((i / max(1, pending)) * 60)
        update_pct(pct, f"완료 • {fname}")

        if should_stop():
            update_msg("🛑 중지 요청 감지 — 현재 파일까지 저장 완료, 종료합니다.")
            break

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
