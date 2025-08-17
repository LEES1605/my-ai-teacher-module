# src/rag/engine.py
from __future__ import annotations
import os, json
from datetime import datetime
from typing import Callable, Any

import streamlit as st
from src.config import settings
from .drive import (
    _normalize_sa, _validate_sa, fetch_drive_manifest,
    try_restore_index_from_drive, export_brain_to_drive, prune_old_backups,
    INDEX_BACKUP_PREFIX,
)
from .checkpoint import CHECKPOINT_PATH
from .index_build import build_index_with_checkpoint

def get_or_build_index(
    update_pct: Callable[[int, str | None], None],
    update_msg: Callable[[str], None],
    gdrive_folder_id: str,
    raw_sa: Any | None,
    persist_dir: str,
    manifest_path: str,
    should_stop: Callable[[], bool] | None = None,
):
    """
    Drive 변경을 감지해 저장본을 쓰거나, 변경 시에만 재인덱싱(체크포인트 & 중지 지원).
    + (옵션) 로컬 저장본 없으면 드라이브 백업에서 자동 복원
    + (옵션) 빌드 성공 시 자동 백업 업로드 + 보관 N개 유지
    """
    update_pct(5, "드라이브 변경 확인 중…")
    gcp_creds = _validate_sa(_normalize_sa(raw_sa))

    # 0) 로컬 저장본이 없고 자동 복원이 켜져 있으면 시도
    if settings.AUTO_RESTORE_ON_START and not os.path.exists(persist_dir):
        update_msg("🗂️ 로컬 저장본 없음 → 드라이브 백업 자동 복원 시도")
        try:
            restored = try_restore_index_from_drive(gcp_creds, persist_dir, gdrive_folder_id)
            if restored:
                update_msg("✅ 드라이브 백업 복원 완료")
                update_pct(20, None)
        except Exception:
            # 실패해도 계속 진행 (처음부터 빌드)
            pass

    # 1) 매니페스트 비교
    remote = fetch_drive_manifest(gcp_creds, gdrive_folder_id)

    local = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fp:
                local = json.load(fp)
        except Exception:
            local = None

    def _manifests_differ(local_m: dict | None, remote_m: dict) -> bool:
        if local_m is None:
            return True
        if set(local_m.keys()) != set(remote_m.keys()):
            return True
        for fid, r in remote_m.items():
            l = local_m.get(fid, {})
            if l.get("md5") and r.get("md5"):
                if l["md5"] != r["md5"]:
                    return True
            if l.get("modifiedTime") != r.get("modifiedTime"):
                return True
        return False

    # 변경 없음 → 저장본 바로 로드
    if os.path.exists(persist_dir) and not _manifests_differ(local, remote):
        update_pct(25, "변경 없음 → 저장된 두뇌 로딩")
        from llama_index.core import StorageContext, load_index_from_storage
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        idx = load_index_from_storage(storage_context)
        update_pct(100, "완료!")
        return idx

    # 변경 있음 → 체크포인트 이어서 빌드
    update_pct(40, "변경 감지 → 전처리/청킹/인덱스 생성 (체크포인트)")
    idx = build_index_with_checkpoint(
        update_pct, update_msg, gdrive_folder_id, gcp_creds, persist_dir, remote,
        should_stop=should_stop
    )

    # '완주' 상태면 새 매니페스트 저장 + 체크포인트 정리 + (옵션) 자동 백업
    try:
        if not (should_stop and should_stop()):
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as fp:
                json.dump(remote, fp, ensure_ascii=False, indent=2, sort_keys=True)
            # 완료 여부 간단 체크: 체크포인트의 키셋과 remote 키셋이 같으면 완료
            if os.path.exists(CHECKPOINT_PATH):
                with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                    cp = json.load(f)
                if set(cp.keys()) == set(remote.keys()):
                    try:
                        os.remove(CHECKPOINT_PATH)
                    except Exception:
                        pass

            # ✅ 자동 백업(옵션)
            if settings.AUTO_BACKUP_ON_SUCCESS:
                update_msg("☁️ 백업 업로드 중…")
                fname = f"{INDEX_BACKUP_PREFIX}-{datetime.now():%Y%m%d_%H%M%S}.zip"
                try:
                    _, name = export_brain_to_drive(gcp_creds, persist_dir, gdrive_folder_id, filename=fname)
                    update_msg(f"✅ 백업 업로드 완료: {name}")
                except Exception as e:
                    update_msg(f"⚠️ 백업 업로드 실패: {e}")

                # 보관 개수 유지
                try:
                    deleted = prune_old_backups(gcp_creds, gdrive_folder_id, keep=settings.BACKUP_KEEP_N)
                    if deleted:
                        update_msg(f"🧹 오래된 백업 정리 {len(deleted)}건")
                except Exception:
                    pass

    except Exception:
        pass

    update_pct(100, "완료!")
    return idx

def get_text_answer(query_engine, question: str, system_prompt: str) -> str:
    """질의응답(출처 파일명 표시)."""
    try:
        full_query = (
            f"{system_prompt}\n\n"
            "[지시사항] 반드시 업로드된 강의 자료를 최우선으로 참고하여 답변하고, "
            "근거를 찾을 수 없다면 그 사실을 명확히 밝혀라.\n\n"
            f"[학생의 질문]\n{question}"
        )
        response = query_engine.query(full_query)
        answer_text = str(response)
        try:
            files = [n.metadata.get("file_name", "알 수 없음") for n in getattr(response, "source_nodes", [])]
            source_files = ", ".join(sorted(list(set(files)))) if files else "출처 정보 없음"
        except Exception:
            source_files = "출처 정보 없음"
        return f"{answer_text}\n\n---\n*참고 자료: {source_files}*"
    except Exception as e:
        return f"텍스트 답변 생성 중 오류 발생: {e}"
