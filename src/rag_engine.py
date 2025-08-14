# src/rag_engine.py — RAG 엔진 유틸 + Google Drive 테스트 함수(완전본)

from __future__ import annotations
import os, json, shutil, re
from typing import Callable, Any, Mapping

import streamlit as st
from src.config import settings

# ============================================================================
# 1) LLM/임베딩 설정 (LlamaIndex Settings)
# ============================================================================

def init_llama_settings(api_key: str, llm_model: str, embed_model: str, temperature: float = 0.0):
    """
    LlamaIndex 전역 Settings에 Google GenAI LLM/임베딩을 설정하고
    임베딩 1회 호출로 키/네트워크를 간단 점검합니다.
    """
    from llama_index.core import Settings
    # 신식 Google GenAI 경로 (requirements: llama-index-llms-google-genai / -embeddings-google-genai)
    from llama_index.llms.google_genai import GoogleGenAI
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

    Settings.llm = GoogleGenAI(model=llm_model, api_key=api_key, temperature=temperature)
    Settings.embed_model = GoogleGenAIEmbedding(model_name=embed_model, api_key=api_key)

    # 임베딩 스모크 테스트 (키/네트워크/모델명 확인)
    try:
        _ = Settings.embed_model.get_text_embedding("ping")
    except Exception as e:
        st.error("임베딩 모델 점검 실패 — API 키/모델명/네트워크를 확인하세요.")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()


# ============================================================================
# 2) Google Drive 연결 테스트/미리보기 유틸
# ============================================================================

def _build_drive_service(creds_dict):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    # cache_discovery=False: Cloud 환경에서 캐시 충돌 방지
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def _normalize_sa(raw_sa: Any | None) -> Mapping[str, Any] | None:
    """
    서비스계정 JSON을 dict로 정규화.
    - dict면 그대로
    - str이면 json.loads 시도
    - private_key가 실제 개행으로 들어간 잘못된 문자열을 자동 보정(개행 → \\n 이스케이프)
    """
    if raw_sa is None:
        return None
    if isinstance(raw_sa, Mapping):
        return dict(raw_sa)

    if isinstance(raw_sa, str):
        s = raw_sa.strip()
        if not s:
            return None
        # 1차 시도
        try:
            return json.loads(s)
        except Exception:
            pass
        # private_key 보정
        try:
            m = re.search(r'"private_key"\s*:\s*"(?P<key>.*?)"', s, re.DOTALL)
            if m:
                key = m.group("key")
                key_fixed = key.replace("\r\n", "\n").replace("\n", "\\n")
                s_fixed = s[:m.start("key")] + key_fixed + s[m.end("key"):]
                return json.loads(s_fixed)
        except Exception:
            pass
    return None

def smoke_test_drive() -> tuple[bool, str]:
    """
    Drive 연결 스모크 테스트.
    (ok, msg) 형태로 반환하여 UI에서 바로 표시 가능.
    """
    folder_id = settings.GDRIVE_FOLDER_ID
    if not str(folder_id).strip():
        return (False, "GDRIVE_FOLDER_ID가 비었습니다. Secrets에 값을 추가하세요.")

    sa = _normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON)
    if not sa:
        return (False, "GOOGLE_SERVICE_ACCOUNT_JSON(서비스계정) 파싱 실패")

    try:
        svc = _build_drive_service(sa)
        # 간단 확인: 폴더 ID 조회
        svc.files().get(fileId=folder_id, fields="id").execute()
        return (True, "Google Drive 연결 OK")
    except Exception as e:
        return (False, f"Drive 연결 점검 실패: {e}")

def preview_drive_files(max_items: int = 10) -> tuple[bool, str, list[dict]]:
    """
    폴더 내 최신 파일 n개를 표로 보여주기 위한 헬퍼.
    rows 예시: [{"name":..., "link":..., "mime":..., "modified":...}, ...]
    """
    folder_id = settings.GDRIVE_FOLDER_ID
    sa = _normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON)
    if not sa or not str(folder_id).strip():
        return (False, "서비스계정/폴더 ID 설정이 부족합니다.", [])

    try:
        svc = _build_drive_service(sa)
        q = f"'{folder_id}' in parents and trashed=false"
        fields = "files(id,name,mimeType,modifiedTime), nextPageToken"
        resp = svc.files().list(
            q=q,
            orderBy="modifiedTime desc",
            pageSize=max_items,
            fields=fields,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files = resp.get("files", [])
        rows = []
        for f in files:
            rows.append({
                "name": f.get("name"),
                "link": f"https://drive.google.com/file/d/{f.get('id')}/view",
                "mime": f.get("mimeType"),
                "modified": f.get("modifiedTime"),
            })
        return (True, f"{len(rows)}개 파일", rows)
    except Exception as e:
        return (False, f"목록 조회 실패: {e}", [])


# ============================================================================
# 3) 인덱스 로딩/빌드 & 변경 감지(매니페스트)
# ============================================================================

@st.cache_resource(show_spinner=False)
def _load_index_from_disk(persist_dir: str):
    """디스크에 저장된 인덱스를 읽어옵니다."""
    from llama_index.core import StorageContext, load_index_from_storage
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index

def _fetch_drive_manifest(creds_dict, folder_id: str) -> dict:
    """드라이브 폴더 내 파일들의 '스냅샷(매니페스트)'를 가져옵니다."""
    svc = _build_drive_service(creds_dict)
    files = []
    page_token = None
    q = f"'{folder_id}' in parents and trashed=false"
    fields = "nextPageToken, files(id,name,mimeType,modifiedTime,md5Checksum,size)"
    while True:
        resp = svc.files().list(
            q=q, fields=fields, pageToken=page_token,
            pageSize=1000, supportsAllDrives=True, includeItemsFromAllDrives=True
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return {
        f["id"]: {
            "name": f.get("name"),
            "mimeType": f.get("mimeType"),
            "modifiedTime": f.get("modifiedTime"),
            "md5": f.get("md5Checksum"),
            "size": f.get("size"),
        } for f in files
    }

def _load_local_manifest(path: str) -> dict | None:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            return None
    return None

def _save_local_manifest(path: str, m: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(m, fp, ensure_ascii=False, indent=2, sort_keys=True)

def _manifests_differ(local: dict | None, remote: dict) -> bool:
    """md5 있으면 md5, 없으면 modifiedTime 비교."""
    if local is None:
        return True
    if set(local.keys()) != set(remote.keys()):
        return True
    for fid, r in remote.items():
        l = local.get(fid, {})
        if l.get("md5") and r.get("md5"):
            if l["md5"] != r["md5"]:
                return True
        if l.get("modifiedTime") != r.get("modifiedTime"):
            return True
    return False

def _build_index_with_progress(update_pct: Callable[[int, str | None], None],
                               update_msg: Callable[[str], None],
                               gdrive_folder_id: str,
                               gcp_creds: Mapping[str, Any],
                               persist_dir: str):
    """Drive → 문서 로드 → 인덱스 생성 → 디스크 저장"""
    from llama_index.core import VectorStoreIndex
    from llama_index.readers.google import GoogleDriveReader

    update_pct(5, "Google Drive 인증 준비")
    if not gcp_creds:
        st.error
