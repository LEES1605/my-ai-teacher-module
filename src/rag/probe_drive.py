# ===== [01] PURPOSE ==========================================================
# Google Drive 연결이 되는지 "목록만" 확인하는 안전한 프로브.
# 기존 코드/기능에 영향이 없도록 독립 파일로 추가합니다.

# ===== [02] IMPORTS ==========================================================
from __future__ import annotations
from typing import Mapping, Dict, Any, List
import streamlit as st

# ===== [03] CREDS / SERVICE ==================================================
def _load_service_account(gcp_creds: Mapping[str, object] | None):
    """st.secrets 또는 인자에서 서비스계정 JSON을 읽어 Credentials 생성."""
    try:
        from google.oauth2.service_account import Credentials  # lazy import
    except Exception as e:
        raise RuntimeError("google-auth 패키지가 필요합니다.") from e

    # 1) 함수 인자 우선, 2) secrets 키 후보군
    info = dict(gcp_creds) if gcp_creds else None
    if info is None:
        raw = None
        for key in ("GDRIVE_SERVICE_ACCOUNT_JSON", "GOOGLE_SERVICE_ACCOUNT_JSON", "SERVICE_ACCOUNT_JSON"):
            if key in st.secrets and str(st.secrets[key]).strip():
                raw = st.secrets[key]
                break
        if raw is None:
            raise KeyError("st.secrets['GDRIVE_SERVICE_ACCOUNT_JSON']가 없습니다.")
        import json as _json
        info = _json.loads(raw) if isinstance(raw, str) else dict(raw)

    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    return Credentials.from_service_account_info(info, scopes=scopes)

def _resolve_folder_id(gdrive_folder_id: str | None) -> str:
    if gdrive_folder_id and str(gdrive_folder_id).strip():
        return str(gdrive_folder_id).strip()
    for key in ("GDRIVE_FOLDER_ID", "DRIVE_FOLDER_ID"):
        if key in st.secrets and str(st.secrets[key]).strip():
            return str(st.secrets[key]).strip()
    raise KeyError("대상 폴더 ID가 없습니다. st.secrets['GDRIVE_FOLDER_ID']를 설정하세요.")

def _drive_service(creds):
    try:
        from googleapiclient.discovery import build  # lazy import
    except Exception as e:
        raise RuntimeError("google-api-python-client 패키지가 필요합니다.") from e
    return build("drive", "v3", credentials=creds, cache_discovery=False)

# ===== [04] LIST FILES (PROBE) ===============================================
def probe_list(gdrive_folder_id: str | None = None, gcp_creds: Mapping[str, object] | None = None) -> Dict[str, Any]:
    """연결/권한/쿼터 등 기본 동작을 점검하기 위한 프로브."""
    creds = _load_service_account(gcp_creds)
    fid = _resolve_folder_id(gdrive_folder_id)
    service = _drive_service(creds)

    q = f"'{fid}' in parents and trashed=false"
    fields = "files(id,name,mimeType,modifiedTime), nextPageToken"
    files, token = [], None
    while True:
        resp = service.files().list(q=q, fields=fields, pageToken=token, pageSize=1000).execute()
        files.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    files.sort(key=lambda x: x.get("name", ""))

    sample = [{k: f.get(k) for k in ("id", "name", "mimeType", "modifiedTime")} for f in files[:10]]
    result = {"ok": True, "files_total": len(files), "sample": sample}
    return result

# ===== [05] STREAMLIT HOOK (optional) =======================================
def render_probe_panel():
    st.subheader("RAG: Drive Probe")
    if st.button("🔍 Test Drive Connection"):
        try:
            res = probe_list()
            st.success(f"OK. files_total={res['files_total']}")
            st.json(res["sample"])
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
