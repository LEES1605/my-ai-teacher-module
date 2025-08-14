# src/rag_engine.py
from __future__ import annotations
import json
import streamlit as st

DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"

def _coerce_service_account(raw_sa):
    """st.secrets 서비스계정 JSON을 dict로 정규화."""
    if raw_sa is None:
        return None
    if isinstance(raw_sa, dict):
        return dict(raw_sa)
    if hasattr(raw_sa, "items"):
        return dict(raw_sa)
    if isinstance(raw_sa, str) and raw_sa.strip():
        return json.loads(raw_sa)
    return None

def smoke_test_drive():
    """
    베이스라인 점검:
    - 필수 secrets 존재 여부
    - 서비스계정 JSON 파싱
    - (가능하면) google-auth 로 Credentials 생성
    """
    s = st.secrets
    folder_id = s.get("GDRIVE_FOLDER_ID", "")
    raw_sa = s.get("GDRIVE_SERVICE_ACCOUNT_JSON", None)

    if not folder_id:
        return False, "❌ GDRIVE_FOLDER_ID 가 비어 있습니다."

    sa_dict = _coerce_service_account(raw_sa)
    if not sa_dict:
        return False, "❌ GDRIVE_SERVICE_ACCOUNT_JSON 이 비어있거나 JSON 파싱 실패."

    try:
        from google.oauth2 import service_account  # lazy import
        _ = service_account.Credentials.from_service_account_info(
            sa_dict, scopes=[DRIVE_READONLY_SCOPE]
        )
        return True, "✅ 서비스 계정 키 형식 OK (google-auth 로 자격증명 생성 성공)"
    except ModuleNotFoundError:
        return False, "⚠️ google-auth 미설치: requirements.txt 에 google-auth==2.40.3 추가 필요"
    except Exception as e:
        return False, f"❌ 자격증명 생성 실패: {e}"

def preview_drive_files(max_items: int = 10):
    """
    Drive 폴더 내 최근 파일 미리보기 (최신 N개).
    반환: (ok: bool, msg: str, rows: list[dict])
    """
    s = st.secrets
    folder_id = s.get("GDRIVE_FOLDER_ID", "")
    raw_sa = s.get("GDRIVE_SERVICE_ACCOUNT_JSON", None)

    if not folder_id:
        return False, "❌ GDRIVE_FOLDER_ID 가 비어 있습니다.", []

    sa_dict = _coerce_service_account(raw_sa)
    if not sa_dict:
        return False, "❌ GDRIVE_SERVICE_ACCOUNT_JSON 이 비어있거나 JSON 파싱 실패.", []

    # 필요한 라이브러리를 여기서만 임포트(지연 임포트)
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ModuleNotFoundError:
        return (
            False,
            "⚠️ google-api-python-client / google-auth-httplib2 / httplib2 가 필요합니다. "
            "requirements.txt 를 업데이트하세요.",
            [],
        )

    try:
        creds = service_account.Credentials.from_service_account_info(
            sa_dict, scopes=[DRIVE_READONLY_SCOPE]
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)

        q = f"'{folder_id}' in parents and trashed=false"
        fields = "files(id,name,mimeType,modifiedTime,size)"
        resp = service.files().list(
            q=q,
            pageSize=max_items,
            orderBy="modifiedTime desc",
            fields=fields,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        files = resp.get("files", [])
        rows = [{
            "name": f.get("name"),
            "mimeType": f.get("mimeType"),
            "modifiedTime": f.get("modifiedTime"),
            "size": f.get("size"),
            "id": f.get("id"),
        } for f in files]

        return True, "", rows

    except Exception as e:
        return False, f"❌ Drive API 호출 실패: {e}", []
