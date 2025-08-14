# src/rag_engine.py
from __future__ import annotations
import json
import streamlit as st

DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"

def _coerce_service_account(raw_sa):
    """st.secrets 에서 온 SA를 dict로 정규화."""
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

    # 여기서만 google-auth 임포트 → 미설치여도 앱 전체가 죽지 않음
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
