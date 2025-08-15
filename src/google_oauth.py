# src/google_oauth.py — OAuth (Drive.file) with st.query_params only
from __future__ import annotations

import json
import os
import time
import secrets
import urllib.parse
import requests
import streamlit as st

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# ---- 필요한 시크릿 키 ----
# st.secrets["GOOGLE_OAUTH_CLIENT_ID"]
# st.secrets["GOOGLE_OAUTH_CLIENT_SECRET"]
# st.secrets["GOOGLE_OAUTH_REDIRECT_URI"]  # e.g. https://your-app.streamlit.app/

SCOPES = [
    "openid",
    "email",
    "https://www.googleapis.com/auth/drive.file",  # 내 드라이브에 파일(폴더) 생성/업데이트
]

AUTH_BASE = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"


def _get_client_info():
    cid = (st.secrets.get("GOOGLE_OAUTH_CLIENT_ID") or "").strip()
    csec = (st.secrets.get("GOOGLE_OAUTH_CLIENT_SECRET") or "").strip()
    redirect = (st.secrets.get("GOOGLE_OAUTH_REDIRECT_URI") or "").strip()
    if not (cid and csec and redirect):
        raise RuntimeError("OAuth 클라이언트 시크릿이 없습니다. (CLIENT_ID/SECRET/REDIRECT_URI)")
    return cid, csec, redirect


# ------------------ 세션 상태 키 ------------------
_KEY_STATE = "_oauth_state"
_KEY_CREDS = "_oauth_creds"         # dict 저장
_KEY_EMAIL = "_oauth_email"
# ------------------------------------------------


def is_signed_in() -> bool:
    return isinstance(st.session_state.get(_KEY_CREDS), dict)


def get_user_email() -> str | None:
    return st.session_state.get(_KEY_EMAIL)


def start_oauth() -> str:
    """사용자를 구글 동의 화면으로 보내는 URL 생성."""
    client_id, _, redirect_uri = _get_client_info()
    state = secrets.token_urlsafe(24)
    st.session_state[_KEY_STATE] = state

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "state": state,
        "access_type": "offline",               # refresh_token 요청
        "include_granted_scopes": "true",
        "prompt": "consent",                    # 항상 동의창 -> refresh_token 확보 용이
    }
    return f"{AUTH_BASE}?{urllib.parse.urlencode(params)}"


def _save_creds(token_json: dict):
    # token_json: access_token, expires_in, refresh_token?, id_token ...
    now = int(time.time())
    token_json["expires_at"] = now + int(token_json.get("expires_in", 0))
    st.session_state[_KEY_CREDS] = token_json


def _build_credentials_from_session() -> Credentials:
    client_id, client_secret, _ = _get_client_info()
    data = st.session_state.get(_KEY_CREDS) or {}
    if not data:
        raise RuntimeError("세션에 OAuth 자격이 없습니다.")

    creds = Credentials(
        token=data.get("access_token"),
        refresh_token=data.get("refresh_token"),
        token_uri=TOKEN_URL,
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES,
    )

    # 만료 시점 반영(가능하면 자동 갱신)
    expires_at = data.get("expires_at")
    if expires_at:
        # google.oauth2.credentials는 expires_at 직접 받지 않음 → refresh로 갱신
        if time.time() >= float(expires_at) - 30:
            try:
                creds.refresh(Request())
                # 새 토큰을 다시 보존
                st.session_state[_KEY_CREDS]["access_token"] = creds.token
                st.session_state[_KEY_CREDS]["expires_at"] = int(time.time()) + 3600
            except Exception as e:
                # 실패 시 로그아웃 상태로 돌려서 재로그인 유도
                sign_out()
                raise

    return creds


def build_drive_service():
    creds = _build_credentials_from_session()
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def sign_out():
    for k in (_KEY_CREDS, _KEY_EMAIL, _KEY_STATE):
        if k in st.session_state:
            del st.session_state[k]


def finish_oauth_if_redirected():
    """
    리다이렉트로 돌아온 경우 쿼리파람(code/state)을 읽어 토큰 교환.
    st.experimental_* 호출 없이 st.query_params만 사용.
    """
    # st.query_params는 딕셔너리처럼 사용 가능
    qp = dict(st.query_params)
    code = qp.get("code")
    state = qp.get("state")
    if not code:
        return  # 리다이렉트 아님

    expected_state = st.session_state.get(_KEY_STATE)
    # state가 없거나 불일치해도(환경에 따라) 허용하되, 보안상 일치할 때가 바람직
    if expected_state and state and state != expected_state:
        st.warning("OAuth state 불일치: 새로고침 후 다시 로그인하세요.")
        return

    try:
        client_id, client_secret, redirect_uri = _get_client_info()
        data = {
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
        resp = requests.post(TOKEN_URL, data=data, timeout=30)
        resp.raise_for_status()
        token_json = resp.json()
        _save_creds(token_json)

        # 유저 이메일 조회(표시용)
        at = token_json.get("access_token")
        if at:
            ui = requests.get(USERINFO_URL, headers={"Authorization": f"Bearer {at}"}, timeout=15)
            if ui.ok:
                st.session_state[_KEY_EMAIL] = (ui.json().get("email") or "")

    except Exception as e:
        st.error(f"Google 로그인 처리 실패: {e}")
    finally:
        # URL 정리: code/state 제거
        try:
            st.query_params.clear()
        except Exception:
            pass
