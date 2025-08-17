# src/google_oauth.py — OAuth 시작/마무리 + Drive 서비스 빌더
from __future__ import annotations
import time, json, secrets
import streamlit as st
import requests

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

AUTH_ENDPOINT  = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
USERINFO_EP    = "https://openidconnect.googleapis.com/v1/userinfo"
SCOPES = [
    "openid", "email", "profile",
    "https://www.googleapis.com/auth/drive.file",
]

def _client():
    cid = st.secrets["GOOGLE_OAUTH_CLIENT_ID"]
    csec = st.secrets["GOOGLE_OAUTH_CLIENT_SECRET"]
    redirect = st.secrets["GOOGLE_OAUTH_REDIRECT_URI"]
    return cid, csec, redirect

def is_signed_in() -> bool:
    tok = st.session_state.get("_oauth_tokens")
    return bool(tok and tok.get("access_token"))

def get_user_email() -> str | None:
    try:
        tok = st.session_state.get("_oauth_tokens", {})
        at = tok.get("access_token")
        if not at: return None
        r = requests.get(USERINFO_EP, headers={"Authorization": f"Bearer {at}"}, timeout=10)
        if r.ok:
            return r.json().get("email")
    except Exception:
        pass
    return None

def sign_out() -> None:
    st.session_state.pop("_oauth_tokens", None)

def start_oauth() -> str:
    client_id, _, redirect_uri = _client()
    state = secrets.token_urlsafe(16)
    st.session_state["_oauth_state"] = state
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
        "include_granted_scopes": "true",
    }
    # URL 구성
    from urllib.parse import urlencode
    return f"{AUTH_ENDPOINT}?{urlencode(params)}"

def finish_oauth_if_redirected() -> bool:
    """현재 URL에 code/state가 있으면 토큰교환을 수행. 처리했으면 True, 아니면 False."""
    try:
        try:
            qp = dict(st.query_params)  # Streamlit 1.48+
        except Exception:
            qp = st.experimental_get_query_params()
        code = qp.get("code")
        state = qp.get("state")
        if not code or not state:
            return False
        # state 검증(있을 때만)
        expected = st.session_state.get("_oauth_state")
        if expected and state != expected:
            # state 불일치 → 무시
            return False

        client_id, client_secret, redirect_uri = _client()
        data = {
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
        r = requests.post(TOKEN_ENDPOINT, data=data, timeout=20)
        r.raise_for_status()
        tokens = r.json()
        tokens["obtained_at"] = int(time.time())
        st.session_state["_oauth_tokens"] = tokens
        return True
    except Exception as e:
        # 페이지에서 토스트/경고는 app.py가 담당
        st.session_state["_oauth_error"] = str(e)
        return False

def build_drive_service():
    """세션 토큰으로 Drive v3 서비스 빌드."""
    tokens = st.session_state.get("_oauth_tokens")
    if not tokens:
        raise RuntimeError("로그인이 필요합니다.")
    client_id, client_secret, _ = _client()
    creds = Credentials(
        tokens.get("access_token"),
        refresh_token=tokens.get("refresh_token"),
        token_uri=TOKEN_ENDPOINT,
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES,
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)
