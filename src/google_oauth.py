# src/google_oauth.py — OAuth: 로그인/토큰교환/Drive 서비스 빌더
from __future__ import annotations
import json, time, requests
from typing import Optional
import streamlit as st
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

SCOPES = [
    "openid",
    "email",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/drive.file",   # 업로드 권한
]

def _secrets(key: str) -> str:
    v = st.secrets.get(key, "") if hasattr(st, "secrets") else ""
    return str(v).strip()

def start_oauth() -> str:
    client_id = _secrets("GOOGLE_OAUTH_CLIENT_ID")
    redirect_uri = _secrets("GOOGLE_OAUTH_REDIRECT_URI")
    state = str(int(time.time()))
    st.session_state["_oauth_state"] = state
    scope = " ".join(SCOPES)
    url = (
        "https://accounts.google.com/o/oauth2/v2/auth?"
        f"response_type=code&client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&scope={requests.utils.quote(scope)}"
        f"&access_type=offline&prompt=consent&state={state}"
    )
    return url

def finish_oauth_if_redirected():
    qp = dict(st.query_params)
    if "code" not in qp or "state" not in qp:
        return
    code = qp["code"]
    state = qp["state"]
    if st.session_state.get("_oauth_state") and st.session_state["_oauth_state"] != state:
        st.warning("OAuth state 불일치. 다시 시도하세요.")
        return
    # 토큰 교환
    token_url = "https://oauth2.googleapis.com/token"
    payload = {
        "code": code,
        "client_id": _secrets("GOOGLE_OAUTH_CLIENT_ID"),
        "client_secret": _secrets("GOOGLE_OAUTH_CLIENT_SECRET"),
        "redirect_uri": _secrets("GOOGLE_OAUTH_REDIRECT_URI"),
        "grant_type": "authorization_code",
    }
    r = requests.post(token_url, data=payload, timeout=30)
    r.raise_for_status()
    tokens = r.json()
    st.session_state["_oauth_tokens"] = tokens
    # 주소창 깨끗이
    st.query_params.clear()

def is_signed_in() -> bool:
    return bool(st.session_state.get("_oauth_tokens"))

def build_drive_service():
    tokens = st.session_state.get("_oauth_tokens") or {}
    creds = Credentials(
        token=tokens.get("access_token"),
        refresh_token=tokens.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=_secrets("GOOGLE_OAUTH_CLIENT_ID"),
        client_secret=_secrets("GOOGLE_OAUTH_CLIENT_SECRET"),
        scopes=SCOPES,
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def get_user_email() -> Optional[str]:
    tokens = st.session_state.get("_oauth_tokens") or {}
    if not tokens.get("access_token"):
        return None
    r = requests.get("https://www.googleapis.com/oauth2/v3/userinfo",
                     headers={"Authorization": f"Bearer {tokens['access_token']}"},
                     timeout=15)
    if r.ok:
        return r.json().get("email")
    return None

def sign_out():
    st.session_state.pop("_oauth_tokens", None)
    st.session_state.pop("_oauth_state", None)
