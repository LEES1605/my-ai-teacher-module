# src/google_oauth.py
from __future__ import annotations
import json, time
import streamlit as st
from typing import Tuple, Dict, Any

from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# 동의화면에서 켠 스코프와 일치
SCOPES = ["openid", "email", "https://www.googleapis.com/auth/drive.file"]

def _client_config() -> Dict[str, Any]:
    return {
        "web": {
            "client_id": st.secrets["GOOGLE_OAUTH_CLIENT_ID"],
            "client_secret": st.secrets["GOOGLE_OAUTH_CLIENT_SECRET"],
            "redirect_uris": [st.secrets["OAUTH_REDIRECT_URI"]],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }

def start_oauth() -> str:
    flow = Flow.from_client_config(_client_config(), scopes=SCOPES)
    flow.redirect_uri = st.secrets["OAUTH_REDIRECT_URI"]
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes=True,
        prompt="consent",
    )
    st.session_state["oauth_state"] = state
    return auth_url

def finish_oauth_if_redirected() -> Tuple[bool, str]:
    """리디렉션으로 돌아왔을 때 code/state 처리"""
    qs = st.experimental_get_query_params()
    if "code" not in qs or "state" not in qs:
        return False, ""
    code = qs["code"][0]
    state = qs["state"][0]
    if state != st.session_state.get("oauth_state"):
        return False, "state mismatch"

    flow = Flow.from_client_config(_client_config(), scopes=SCOPES)
    flow.redirect_uri = st.secrets["OAUTH_REDIRECT_URI"]
    flow.fetch_token(code=code)
    creds = flow.credentials
    st.session_state["oauth_creds"] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or []),
        "expiry": getattr(creds, "expiry", None),
    }
    return True, "ok"

def is_signed_in() -> bool:
    return "oauth_creds" in st.session_state

def build_drive_service():
    data = st.session_state.get("oauth_creds")
    if not data: return None
    creds = Credentials(
        token=data.get("token"),
        refresh_token=data.get("refresh_token"),
        token_uri=data.get("token_uri"),
        client_id=data.get("client_id"),
        client_secret=data.get("client_secret"),
        scopes=data.get("scopes"),
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        st.session_state["oauth_creds"]["token"] = creds.token
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def get_user_email() -> str:
    svc = build_drive_service()
    if not svc: return ""
    info = svc.about().get(fields="user(emailAddress)").execute()
    return info.get("user", {}).get("emailAddress", "")

def sign_out():
    st.session_state.pop("oauth_creds", None)
    st.session_state.pop("oauth_state", None)
