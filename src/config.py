# src/config.py — 앱 경로/브랜딩 상수 + Secrets 래핑(settings)

from __future__ import annotations
from pathlib import Path
import os, json, tempfile
from typing import Any, Mapping

# ========== 앱 런타임 산출물 경로 (인덱스/캐시 등) ==========
APP_DATA_DIR = Path(tempfile.gettempdir()) / "my_ai_teacher"
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

PERSIST_DIR = str(APP_DATA_DIR / "storage_gdrive")
MANIFEST_PATH = str(APP_DATA_DIR / "drive_manifest.json")

# ========== UI/브랜딩 상수 ==========
BRAND_COLOR = "#9067C6"
TITLE_TEXT = "세상에서 가장 쉬운 이유문법"
TITLE_SIZE_REM = 3.0
LOGO_HEIGHT_PX = 110

# ========== SecretStr 간단 래퍼 (pydantic 없이 get_secret_value 제공) ==========
class SecretStr:
    def __init__(self, v: Any | None):
        self._v = "" if v is None else str(v)
    def get_secret_value(self) -> str:
        return self._v
    def __str__(self) -> str:
        return "*****" if self._v else ""
    def __repr__(self) -> str:
        return "SecretStr(*****)"

# ========== settings: Streamlit Secrets / 환경변수 래핑 ==========
try:
    import streamlit as st
    _SECRETS: Mapping[str, Any] = st.secrets if hasattr(st, "secrets") else {}
except Exception:
    _SECRETS = {}

def _get(key: str, default: Any = "") -> Any:
    """
    우선순위: st.secrets → 환경변수 → 기본값
    """
    if key in _SECRETS and str(_SECRETS[key]).strip():
        return _SECRETS[key]
    env = os.getenv(key)
    if env is not None and str(env).strip():
        return env
    return default

def _get_
