# src/config.py — 앱 경로/브랜딩 상수 + Secrets 래핑(settings)

from __future__ import annotations
from pathlib import Path
import os, json
from typing import Any, Mapping

# ========= 앱 런타임 산출물 경로 (인덱스/캐시 등) =========
# 우선순위: 환경변수 AI_TEACHER_DATA_DIR → 프로젝트/.data/my_ai_teacher
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = Path(os.getenv("AI_TEACHER_DATA_DIR", _PROJECT_ROOT / ".data" / "my_ai_teacher"))
_DATA_ROOT.mkdir(parents=True, exist_ok=True)

APP_DATA_DIR = _DATA_ROOT
PERSIST_DIR = str(APP_DATA_DIR / "storage_gdrive")          # ← 인덱스 영구 저장
MANIFEST_PATH = str(APP_DATA_DIR / "drive_manifest.json")   # ← 지난번 매니페스트

# ========= UI/브랜딩 상수 =========
BRAND_COLOR = "#9067C6"
TITLE_TEXT = "세상에서 가장 쉬운 이유문법"
TITLE_SIZE_REM = 3.0
LOGO_HEIGHT_PX = 110

# ========= SecretStr 간단 래퍼 =========
class SecretStr:
    def __init__(self, v: Any | None):
        self._v = "" if v is None else str(v)
    def get_secret_value(self) -> str:
        return self._v
    def __str__(self) -> str:
        return "*****" if self._v else ""
    def __repr__(self) -> str:
        return "SecretStr(*****)"

# ========= settings: Streamlit Secrets / 환경변수 래핑 =========
try:
    import streamlit as st
    _SECRETS: Mapping[str, Any] = st.secrets if hasattr(st, "secrets") else {}
except Exception:
    _SECRETS = {}

def _get(key: str, default: Any = "") -> Any:
    """우선순위: st.secrets → 환경변수 → 기본값"""
    if key in _SECRETS and str(_SECRETS[key]).strip():
        return _SECRETS[key]
    env = os.getenv(key)
    if env is not None and str(env).strip():
        return env
    return default

def _get_jsonish(key: str) -> Any:
    """서비스계정 JSON: dict 또는 JSON 문자열 둘 다 허용."""
    v = _get(key, "")
    return v

class _Settings:
    # --- 필수/주요 ---
    GEMINI_API_KEY: SecretStr
    GDRIVE_FOLDER_ID: str                   # ✅ 인덱싱 대상: "prepared" 폴더 ID를 넣으세요!
    GDRIVE_SERVICE_ACCOUNT_JSON: Any        # str(JSON) 또는 dict

    # --- LLM/RAG 기본값 ---
    LLM_MODEL: str
    EMBED_MODEL: str
    RESPONSE_MODE: str
    SIMILARITY_TOP_K: int

    # --- 선택: OpenAI ---
    OPENAI_API_KEY: SecretStr | None
    OPENAI_LLM_MODEL: str | None
    OPENAI_EMBED_MODEL: str | None

    # --- 경로 ---
    PERSIST_DIR: str = PERSIST_DIR
    MANIFEST_PATH: str = MANIFEST_PATH

    def __init__(self) -> None:
        # 필수 키
        self.GEMINI_API_KEY = SecretStr(_get("GEMINI_API_KEY", ""))
        self.GDRIVE_FOLDER_ID = str(_get("GDRIVE_FOLDER_ID", "")).strip()
        self.GDRIVE_SERVICE_ACCOUNT_JSON = _get_jsonish("GOOGLE_SERVICE_ACCOUNT_JSON")

        # 기본 모델/옵션
        self.LLM_MODEL = str(_get("LLM_MODEL", "gemini-1.5-pro")).strip()
        self.EMBED_MODEL = str(_get("EMBED_MODEL", "text-embedding-004")).strip()
        self.RESPONSE_MODE = str(_get("RESPONSE_MODE", "compact")).strip()
        try:
            self.SIMILARITY_TOP_K = int(_get("SIMILARITY_TOP_K", 5))
        except Exception:
            self.SIMILARITY_TOP_K = 5

        # OpenAI (있으면 사용)
        oai_key = _get("OPENAI_API_KEY", "").strip()
        self.OPENAI_API_KEY = SecretStr(oai_key) if oai_key else None
        self.OPENAI_LLM_MODEL = str(_get("OPENAI_LLM_MODEL", "gpt-4o-mini")).strip()
        self.OPENAI_EMBED_MODEL = str(_get("OPENAI_EMBED_MODEL", "text-embedding-3-small")).strip()

# 외부에서 import할 settings 싱글턴
settings = _Settings()

__all__ = [
    "APP_DATA_DIR",
    "PERSIST_DIR",
    "MANIFEST_PATH",
    "BRAND_COLOR",
    "TITLE_TEXT",
    "TITLE_SIZE_REM",
    "LOGO_HEIGHT_PX",
    "SecretStr",
    "settings",
]
