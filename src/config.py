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

def _get_jsonish(key: str) -> Any:
    """
    서비스계정 JSON처럼 dict 또는 JSON 문자열 둘 다 허용.
    그대로 반환(스트링이면 그대로, 매핑이면 그대로).
    """
    v = _get(key, "")
    # str이면 그대로, 매핑이면 그대로. (rag_engine에서 str/dict 모두 처리)
    return v

class _Settings:
    # --- 필수/주요 ---
    GEMINI_API_KEY: SecretStr
    GDRIVE_FOLDER_ID: str
    GDRIVE_SERVICE_ACCOUNT_JSON: Any  # str(JSON) 또는 dict

    # --- LLM/RAG 기본값(없으면 폴백) ---
    LLM_MODEL: str
    EMBED_MODEL: str
    RESPONSE_MODE: str
    SIMILARITY_TOP_K: int

    # --- 경로(위 상수를 settings에서도 접근하도록)---
    PERSIST_DIR: str = PERSIST_DIR
    MANIFEST_PATH: str = MANIFEST_PATH

    def __init__(self) -> None:
        # 필수 키(없으면 빈 문자열 → 상위 로직에서 에러 처리)
        self.GEMINI_API_KEY = SecretStr(_get("GEMINI_API_KEY", ""))
        self.GDRIVE_FOLDER_ID = str(_get("GDRIVE_FOLDER_ID", "")).strip()
        self.GDRIVE_SERVICE_ACCOUNT_JSON = _get_jsonish("GOOGLE_SERVICE_ACCOUNT_JSON")

        # 기본 모델/옵션 (필요시 secrets/환경변수로 덮어쓰기 가능)
        self.LLM_MODEL = str(_get("LLM_MODEL", "gemini-1.5-pro")).strip()
        # 임베딩 모델 명칭은 프로젝트에 맞게 조정(예: "text-embedding-004" 등)
        self.EMBED_MODEL = str(_get("EMBED_MODEL", "text-embedding-004")).strip()
        self.RESPONSE_MODE = str(_get("RESPONSE_MODE", "compact")).strip()
        try:
            self.SIMILARITY_TOP_K = int(_get("SIMILARITY_TOP_K", 5))
        except Exception:
            self.SIMILARITY_TOP_K = 5

# 외부에서 import할 settings 싱글턴
settings = _Settings()

# 외부에서 바로 참조하고 싶을 때를 위한 __all__
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
