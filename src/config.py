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

# ========== UI/브랜딩 상수 (필요시 settings에서 override 값도 제공) ==========
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

def _from_secrets(key: str) -> Any | None:
    if key in _SECRETS:
        v = _SECRETS[key]
        if isinstance(v, str):
            if v.strip():
                return v
        else:
            # dict/list 등 비어있지 않으면 사용
            try:
                if v is not None and str(v).strip():
                    return v
            except Exception:
                return v
    return None

def _get(key: str, default: Any = "") -> Any:
    """
    우선순위: st.secrets → 환경변수 → 기본값
    """
    v = _from_secrets(key)
    if v is not None:
        return v
    env = os.getenv(key)
    if env is not None and str(env).strip():
        return env
    return default

def _get_first(keys: list[str], default: Any = "") -> Any:
    """
    여러 키 중 먼저 발견되는 값을 반환 (secrets → env → default)
    """
    for k in keys:
        v = _get(k, None)
        if v is not None and str(v).strip():
            return v
    return default

def _get_jsonish_first(keys: list[str]) -> Any:
    """
    서비스계정 JSON처럼 dict 또는 JSON 문자열 둘 다 허용.
    그대로 반환(스트링이면 그대로, 매핑이면 그대로).
    """
    v = _get_first(keys, "")
    return v

class _Settings:
    # --- 필수/주요 ---
    GEMINI_API_KEY: SecretStr
    OPENAI_API_KEY: SecretStr             # ✅ ChatGPT 단계용
    GDRIVE_FOLDER_ID: str
    GDRIVE_SERVICE_ACCOUNT_JSON: Any      # str(JSON) 또는 dict

    # --- LLM/RAG 기본값(없으면 폴백) ---
    LLM_MODEL: str
    EMBED_MODEL: str
    RESPONSE_MODE: str
    SIMILARITY_TOP_K: int

    # --- 경로(위 상수를 settings에서도 접근하도록) ---
    PERSIST_DIR: str = PERSIST_DIR
    MANIFEST_PATH: str = MANIFEST_PATH

    # --- 브랜딩(상수의 런타임 override 값; 필요 시 사용) ---
    BRAND_COLOR: str = BRAND_COLOR
    TITLE_TEXT: str = TITLE_TEXT
    TITLE_SIZE_REM: float = TITLE_SIZE_REM
    LOGO_HEIGHT_PX: int = LOGO_HEIGHT_PX

    def __init__(self) -> None:
        # 필수 키(없으면 빈 문자열 → 상위 로직에서 에러/비활성 처리)
        self.GEMINI_API_KEY = SecretStr(_get("GEMINI_API_KEY", ""))
        self.OPENAI_API_KEY = SecretStr(_get_first(
            ["OPENAI_API_KEY", "OPENAI_APIKEY", "OPENAI_KEY"], ""
        ))

        # Google Drive
        self.GDRIVE_FOLDER_ID = str(_get_first(
            ["GDRIVE_FOLDER_ID", "GOOGLE_DRIVE_FOLDER_ID"], ""
        )).strip()
        
        # (선택) 채팅 로그를 따로 둘 폴더 (공유드라이브 내부)
        self.CHATLOG_FOLDER_ID = str(_get("CHATLOG_FOLDER_ID", "")).strip()
        
        # 서비스계정 JSON: 다양한 키 이름을 지원(호환성)
        self.GDRIVE_SERVICE_ACCOUNT_JSON = _get_jsonish_first([
            "GOOGLE_SERVICE_ACCOUNT_JSON",
            "google_service_account_json",
            "GDRIVE_SERVICE_ACCOUNT_JSON",
            "SERVICE_ACCOUNT_JSON",
        ])

        # 기본 모델/옵션 (필요시 secrets/환경변수로 덮어쓰기 가능)
        self.LLM_MODEL = str(_get("LLM_MODEL", "gemini-1.5-pro")).strip()
        # 임베딩 모델 명칭은 프로젝트에 맞게 조정(예: "text-embedding-004" 등)
        self.EMBED_MODEL = str(_get("EMBED_MODEL", "text-embedding-004")).strip()
        self.RESPONSE_MODE = str(_get("RESPONSE_MODE", "compact")).strip()
        try:
            self.SIMILARITY_TOP_K = int(_get("SIMILARITY_TOP_K", 5))
        except Exception:
            self.SIMILARITY_TOP_K = 5

        # 브랜딩 런타임 override (secrets/env에 있으면 사용)
        try:
            bc = str(_get("BRAND_COLOR", BRAND_COLOR)).strip()
            if bc: self.BRAND_COLOR = bc
        except Exception:
            pass
        try:
            tt = str(_get("TITLE_TEXT", TITLE_TEXT)).strip()
            if tt: self.TITLE_TEXT = tt
        except Exception:
            pass
        try:
            ts = _get("TITLE_SIZE_REM", TITLE_SIZE_REM)
            self.TITLE_SIZE_REM = float(ts)
        except Exception:
            pass
        try:
            lh = _get("LOGO_HEIGHT_PX", LOGO_HEIGHT_PX)
            self.LOGO_HEIGHT_PX = int(lh)
        except Exception:
            pass

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
