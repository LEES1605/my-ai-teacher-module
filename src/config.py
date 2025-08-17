# src/config.py
from __future__ import annotations
import os
import tempfile
from pathlib import Path
import streamlit as st
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from typing import Any, Mapping
import json

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
    )

    # === 필수 ===
    GEMINI_API_KEY: SecretStr
    GDRIVE_FOLDER_ID: str

    # === 선택 ===
    ADMIN_PASSWORD: str | None = None
    # 서비스 계정 JSON은 dict(권장) 또는 JSON 문자열 모두 허용
    GDRIVE_SERVICE_ACCOUNT_JSON: Any | None = None

    # === 앱/LLM 설정 ===
    USE_BG_IMAGE: bool = False
    LLM_MODEL: str = "gemini-1.5-pro"
    EMBED_MODEL: str = "text-embedding-004"
    SIMILARITY_TOP_K: int = 5
    RESPONSE_MODE: str = "compact"

    # === UI/브랜딩 ===
    BRAND_COLOR: str = "#9067C6"
    TITLE_TEXT: str = "세상에서 가장 쉬운 이유문법"
    TITLE_SIZE_REM: float = 3.0
    LOGO_HEIGHT_PX: int = 110

# Streamlit Cloud에서는 st.secrets를 통해 주입되므로,
# st.secrets 값을 우선으로 하여 Settings 인스턴스를 구성한다.
def _build_settings_from_streamlit() -> Settings:
    data: dict[str, Any] = {}
    for k in ("GEMINI_API_KEY", "GDRIVE_FOLDER_ID", "ADMIN_PASSWORD", "USE_BG_IMAGE",
              "LLM_MODEL", "EMBED_MODEL", "SIMILARITY_TOP_K", "RESPONSE_MODE",
              "BRAND_COLOR", "TITLE_TEXT", "TITLE_SIZE_REM", "LOGO_HEIGHT_PX"):
        if k in st.secrets:
            data[k] = st.secrets[k]
    # 서비스 계정 JSON은 dict 또는 문자열(JSON) 모두 지원
    if "GDRIVE_SERVICE_ACCOUNT_JSON" in st.secrets:
        raw = st.secrets["GDRIVE_SERVICE_ACCOUNT_JSON"]
        if isinstance(raw, Mapping):
            data["GDRIVE_SERVICE_ACCOUNT_JSON"] = dict(raw)  # AttrDict 포함
        elif isinstance(raw, str) and raw.strip():
            try:
                data["GDRIVE_SERVICE_ACCOUNT_JSON"] = json.loads(raw)
            except Exception:
                data["GDRIVE_SERVICE_ACCOUNT_JSON"] = raw  # 나중에 파싱 시도
        else:
            data["GDRIVE_SERVICE_ACCOUNT_JSON"] = None
    return Settings(**data)

settings = _build_settings_from_streamlit()

# ===== 애플리케이션 데이터 경로 =====
APP_DATA_DIR: Path = Path(tempfile.gettempdir()) / "my_ai_teacher"
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

PERSIST_DIR: str = str(APP_DATA_DIR / "storage_gdrive")
MANIFEST_PATH: str = str(APP_DATA_DIR / "drive_manifest.json")

# 외부 접근 편의를 위해 settings에 경로 부착
settings.PERSIST_DIR = PERSIST_DIR  # type: ignore[attr-defined]
settings.MANIFEST_PATH = MANIFEST_PATH  # type: ignore[attr-defined]
