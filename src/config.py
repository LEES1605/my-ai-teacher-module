# src/config.py
from __future__ import annotations
import tempfile
from pathlib import Path
import streamlit as st
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from typing import Any, Mapping
import json

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # === 필수 ===
    GEMINI_API_KEY: SecretStr
    GDRIVE_FOLDER_ID: str  # 학습 자료 폴더(=prepared)

    # === 선택 ===
    ADMIN_PASSWORD: str | None = None
    GDRIVE_SERVICE_ACCOUNT_JSON: Any | None = None  # dict 또는 문자열(JSON)

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

    # === 백업/복원 옵션 ===
    AUTO_BACKUP_TO_DRIVE: bool = True            # 인덱싱 완료 직후 ZIP 자동 업로드
    BACKUP_FOLDER_ID: str | None = None          # 지정 없으면 GDRIVE_FOLDER_ID 사용
    BACKUP_KEEP_N: int = 5                       # 보관 개수(최신 N개만 유지)

    # === 최적화(품질) 옵션 ===
    CHUNK_SIZE: int = 900                         # 문장기반 청킹 크기(문자)
    CHUNK_OVERLAP: int = 120                      # 청크 오버랩(문자)
    MIN_CHARS_PER_DOC: int = 500                  # 이보다 짧으면 저품질로 판단(스캔/이미지 가능성)
    DEDUP_BY_TEXT_HASH: bool = True               # 텍스트 완전중복 제거
    SKIP_LOW_TEXT_DOCS: bool = True               # 저품질(짧은/텍스트 빈약) 문서 건너뛰기
    PRE_SUMMARIZE_DOCS: bool = False              # 문서별 요약 메타데이터 생성(토큰/시간 증가)

def _coerce_sa_dict(val: Any | None) -> dict[str, Any] | None:
    """서비스계정 JSON을 dict로 보정: 문자열 파싱/중첩 키/개행 복원."""
    if val is None:
        return None
    d: dict[str, Any] | None = None

    if isinstance(val, Mapping):
        d = dict(val)
    elif isinstance(val, str) and val.strip():
        try:
            d = json.loads(val)
        except Exception:
            return None
    else:
        return None

    # {"service_account": {...}} 형태도 허용
    for k in ("service_account", "serviceAccount"):
        if isinstance(d.get(k), Mapping):
            d = dict(d[k])  # type: ignore[index]

    # private_key 개행 복원(\n → 실제 개행)
    pk = d.get("private_key")
    if isinstance(pk, str) and "\\n" in pk and "\n" not in pk:
        d["private_key"] = pk.replace("\\n", "\n")

    return d

def _build_settings_from_streamlit() -> Settings:
    """st.secrets를 우선으로 Settings 인스턴스를 구성."""
    data: dict[str, Any] = {}

    for k in (
        "GEMINI_API_KEY", "GDRIVE_FOLDER_ID", "ADMIN_PASSWORD", "USE_BG_IMAGE",
        "LLM_MODEL", "EMBED_MODEL", "SIMILARITY_TOP_K", "RESPONSE_MODE",
        "BRAND_COLOR", "TITLE_TEXT", "TITLE_SIZE_REM", "LOGO_HEIGHT_PX",
        "AUTO_BACKUP_TO_DRIVE", "BACKUP_FOLDER_ID", "BACKUP_KEEP_N",
        "CHUNK_SIZE", "CHUNK_OVERLAP", "MIN_CHARS_PER_DOC",
        "DEDUP_BY_TEXT_HASH", "SKIP_LOW_TEXT_DOCS", "PRE_SUMMARIZE_DOCS",
    ):
        if k in st.secrets:
            data[k] = st.secrets[k]

    # 서비스계정 JSON: 다양한 키 후보를 지원
    candidates = [
        "GDRIVE_SERVICE_ACCOUNT_JSON",
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "google_service_account_json",
        "SERVICE_ACCOUNT_JSON",
    ]
    sa_raw = None
    for c in candidates:
        if c in st.secrets:
            sa_raw = st.secrets[c]
            break

    data["GDRIVE_SERVICE_ACCOUNT_JSON"] = _coerce_sa_dict(sa_raw)
    return Settings(**data)

settings = _build_settings_from_streamlit()

# ===== 애플리케이션 데이터 경로(모듈 상수로 제공) =====
APP_DATA_DIR: Path = Path(tempfile.gettempdir()) / "my_ai_teacher"
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

PERSIST_DIR: str = str(APP_DATA_DIR / "storage_gdrive")
MANIFEST_PATH: str = str(APP_DATA_DIR / "drive_manifest.json")
QUALITY_REPORT_PATH: str = str(APP_DATA_DIR / "quality_report.json")
