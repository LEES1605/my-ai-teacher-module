# src/config.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 프로젝트 루트
ROOT_DIR = Path(__file__).resolve().parent.parent

# 앱 데이터/저장 경로
APP_DATA_DIR = Path(os.environ.get("APP_DATA_DIR", "/tmp/my_ai_teacher")).resolve()
PERSIST_DIR = (APP_DATA_DIR / "storage_gdrive").resolve()
REPORT_DIR = (APP_DATA_DIR / "reports").resolve()
REPORT_DIR.mkdir(parents=True, exist_ok=True)

QUALITY_REPORT_PATH = str((REPORT_DIR / "quality_report.json").resolve())
MANIFEST_PATH = str((APP_DATA_DIR / "drive_manifest.json").resolve())

class Settings(BaseSettings):
    """
    환경변수/Secrets(.env)로 오버라이드 가능한 앱 설정.
    기본 접두사는 APP_  (예: APP_GDRIVE_SERVICE_ACCOUNT_JSON)
    """
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", case_sensitive=False)

    # ----- UI -----
    USE_BG_IMAGE: bool = True
    BG_IMAGE_PATH: str = "assets/background_book.png"

    # ----- 인증/관리 -----
    ADMIN_PASSWORD: Optional[str] = None  # 관리자 비밀번호(옵션)

    # ----- 모델/임베딩 -----
    GEMINI_API_KEY: str = Field(default="", description="Google Generative AI API Key")
    LLM_MODEL: str = "models/gemini-1.5-pro"
    EMBED_MODEL: str = "models/text-embedding-004"
    LLM_TEMPERATURE: float = 0.0
    SIMILARITY_TOP_K: int = 5

    # ----- RAG 품질 옵션 -----
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 80
    MIN_CHARS_PER_DOC: int = 80
    DEDUP_BY_TEXT_HASH: bool = True
    SKIP_LOW_TEXT_DOCS: bool = True
    PRE_SUMMARIZE_DOCS: bool = False

    # ----- 구글 드라이브 -----
    GDRIVE_FOLDER_ID: str = Field(default="prepared", description="강의자료 폴더 ID(혹은 prepared)")
    GDRIVE_SERVICE_ACCOUNT_JSON: str = Field(default="", description="서비스계정 JSON(문자열)")

    # 백업 폴더(선택: 지정 시 이곳 우선 사용)
    BACKUP_FOLDER_ID: Optional[str] = None

    # ----- 자동 백업/복원 -----
    AUTO_RESTORE_ON_START: bool = True       # 로컬 저장본 없으면 백업 자동 복원
    AUTO_BACKUP_ON_SUCCESS: bool = True      # 빌드 성공 시 ZIP 자동 백업
    BACKUP_KEEP_N: int = 5                   # 백업 보관 개수

    # ----- 기타 -----
    RESPONSE_MODE: str = "compact"

# 인스턴스 생성
settings = Settings()

# ---------------------------------------------------------------------------
# 하위호환: Streamlit Secrets의 기존 키도 자동으로 인식해서 보정
#  - APP_ 접두사 값이 비어 있으면, 동일 키의 비접두사 버전을 찾아 채워줌
#  - bool/str 변환도 안전하게 처리
# ---------------------------------------------------------------------------
def _secrets_get(key: str):
    try:
        import streamlit as st  # type: ignore
        return st.secrets.get(key, None)
    except Exception:
        return None

def _first_nonempty(*keys: str):
    for k in keys:
        v = _secrets_get(k)
        if v not in (None, "", "null"):
            return v
    return None

try:
    # 서비스계정 JSON
    if not settings.GDRIVE_SERVICE_ACCOUNT_JSON:
        v = _first_nonempty("APP_GDRIVE_SERVICE_ACCOUNT_JSON", "GDRIVE_SERVICE_ACCOUNT_JSON")
        if v:
            settings.GDRIVE_SERVICE_ACCOUNT_JSON = str(v)

    # 백업 폴더 ID
    if not settings.BACKUP_FOLDER_ID:
        v = _first_nonempty("APP_BACKUP_FOLDER_ID", "BACKUP_FOLDER_ID")
        if v:
            settings.BACKUP_FOLDER_ID = str(v)

    # 관리자 비번
    if not settings.ADMIN_PASSWORD:
        v = _first_nonempty("APP_ADMIN_PASSWORD", "ADMIN_PASSWORD")
        if v:
            settings.ADMIN_PASSWORD = str(v)

    # 배경 이미지 경로/ON/OFF
    v = _first_nonempty("APP_BG_IMAGE_PATH", "BG_IMAGE_PATH")
    if v:
        settings.BG_IMAGE_PATH = str(v)

    v = _first_nonempty("APP_USE_BG_IMAGE", "USE_BG_IMAGE")
    if v not in (None, ""):
        s = str(v).strip().lower()
        settings.USE_BG_IMAGE = s in ("1", "true", "yes", "on")
except Exception:
    # secrets 미사용 환경 등에서는 조용히 패스
    pass
