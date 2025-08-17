# src/config.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 프로젝트 디렉토리
ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DATA_DIR = Path(os.environ.get("APP_DATA_DIR", "/tmp/my_ai_teacher")).resolve()
PERSIST_DIR = (APP_DATA_DIR / "storage_gdrive").resolve()
REPORT_DIR = (APP_DATA_DIR / "reports").resolve()
REPORT_DIR.mkdir(parents=True, exist_ok=True)
QUALITY_REPORT_PATH = str((REPORT_DIR / "quality_report.json").resolve())
MANIFEST_PATH = str((APP_DATA_DIR / "drive_manifest.json").resolve())

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", case_sensitive=False)

    # ——— 모델/임베딩 ———
    GEMINI_API_KEY: str = Field(default="", description="Google Generative AI API Key")
    LLM_MODEL: str = "models/gemini-1.5-pro"
    EMBED_MODEL: str = "models/text-embedding-004"
    LLM_TEMPERATURE: float = 0.0

    # ——— RAG 품질 옵션 ———
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 80
    MIN_CHARS_PER_DOC: int = 80
    DEDUP_BY_TEXT_HASH: bool = True
    SKIP_LOW_TEXT_DOCS: bool = True
    PRE_SUMMARIZE_DOCS: bool = False

    # ——— 구글 드라이브 ———
    GDRIVE_FOLDER_ID: str = Field(default="prepared", description="강의자료 폴더 ID(또는 prepared)")
    GDRIVE_SERVICE_ACCOUNT_JSON: str = Field(default="", description="서비스계정 JSON 또는 secrets 키")

    # ——— 자동 백업/복원 ———
    AUTO_RESTORE_ON_START: bool = True         # 로컬 저장본이 없으면 백업 자동 복원
    AUTO_BACKUP_ON_SUCCESS: bool = True        # 빌드 성공 시 ZIP 자동 백업
    BACKUP_KEEP_N: int = 5                     # 백업 보관 개수

    # ——— 기타 ———
    RESPONSE_MODE: str = "compact"

# 외부에서 바로 쓰는 경로/설정
settings = Settings()
