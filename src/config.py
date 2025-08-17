# src/config.py
from __future__ import annotations
import os, json, base64
from pathlib import Path
from typing import Optional, Any
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

# --------------------- 하위호환 + 견고한 시크릿 로딩 --------------------------
def _st_secrets_get(key: str) -> Any:
    try:
        import streamlit as st  # type: ignore
        return st.secrets.get(key, None)
    except Exception:
        return None

def _first_nonempty(*keys: str) -> Any:
    # env -> secrets 순서로 조회
    for k in keys:
        v = os.environ.get(k, None)
        if v not in (None, "", "null"):
            return v
    for k in keys:
        v = _st_secrets_get(k)
        if v not in (None, "", "null"):
            return v
    return None

def _coerce_json_str(v: Any) -> str:
    """dict/bytes/base64/str 모두 JSON 문자열로 강제 변환."""
    if v is None:
        return ""
    # dict로 넣은 경우
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    # bytes -> str
    if isinstance(v, (bytes, bytearray)):
        try:
            v = v.decode("utf-8")
        except Exception:
            v = str(v)
    s = str(v).strip()
    if not s:
        return ""

    # base64 가능성
    # (토큰에 { 나 "type":"service_account" 가 없고, base64처럼 보이면 복호화 시도)
    looks_b64 = all(c.isalnum() or c in "+/=\n\r" for c in s) and ("{" not in s and "}" not in s)
    if looks_b64:
        try:
            dec = base64.b64decode(s, validate=True).decode("utf-8", errors="ignore").strip()
            if dec.startswith("{") and dec.endswith("}"):
                return dec
        except Exception:
            pass

    return s

# 서비스계정 JSON 여러 키 이름을 모두 지원
try:
    if not settings.GDRIVE_SERVICE_ACCOUNT_JSON:
        cand = _first_nonempty(
            # 권장
            "APP_GDRIVE_SERVICE_ACCOUNT_JSON",
            "APP_GDRIVE_SERVICE_ACCOUNT_JSON_B64",
            # 구키
            "GDRIVE_SERVICE_ACCOUNT_JSON",
            "GDRIVE_SERVICE_ACCOUNT_JSON_B64",
            # 과거/기타 호환 키들
            "GOOGLE_SERVICE_ACCOUNT_JSON",
            "SERVICE_ACCOUNT_JSON",
        )
        if cand:
            settings.GDRIVE_SERVICE_ACCOUNT_JSON = _coerce_json_str(cand)

    if not settings.BACKUP_FOLDER_ID:
        cand = _first_nonempty("APP_BACKUP_FOLDER_ID", "BACKUP_FOLDER_ID")
        if cand:
            settings.BACKUP_FOLDER_ID = str(cand)

    if not settings.ADMIN_PASSWORD:
        cand = _first_nonempty("APP_ADMIN_PASSWORD", "ADMIN_PASSWORD")
        if cand:
            settings.ADMIN_PASSWORD = str(cand)

    # UI 옵션도 덮어쓰기 허용
    bgp = _first_nonempty("APP_BG_IMAGE_PATH", "BG_IMAGE_PATH")
    if bgp:
        settings.BG_IMAGE_PATH = str(bgp)

    bgf = _first_nonempty("APP_USE_BG_IMAGE", "USE_BG_IMAGE")
    if bgf not in (None, ""):
        settings.USE_BG_IMAGE = str(bgf).strip().lower() in ("1", "true", "yes", "on")
except Exception:
    pass
# -- end compat
