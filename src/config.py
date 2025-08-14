# src/config.py
from pathlib import Path
import tempfile

# 앱 런타임 산출물(캐시/인덱스 등)을 소스폴더 밖 임시경로에 저장
APP_DATA_DIR = Path(tempfile.gettempdir()) / "my_ai_teacher"
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

PERSIST_DIR = str(APP_DATA_DIR / "storage_gdrive")
MANIFEST_PATH = str(APP_DATA_DIR / "drive_manifest.json")

# UI/브랜딩 상수
BRAND_COLOR = "#9067C6"
TITLE_TEXT = "세상에서 가장 쉬운 이유문법"
TITLE_SIZE_REM = 3.0
LOGO_HEIGHT_PX = 110
