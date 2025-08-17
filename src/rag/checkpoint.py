# src/rag/checkpoint.py
from __future__ import annotations
import os, json
from typing import Dict
from src.config import APP_DATA_DIR

CHECKPOINT_PATH = str((APP_DATA_DIR / "index_checkpoint.json").resolve())

def _load_checkpoint(path: str = CHECKPOINT_PATH, also_from_persist_dir: str | None = None) -> Dict[str, Dict]:
    """기본 경로에서 로드, 없으면 persist_dir/checkpoint.json도 시도 후 복사."""
    data: Dict[str, Dict] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass

    if also_from_persist_dir:
        alt = os.path.join(also_from_persist_dir, "checkpoint.json")
        try:
            with open(alt, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 로컬에도 복사해 둔다
            _save_checkpoint(data, path)
            return data
        except Exception:
            pass

    return {}

def _save_checkpoint(data: Dict[str, Dict], path: str = CHECKPOINT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)

def save_checkpoint_copy_to_persist(data: Dict[str, Dict], persist_dir: str) -> None:
    """persist_dir/checkpoint.json 에도 동기화 (ZIP 백업에 포함되도록)."""
    try:
        os.makedirs(persist_dir, exist_ok=True)
        with open(os.path.join(persist_dir, "checkpoint.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        pass

def _mark_done(cp: Dict[str, Dict], file_id: str, entry: Dict) -> None:
    cp[file_id] = {
        "md5": entry.get("md5"),
        "modifiedTime": entry.get("modifiedTime"),
        "name": entry.get("name"),
    }
    _save_checkpoint(cp)

def clear_checkpoint(path: str = CHECKPOINT_PATH) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
