# src/rag/checkpoint.py
from __future__ import annotations
import os, json
from typing import Dict
from src.config import APP_DATA_DIR

CHECKPOINT_PATH = str((APP_DATA_DIR / "index_checkpoint.json").resolve())

def _load_checkpoint(path: str = CHECKPOINT_PATH) -> Dict[str, Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_checkpoint(data: Dict[str, Dict], path: str = CHECKPOINT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)

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
