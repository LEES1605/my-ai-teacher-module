# src/rag/__init__.py
# 모듈화된 RAG 컴포넌트의 퍼블릭 API를 한 곳으로 모아 re-export 합니다.

from .llm import init_llama_settings
from .storage import _load_index_from_disk
from .drive import (
    _normalize_sa, _validate_sa,
    export_brain_to_drive, import_brain_from_drive,
    try_restore_index_from_drive, prune_old_backups,
    fetch_drive_manifest, INDEX_BACKUP_PREFIX,
)
from .checkpoint import (
    CHECKPOINT_PATH, clear_checkpoint, _load_checkpoint, _save_checkpoint, _mark_done
)
from .engine import get_or_build_index, get_text_answer

__all__ = [
    # LLM
    "init_llama_settings",
    # Storage
    "_load_index_from_disk",
    # Drive helpers
    "_normalize_sa", "_validate_sa",
    "export_brain_to_drive", "import_brain_from_drive",
    "try_restore_index_from_drive", "prune_old_backups",
    "fetch_drive_manifest", "INDEX_BACKUP_PREFIX",
    # Checkpoint
    "CHECKPOINT_PATH", "clear_checkpoint", "_load_checkpoint", "_save_checkpoint", "_mark_done",
    # Engine
    "get_or_build_index", "get_text_answer",
]
