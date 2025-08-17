# src/rag_engine.py
# 얇은 호환 레이어: app.py가 import하는 이름을 그대로 유지하면서
# 실제 구현은 src/rag/* 모듈에서 가져옵니다.

from .rag import (
    # LLM
    init_llama_settings,
    # Storage
    _load_index_from_disk,
    # Drive helpers
    _normalize_sa, _validate_sa,
    export_brain_to_drive, import_brain_from_drive,
    try_restore_index_from_drive, prune_old_backups,
    INDEX_BACKUP_PREFIX,
    # Checkpoint
    CHECKPOINT_PATH,
    # Engine
    get_or_build_index, get_text_answer,
)

# re-export 끝
