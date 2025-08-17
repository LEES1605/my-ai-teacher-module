# src/rag/storage.py
from __future__ import annotations
import os
import streamlit as st

def _has_persisted_index(persist_dir: str) -> bool:
    """LlamaIndex 저장본 핵심 파일이 하나라도 있으면 '있다'로 간주."""
    names = ("docstore.json", "index_store.json", "vector_store.json")
    return any(os.path.exists(os.path.join(persist_dir, n)) for n in names)

def _make_storage_context(persist_dir: str):
    """저장본이 있으면 로드, 없으면 빈 컨텍스트로 시작."""
    from llama_index.core import StorageContext
    try:
        if _has_persisted_index(persist_dir):
            return StorageContext.from_defaults(persist_dir=persist_dir)
    except Exception:
        pass
    return StorageContext.from_defaults()

@st.cache_resource(show_spinner=False)
def _load_index_from_disk(persist_dir: str):
    """디스크에서 인덱스를 로드(캐시)."""
    from llama_index.core import StorageContext, load_index_from_storage
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index
