# src/rag/llm.py
from __future__ import annotations
import streamlit as st

def init_llama_settings(api_key: str, llm_model: str, embed_model: str, temperature: float = 0.0):
    """
    Gemini LLM/Embedding 초기화 (+ 간단한 헬스체크).
    실패 시 Streamlit UI에 에러를 띄우고 안전 종료합니다.
    """
    from llama_index.core import Settings
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    from llama_index.llms.google_genai import GoogleGenAI

    Settings.llm = GoogleGenAI(model=llm_model, api_key=api_key, temperature=temperature)
    Settings.embed_model = GoogleGenAIEmbedding(model_name=embed_model, api_key=api_key)

    try:
        _ = Settings.embed_model.get_text_embedding("ping")
    except Exception as e:
        st.error("임베딩 모델 점검 실패 — API 키/모델명/네트워크를 확인하세요.")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()
