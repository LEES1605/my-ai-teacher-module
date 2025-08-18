# ===== [01] TOP OF FILE ======================================================
# Streamlit AI-Teacher (slim shell using modular features)

# ===== [02] ENV VARS =========================================================
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_RUN_ON_SAVE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ===== [03] IMPORTS ==========================================================
import streamlit as st
from src.config import settings, PERSIST_DIR
from src.ui import load_css, render_header
from src.auth import admin_login_flow
from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
from src.rag_engine import (
    init_llama_settings, _load_index_from_disk, try_restore_index_from_drive,
    _normalize_sa, _validate_sa, get_text_answer
)
from src.features.stepper_ui import ensure_progress_css
from src.features.admin_panels import (
    render_admin_panels, render_admin_quickbar, render_student_waiting_view
)
from src.features.drive_card import get_effective_gdrive_folder_id
from src.patches.overrides import STATE_KEYS

# ===== [04] PAGE SETUP =======================================================
st.set_page_config(page_title="나의 AI 영어 교사", layout="wide", initial_sidebar_state="collapsed")
load_css("assets/style.css", use_bg=getattr(settings, "USE_BG_IMAGE", True),
         bg_path=getattr(settings, "BG_IMAGE_PATH", "assets/background_book.png"))
ensure_progress_css()
render_header("세상에서 가장 쉬운 이유문법","AI 교사와 함께하는 똑똑한 학습",logo_path="assets/academy_logo.png")

# 상단 우측 아이콘
_, _, _c3 = st.columns([0.8, 0.1, 0.1])
with _c3:
    if st.button("🛠️", key="admin_icon_top_bar"):
        st.session_state.admin_mode = True

# ===== [05] ADMIN AUTH =======================================================
is_admin = admin_login_flow(getattr(settings, "ADMIN_PASSWORD", ""))

# ===== [06] AUTO ATTACH/RESTORE (SILENT) =====================================
def _secret_or_str(v):
    try: return v.get_secret_value()
    except Exception: return str(v)

def _auto_attach_or_restore_silently() -> bool:
    try:
        if os.path.exists(PERSIST_DIR):
            init_llama_settings(
                api_key=_secret_or_str(settings.GEMINI_API_KEY),
                llm_model=settings.LLM_MODEL, embed_model=settings.EMBED_MODEL,
                temperature=float(st.session_state.get("temperature", 0.0)),
            )
            index = _load_index_from_disk(PERSIST_DIR)
            st.session_state.query_engine = index.as_query_engine(
                response_mode=st.session_state.get("response_mode", settings.RESPONSE_MODE),
                similarity_top_k=int(st.session_state.get("similarity_top_k", getattr(settings,"SIMILARITY_TOP_K",5))),
            )
            return True

        creds = _validate_sa(_normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON))
        dest = getattr(settings, "BACKUP_FOLDER_ID", None) or get_effective_gdrive_folder_id()
        if try_restore_index_from_drive(creds, PERSIST_DIR, dest):
            index = _load_index_from_disk(PERSIST_DIR)
            st.session_state.query_engine = index.as_query_engine(
                response_mode=st.session_state.get("response_mode", settings.RESPONSE_MODE),
                similarity_top_k=int(st.session_state.get("similarity_top_k", getattr(settings,"SIMILARITY_TOP_K",5))),
            )
            return True
    except Exception as e:
        st.session_state["_attach_error"] = repr(e)
    return False

if "query_engine" not in st.session_state:
    _auto_attach_or_restore_silently()

# ===== [07] CHAT UI ==========================================================
def render_chat_ui():
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    st.markdown("---")

    mode = st.radio("**어떤 도움이 필요한가요?**",
                    ["💬 이유문법 설명","🔎 구문 분석","📚 독해 및 요약"],
                    horizontal=True, key="mode_select")

    prompt = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.spinner("AI 선생님이 답변을 생각하고 있어요..."):
            selected = EXPLAINER_PROMPT if mode=="💬 이유문법 설명" else \
                       ANALYST_PROMPT  if mode=="🔎 구문 분석"  else READER_PROMPT
            answer = get_text_answer(st.session_state.query_engine, prompt, selected)
        st.session_state.messages.append({"role":"assistant","content":answer})
        st.rerun()

# ===== [08] MAIN =============================================================
def main():
    if "query_engine" in st.session_state and not st.session_state.get(STATE_KEYS.BUILD_PAUSED):
        if is_admin:
            render_admin_panels()
            render_admin_quickbar()
        render_chat_ui(); return

    if is_admin:
        render_admin_panels()
        render_admin_quickbar()
        return

    render_student_waiting_view()

if __name__ == "__main__":
    main()
