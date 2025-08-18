# ===== [01] TOP OF FILE ======================================================
# Streamlit AI-Teacher (sidebar 항상 표시 + 관리자 응답모드 수동/자동 + 오류로그 슬롯)

# ===== [02] ENV VARS =========================================================
import os, time, re
import streamlit as st

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_RUN_ON_SAVE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ===== [03] IMPORTS ==========================================================
from src.config import settings, APP_DATA_DIR, PERSIST_DIR
from src.ui import load_css, safe_render_header, ensure_progress_css, render_progress_bar
from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
from src.rag_engine import get_or_build_index, init_llama_settings, get_text_answer
from src.auth import admin_login_flow

# ===== [04] PAGE SETUP =======================================================
st.set_page_config(page_title="나의 AI 영어 교사",
                   layout="wide",
                   initial_sidebar_state="expanded")   # ← 사이드바 기본 확장

load_css("assets/style.css",
         use_bg=settings.USE_BG_IMAGE,
         bg_path="assets/background_book.png")
ensure_progress_css()
safe_render_header()

# ===== [05] ADMIN TOOLS ======================================================
# 상단 관리자 아이콘
_, _, c3 = st.columns([0.8, 0.1, 0.1])
with c3:
    if st.button("🛠️", key="admin_icon_top_bar"):
        st.session_state.admin_mode = True

# 관리자 인증
is_admin = admin_login_flow(settings.ADMIN_PASSWORD or "")

if is_admin:
    with st.sidebar:
        st.markdown("### 🧭 응답 모드 (자동/수동)")
        st.session_state.setdefault("use_manual_override", False)
        use_manual = st.checkbox("수동 모드 사용", value=st.session_state["use_manual_override"])
        st.session_state["use_manual_override"] = use_manual

        st.session_state.setdefault("manual_prompt_mode", "explainer")
        manual_mode = st.selectbox("수동 모드 선택",
                                   ["explainer", "analyst", "reader"],
                                   index=["explainer","analyst","reader"].index(
                                       st.session_state["manual_prompt_mode"]))
        st.session_state["manual_prompt_mode"] = manual_mode

    with st.sidebar.expander("⚙️ 고급 RAG/LLM 설정", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.setdefault("similarity_top_k", settings.SIMILARITY_TOP_K)
            k = st.slider("similarity_top_k", 1, 12, st.session_state["similarity_top_k"])
        with col2:
            st.session_state.setdefault("temperature", 0.0)
            temp = st.slider("LLM temperature", 0.0, 1.0,
                             float(st.session_state["temperature"]), 0.05)
        with col3:
            st.session_state.setdefault("response_mode", settings.RESPONSE_MODE)
            mode_sel = st.selectbox("response_mode",
                                    ["compact","refine","tree_summarize"],
                                    index=["compact","refine","tree_summarize"].index(
                                        st.session_state["response_mode"]))

        if st.button("적용", key="apply_rag"):
            st.session_state["similarity_top_k"] = k
            st.session_state["temperature"] = temp
            st.session_state["response_mode"] = mode_sel
            st.success("RAG/LLM 설정이 저장되었습니다.")

    with st.sidebar.expander("🛠️ 관리자 도구", expanded=False):
        if st.button("📥 강의 자료 다시 불러오기 (두뇌 초기화)"):
            import shutil
            if os.path.exists(PERSIST_DIR):
                shutil.rmtree(PERSIST_DIR)
            st.session_state.pop("query_engine", None)
            st.success("두뇌 파일이 초기화되었습니다.")

# ===== [06] ERROR/LOG SLOT ===================================================
st.session_state.setdefault("_error_log", [])
with st.sidebar.expander("🚨 오류 / 실패 로그", expanded=True):
    if not st.session_state["_error_log"]:
        st.write("아직 오류 없음.")
    else:
        for msg in st.session_state["_error_log"][-5:]:
            st.error(msg)

# ===== [07] MAIN WORKFLOW ====================================================
def main():
    # 두뇌 준비 안됐을 때
    if "query_engine" not in st.session_state:
        st.info("AI 교사를 시작하려면 아래 버튼을 누르세요.")

        if st.button("🧠 AI 두뇌 준비 시작하기"):
            bar_slot = st.empty(); msg_slot = st.empty()
            state_key = "_gp_pct"; st.session_state[state_key] = 0

            def update_pct(p, m=None):
                st.session_state[state_key] = max(0, min(100, int(p)))
                render_progress_bar(bar_slot, st.session_state[state_key])
                if m: msg_slot.markdown(m)

            def update_msg(m): msg_slot.markdown(m)

            update_msg("두뇌 준비를 시작합니다…")

            try:
                init_llama_settings(api_key=settings.GEMINI_API_KEY.get_secret_value(),
                                    llm_model=settings.LLM_MODEL,
                                    embed_model=settings.EMBED_MODEL,
                                    temperature=float(st.session_state.get("temperature", 0.0)))

                index = get_or_build_index(update_pct=update_pct,
                                           update_msg=update_msg,
                                           gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
                                           raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
                                           persist_dir=PERSIST_DIR,
                                           manifest_path=settings.MANIFEST_PATH)

                st.session_state.query_engine = index.as_query_engine(
                    response_mode=st.session_state.get("response_mode", settings.RESPONSE_MODE),
                    similarity_top_k=int(st.session_state.get("similarity_top_k", settings.SIMILARITY_TOP_K)))
                update_pct(100, "완료!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.session_state["_error_log"].append(repr(e))
                st.error(f"두뇌 준비 실패: {e}")
        st.stop()

    # === 채팅 UI ===
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    st.markdown("---")

    mode = st.radio("**어떤 도움이 필요한가요?**",
                    ["💬 이유문법 설명","🔎 구문 분석","📚 독해 및 요약"],
                    horizontal=True, key="mode_select")

    prompt = st.chat_input("질문을 입력하거나, 분석/요약할 문장을 붙여넣으세요.")
    if not prompt: return

    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # --- 응답 모드 결정 ---
    final_mode = None
    if is_admin and st.session_state.get("use_manual_override"):
        final_mode = st.session_state["manual_prompt_mode"]
    else:
        # 자동 모드 (질문 기반 간단 라우팅)
        if re.search(r"(분석|품사|구문)", prompt): final_mode = "analyst"
        elif re.search(r"(요약|정리|번역)", prompt): final_mode = "reader"
        else: final_mode = "explainer"

    if final_mode=="analyst": sel_prompt=ANALYST_PROMPT
    elif final_mode=="reader": sel_prompt=READER_PROMPT
    else: sel_prompt=EXPLAINER_PROMPT

    with st.spinner("AI 선생님이 답변을 생각하는 중..."):
        try:
            answer=get_text_answer(st.session_state.query_engine, prompt, sel_prompt)
        except Exception as e:
            st.session_state["_error_log"].append(repr(e))
            answer=f"⚠️ 오류 발생: {e}"

    st.session_state.messages.append({"role":"assistant","content":answer})
    st.rerun()

# ===== [08] MAIN ENTRY =======================================================
if __name__=="__main__":
    main()
