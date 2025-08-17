# app.py
# ===== TOP OF FILE ============================================================
# 0) Streamlit 페이지 설정은 반드시 최상단에서!
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_RUN_ON_SAVE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

import time
import streamlit as st

from src.config import settings, APP_DATA_DIR, PERSIST_DIR, MANIFEST_PATH
from src.ui import load_css, safe_render_header, ensure_progress_css, render_progress_bar, render_stepper
from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
from src.rag_engine import (
    get_or_build_index,
    init_llama_settings,
    get_text_answer,
    _load_index_from_disk,   # ← 저장된 두뇌를 즉시 불러오기 위한 내부 헬퍼
)
from src.auth import admin_login_flow

# 2) 페이지 메타
st.set_page_config(page_title="나의 AI 영어 교사", layout="wide", initial_sidebar_state="collapsed")

# 3) 전역 CSS + (선택) 배경
load_css("assets/style.css", use_bg=settings.USE_BG_IMAGE, bg_path="assets/background_book.png")
ensure_progress_css()

# 4) 헤더
safe_render_header()

# 5) 상단 우측 관리자 버튼
_, _, c3 = st.columns([0.8, 0.1, 0.1])
with c3:
    if st.button("🛠️", key="admin_icon_top_bar"):
        st.session_state.admin_mode = True

# 6) 고급/RAG 튜닝 패널(관리자만 보이게)
if admin_login_flow(settings.ADMIN_PASSWORD or ""):
    with st.expander("⚙️ 고급 RAG/LLM 설정", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.setdefault("similarity_top_k", settings.SIMILARITY_TOP_K)
            k = st.slider("similarity_top_k", 1, 12, st.session_state["similarity_top_k"])
        with col2:
            st.session_state.setdefault("temperature", 0.0)
            temp = st.slider("LLM temperature", 0.0, 1.0, float(st.session_state["temperature"]), 0.05)
        with col3:
            st.session_state.setdefault("response_mode", settings.RESPONSE_MODE)
            mode_sel = st.selectbox(
                "response_mode",
                ["compact", "refine", "tree_summarize"],
                index=["compact", "refine", "tree_summarize"].index(st.session_state["response_mode"]),
            )
        if st.button("적용"):
            st.session_state["similarity_top_k"] = k
            st.session_state["temperature"] = temp
            st.session_state["response_mode"] = mode_sel
            st.success("RAG/LLM 설정이 저장되었습니다. (인덱스/엔진은 다음 쿼리부터 반영)")

    with st.expander("🛠️ 관리자 도구", expanded=False):
        if st.button("📥 강의 자료 다시 불러오기 (두뇌 초기화)"):
            import shutil
            if os.path.exists(PERSIST_DIR):
                shutil.rmtree(PERSIST_DIR)
            if "query_engine" in st.session_state:
                del st.session_state["query_engine"]
            st.success("두뇌 파일이 초기화되었습니다. 아래 버튼으로 다시 준비하세요.")

# 7) 메인 워크플로우
def _auto_attach_saved_brain() -> bool:
    """
    이미 저장된 두뇌가 있으면 자동으로 연결.
    성공 시 True 반환(채팅 UI로 진입), 실패/없음이면 False.
    """
    # 저장된 인덱스가 없으면 패스
    if not os.path.exists(PERSIST_DIR):
        return False
    try:
        with st.spinner("🔄 이전에 준비된 두뇌를 불러오는 중..."):
            # LLM/임베딩 엔진은 쿼리 엔진 생성 전에 준비해야 함
            init_llama_settings(
                api_key=settings.GEMINI_API_KEY.get_secret_value(),
                llm_model=settings.LLM_MODEL,
                embed_model=settings.EMBED_MODEL,
                temperature=float(st.session_state.get("temperature", 0.0)),
            )
            index = _load_index_from_disk(PERSIST_DIR)  # 캐시 리소스라 빠름
            st.session_state.query_engine = index.as_query_engine(
                response_mode=st.session_state.get("response_mode", settings.RESPONSE_MODE),
                similarity_top_k=int(st.session_state.get("similarity_top_k", settings.SIMILARITY_TOP_K)),
            )
        st.success("이전에 준비된 두뇌에 자동 연결했습니다.")
        st.rerun()
        return True
    except Exception:
        # 자동 연결 실패해도 사용자에게 버튼을 보여주고 수동 준비로 진행
        st.warning("저장된 두뇌 자동 연결에 실패했습니다. 아래 버튼으로 다시 준비해 주세요.")
        return False

def main():
    # ➊ 세션에 엔진이 없고, 저장본이 있으면 자동 연결 시도
    if "query_engine" not in st.session_state:
        if _auto_attach_saved_brain():
            st.stop()

    # ➋ 여전히 엔진이 없으면 버튼 표시 → 수동 준비
    if "query_engine" not in st.session_state:
        st.info("AI 교사를 시작하려면 아래 버튼을 눌러주세요. 처음에는 학습량에 따라 시간이 소요될 수 있습니다.")

        if st.button("🧠 AI 두뇌 준비 시작하기"):
            # ── 진행 표시용 슬롯들 ───────────────────────────────────────────────
            stepper_slot = st.empty()
            bar_slot = st.empty()
            msg_slot = st.empty()

            # ── 스텝 정의 & 상태 초기화 ─────────────────────────────────────────
            steps = [
                ("check", "드라이브 변경 확인"),
                ("init", "Drive 리더 초기화"),
                ("list", "문서 목록 불러오는 중"),
                ("index", "인덱스 생성"),
                ("save", "두뇌 저장"),
            ]
            st.session_state["_step_status"] = {k: "pending" for k, _ in steps}
            st.session_state["_step_curr"] = None

            def _set_active(key: str):
                prev = st.session_state.get("_step_curr")
                if prev and st.session_state["_step_status"].get(prev) == "active":
                    st.session_state["_step_status"][prev] = "done"
                st.session_state["_step_status"][key] = "active"
                st.session_state["_step_curr"] = key
                render_stepper(stepper_slot, steps, st.session_state["_step_status"], sticky=True)

            def _set_done_all():
                for k, _ in steps:
                    st.session_state["_step_status"][k] = "done"
                render_stepper(stepper_slot, steps, st.session_state["_step_status"], sticky=True)

            # 최초 표시
            _set_active("check")
            render_progress_bar(bar_slot, 0)
            msg_slot.markdown("<div class='gp-msg'>두뇌 준비를 시작합니다…</div>", unsafe_allow_html=True)

            # update 콜백: 진행바 + 메시지 + 스텝퍼 동기화
            st.session_state["_gp_pct"] = 0
            def update_pct(pct: int, msg: str | None = None):
                st.session_state["_gp_pct"] = max(0, min(100, int(pct)))
                render_progress_bar(bar_slot, st.session_state["_gp_pct"])
                if msg is not None:
                    update_msg(msg)

            def update_msg(msg: str):
                text = msg or ""
                if "변경 확인" in text:
                    _set_active("check")
                elif "리더 초기화" in text:
                    _set_active("init")
                elif "목록 불러오는 중" in text:
                    _set_active("list")
                elif "인덱스 생성" in text:
                    _set_active("index")
                elif "저장 중" in text:
                    _set_active("save")
                elif "완료" in text:
                    _set_done_all()
                msg_slot.markdown(f"<div class='gp-msg'>{text}</div>", unsafe_allow_html=True)

            # 1) LLM/Embedding 설정(온디맨드)
            init_llama_settings(
                api_key=settings.GEMINI_API_KEY.get_secret_value(),
                llm_model=settings.LLM_MODEL,
                embed_model=settings.EMBED_MODEL,
                temperature=float(st.session_state.get("temperature", 0.0)),
            )

            # 2) 인덱스 준비(변경 감지 → 필요 시 재빌드)
            index = get_or_build_index(
                update_pct=update_pct,
                update_msg=update_msg,
                gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
                raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
                persist_dir=PERSIST_DIR,
                manifest_path=MANIFEST_PATH,
            )

            # 3) 질의 엔진 생성
            st.session_state.query_engine = index.as_query_engine(
                response_mode=st.session_state.get("response_mode", settings.RESPONSE_MODE),
                similarity_top_k=int(st.session_state.get("similarity_top_k", settings.SIMILARITY_TOP_K)),
            )
            update_pct(100, "완료!")
            time.sleep(0.5)
            stepper_slot.empty(); bar_slot.empty(); msg_slot.empty()
            st.rerun()

        st.stop()

    # === 채팅 UI ===
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.markdown("---")

    mode = st.radio(
        "**어떤 도움이 필요한가요?**",
        ["💬 이유문법 설명", "🔎 구문 분석", "📚 독해 및 요약"],
        horizontal=True,
        key="mode_select",
    )

    prompt = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("AI 선생님이 답변을 생각하고 있어요..."):
            if mode == "💬 이유문법 설명":
                selected_prompt = EXPLAINER_PROMPT
            elif mode == "🔎 구문 분석":
                selected_prompt = ANALYST_PROMPT
            else:
                selected_prompt = READER_PROMPT

            answer = get_text_answer(st.session_state.query_engine, prompt, selected_prompt)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

if __name__ == "__main__":
    main()
# ===== END OF FILE ============================================================
