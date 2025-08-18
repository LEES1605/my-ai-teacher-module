# ===== [01] TOP OF FILE ======================================================
# Streamlit AI-Teacher
# - 사이드바 고정
# - 관리자: 응답 모드(자동/수동), RAG/LLM 설정
# - 우측 로그/오류 패널
# - ✅ 관리자: 글자 "크기/색깔" 테마 편집 UI + 실시간 CSS 적용

# ===== [02] ENV VARS =========================================================
import os, time, re, datetime as dt, traceback
import streamlit as st

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_RUN_ON_SAVE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ===== [03] IMPORTS ==========================================================
from src.config import settings, PERSIST_DIR
from src.ui import load_css, safe_render_header, ensure_progress_css, render_progress_bar
from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
from src.rag_engine import get_or_build_index, init_llama_settings, get_text_answer
from src.auth import admin_login_flow

# ===== [04] PAGE SETUP =======================================================
st.set_page_config(
    page_title="나의 AI 영어 교사",
    layout="wide",
    initial_sidebar_state="expanded"  # 사이드바 항상 표시
)
# 배경 이미지는 강제로 사용(없으면 CSS의 폴백 배경색 적용)
load_css("assets/style.css", use_bg=True, bg_path="assets/background_book.png")
# 혹시 테마/확장 CSS가 사이드바를 숨길 경우를 대비해 강제 노출
st.markdown("<style>[data-testid='stSidebar']{display:block!important;}</style>", unsafe_allow_html=True)
ensure_progress_css()
safe_render_header()

# ===== [05] LOG PANEL (공용) =================================================
def _log(msg: str):
    st.session_state.setdefault("_ui_logs", [])
    ts = dt.datetime.now().strftime("%H:%M:%S")
    st.session_state["_ui_logs"].append(f"[{ts}] {msg}")

def _log_exception(prefix: str, exc: Exception):
    _log(f"{prefix}: {exc}")
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    st.session_state["_ui_traceback"] = tb

# ===== [06] THEME: 관리자 ‘글자 크기/색’ 편집기 ===============================
# - 관리자 사이드바에서 값을 고르면 session_state 에 저장
# - 아래 _emit_dynamic_css() 가 매 프레임 CSS를 주입하여 실시간 반영
THEME_KEYS = {
    "title_size": 2.0,      # rem
    "subtitle_size": 1.15,  # rem
    "body_size": 1.00,      # rem
    "title_color": "#F7FAFC",
    "subtitle_color": "#E2E8F0",
    "body_color": "#F7FAFC",
    "user_bubble_bg": "#1F2A44",
    "assistant_bubble_bg": "#2D3748",
    "input_fg": "#FFFFFF",
    "input_bg": "rgba(255,255,255,0.12)",
    "input_border": "rgba(255,255,255,0.25)",
}

def _init_theme_defaults():
    for k, v in THEME_KEYS.items():
        st.session_state.setdefault(k, v)

def _emit_dynamic_css():
    """session_state 값으로 동적 CSS 주입 (관리자 테마 UI 실시간 반영)"""
    css = f"""
    <style>
      /* 제목/부제/본문 글자 크기 & 색상 */
      h1, .brand-title {{ font-size: {st.session_state['title_size']}rem !important;
                         color: {st.session_state['title_color']} !important; }}
      h2, h3 {{ color: {st.session_state['subtitle_color']} !important; }}
      body, .block-container, p, li, label, span, .stMarkdown p, .stMarkdown li {{
        font-size: {st.session_state['body_size']}rem !important;
        color: {st.session_state['body_color']} !important;
      }}

      /* 채팅 버블 배경 (사용자/어시스턴트) */
      [data-testid="stChatMessage"][data-testid="stChatMessage"] {{
        border-radius: 12px;
      }}
      /* 첫번째가 user, 두번째가 assistant가 오도록 streamlit이 렌더링하므로
         role 속성 대신 순서를 기준으로 배경을 지정 */
      .stChatMessage:nth-child(odd) {{ background: {st.session_state['user_bubble_bg']} !important; }}
      .stChatMessage:nth-child(even) {{ background: {st.session_state['assistant_bubble_bg']} !important; }}

      /* 입력창(텍스트/비밀번호) 대비 강화 */
      [data-testid="stTextInput"] input,
      input[type="text"], input[type="password"], textarea {{
        background-color: {st.session_state['input_bg']} !important;
        border: 1px solid {st.session_state['input_border']} !important;
        color: {st.session_state['input_fg']} !important;
        caret-color: {st.session_state['input_fg']} !important;
        border-radius: 10px !important;
      }}
      [data-testid="stTextInput"] input::placeholder, textarea::placeholder {{
        color: rgba(255,255,255,.6) !important;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ===== [07] ADMIN AUTH & SIDEBAR PANELS =====================================
# 상단 관리자 아이콘(가시성용)
_, _, _c3 = st.columns([0.8, 0.1, 0.1])
with _c3:
    if st.button("🛠️", key="admin_icon_top_bar"):
        st.session_state.admin_mode = True
        _log("관리자 버튼 클릭")

is_admin = admin_login_flow(settings.ADMIN_PASSWORD or "")

# 테마 기본값 초기화 + 즉시 CSS 주입(관리자 여부와 무관하게)
_init_theme_defaults()
_emit_dynamic_css()

with st.sidebar:
    st.markdown("## ⚙️ 관리자 패널")

    if is_admin:
        # --- [07-1] 응답 모드(자동/수동) --------------------------------------
        st.markdown("### 🧭 응답 모드")
        st.session_state.setdefault("use_manual_override", False)
        st.session_state["use_manual_override"] = st.checkbox(
            "수동 모드(관리자 오버라이드) 사용",
            value=st.session_state["use_manual_override"]
        )
        st.session_state.setdefault("manual_prompt_mode", "explainer")
        manual_mode = st.selectbox(
            "수동 모드 선택",
            ["explainer", "analyst", "reader"],
            index=["explainer","analyst","reader"].index(st.session_state["manual_prompt_mode"])
        )
        st.session_state["manual_prompt_mode"] = manual_mode

        # --- ✅ [07-2] 테마 편집기(글자 크기/색) ------------------------------
        with st.expander("🎨 테마/서체 편집 (글자 크기·색)", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.session_state["title_size"] = st.slider("제목 크기 (rem)", 1.2, 3.0, float(st.session_state["title_size"]), 0.05)
                st.session_state["subtitle_size"] = st.slider("부제 크기 (rem)", 0.9, 2.0, float(st.session_state["subtitle_size"]), 0.05)
                st.session_state["body_size"] = st.slider("본문 크기 (rem)", 0.9, 1.6, float(st.session_state["body_size"]), 0.05)
            with c2:
                st.session_state["title_color"] = st.color_picker("제목 색", st.session_state["title_color"])
                st.session_state["subtitle_color"] = st.color_picker("부제 색", st.session_state["subtitle_color"])
                st.session_state["body_color"] = st.color_picker("본문 색", st.session_state["body_color"])

            st.markdown("---")
            d1, d2 = st.columns(2)
            with d1:
                st.session_state["user_bubble_bg"] = st.color_picker("사용자 말풍선 배경", st.session_state["user_bubble_bg"])
                st.session_state["assistant_bubble_bg"] = st.color_picker("어시스턴트 말풍선 배경", st.session_state["assistant_bubble_bg"])
            with d2:
                st.session_state["input_fg"] = st.color_picker("입력창 글자색", st.session_state["input_fg"])
                st.session_state["input_bg"] = st.text_input("입력창 배경(css 값)", st.session_state["input_bg"])
                st.session_state["input_border"] = st.text_input("입력창 테두리(css 값)", st.session_state["input_border"])

            col_apply, col_reset = st.columns(2)
            with col_apply:
                if st.button("적용", key="apply_theme"):
                    _emit_dynamic_css()
                    st.success("테마가 적용되었습니다.")
            with col_reset:
                if st.button("기본값으로", key="reset_theme"):
                    for k, v in THEME_KEYS.items():
                        st.session_state[k] = v
                    _emit_dynamic_css()
                    st.success("기본 테마로 초기화했습니다.")

        # --- [07-3] RAG/LLM 설정 --------------------------------------------
        with st.expander("🤖 RAG/LLM 설정", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.session_state.setdefault("similarity_top_k", settings.SIMILARITY_TOP_K)
                st.session_state["similarity_top_k"] = st.slider("similarity_top_k", 1, 12, int(st.session_state["similarity_top_k"]))
            with col2:
                st.session_state.setdefault("temperature", 0.0)
                st.session_state["temperature"] = st.slider("LLM temperature", 0.0, 1.0, float(st.session_state["temperature"]), 0.05)
            with col3:
                st.session_state.setdefault("response_mode", settings.RESPONSE_MODE)
                st.session_state["response_mode"] = st.selectbox(
                    "response_mode",
                    ["compact","refine","tree_summarize"],
                    index=["compact","refine","tree_summarize"].index(st.session_state["response_mode"])
                )

        # --- [07-4] 관리자 도구 ----------------------------------------------
        with st.expander("🛠️ 관리자 도구", expanded=False):
            if st.button("↺ 두뇌 초기화(인덱스 삭제)"):
                import shutil
                try:
                    if os.path.exists(PERSIST_DIR):
                        shutil.rmtree(PERSIST_DIR)
                    st.session_state.pop("query_engine", None)
                    _log("두뇌 초기화 완료")
                    st.success("두뇌 파일을 삭제했습니다. 메인에서 다시 준비하세요.")
                except Exception as e:
                    _log_exception("두뇌 초기화 실패", e)
                    st.error("초기화 중 오류. 우측 로그/Traceback 확인.")

    else:
        st.info("우측 상단 '🛠️' 버튼으로 관리자 인증을 진행하세요.")

# ===== [08] LAYOUT: 좌측 본문 / 우측 로그 ====================================
left, right = st.columns([0.66, 0.34], gap="large")

with right:
    st.markdown("### 🔎 로그 / 오류 메시지")
    st.caption("오류나 진행 메시지를 여기에 자동 기록합니다. 복붙해서 공유하세요.")
    st.code("\n".join(st.session_state.get("_ui_logs", [])) or "로그 없음", language="text")
    st.markdown("**Traceback (있다면)**")
    st.code(st.session_state.get("_ui_traceback", "") or "(없음)", language="text")

# ===== [09] MAIN: 강의 준비 & 채팅 ===========================================
with left:
    # --- [09-1] 두뇌 준비 ----------------------------------------------------
    if "query_engine" not in st.session_state:
        st.markdown("## 📚 강의 준비")
        st.info("AI 두뇌가 아직 준비되지 않았습니다. 아래 버튼을 눌러 주세요.")

        if st.button("🧠 AI 두뇌 준비(복구/연결)"):
            try:
                bar_slot = st.empty(); msg_slot = st.empty()
                state_key = "_gp_pct"; st.session_state[state_key] = 0

                def update_pct(p, m=None):
                    st.session_state[state_key] = max(0, min(100, int(p)))
                    render_progress_bar(bar_slot, st.session_state[state_key])
                    if m:
                        msg_slot.markdown(f"<div class='gp-msg'>{m}</div>", unsafe_allow_html=True)
                        _log(m)

                update_pct(0, "두뇌 준비를 시작합니다…")

                init_llama_settings(
                    api_key=settings.GEMINI_API_KEY.get_secret_value(),
                    llm_model=settings.LLM_MODEL, embed_model=settings.EMBED_MODEL,
                    temperature=float(st.session_state.get("temperature", 0.0))
                )

                index = get_or_build_index(
                    update_pct=update_pct,
                    update_msg=lambda m: update_pct(st.session_state[state_key], m),
                    gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
                    raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
                    persist_dir=PERSIST_DIR,
                    manifest_path=settings.MANIFEST_PATH
                )

                st.session_state.query_engine = index.as_query_engine(
                    response_mode=st.session_state.get("response_mode", settings.RESPONSE_MODE),
                    similarity_top_k=int(st.session_state.get("similarity_top_k", settings.SIMILARITY_TOP_K))
                )
                update_pct(100, "두뇌 준비 완료!")
                time.sleep(0.4)
                st.rerun()

            except Exception as e:
                _log_exception("두뇌 준비 실패", e)
                st.error("두뇌 준비 중 오류. 우측 로그/Traceback 확인.")
                st.stop()

        # 학생 화면 보조 버튼
        if st.button("📥 강의 자료 다시 불러오기(두뇌 초기화)"):
            import shutil
            try:
                if os.path.exists(PERSIST_DIR):
                    shutil.rmtree(PERSIST_DIR)
                st.session_state.pop("query_engine", None)
                _log("본문에서 두뇌 초기화 실행")
                st.success("두뇌 파일을 삭제했습니다. 다시 ‘AI 두뇌 준비’를 눌러주세요.")
            except Exception as e:
                _log_exception("본문 초기화 실패", e)
                st.error("초기화 중 오류. 우측 로그/Traceback 확인.")
        st.stop()

    # --- [09-2] 채팅 UI ------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    st.markdown("---")

    # 학생용 라디오(표시는 유지, 실제 라우팅은 정책에 따름)
    mode_label = st.radio(
        "**어떤 도움이 필요한가요?**",
        ["💬 이유문법 설명", "🔎 구문 분석", "📚 독해 및 요약"],
        horizontal=True,
        key="mode_select"
    )

    prompt = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
    if not prompt:
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- [09-3] 응답 모드 결정(관리자 수동 > 학생 라디오) -------------------
    if is_admin and st.session_state.get("use_manual_override"):
        final_mode = st.session_state.get("manual_prompt_mode", "explainer")
        origin = "관리자 수동"
    else:
        final_mode = (
            "explainer" if mode_label.startswith("💬")
            else "analyst" if mode_label.startswith("🔎")
            else "reader"
        )
        origin = "학생 선택"

    if final_mode == "analyst":
        selected_prompt = ANALYST_PROMPT
    elif final_mode == "reader":
        selected_prompt = READER_PROMPT
    else:
        selected_prompt = EXPLAINER_PROMPT

    _log(f"모드 결정: {origin} → {final_mode}")

    try:
        with st.spinner("AI 선생님이 답변을 생각하고 있어요..."):
            answer = get_text_answer(st.session_state.query_engine, prompt, selected_prompt)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
    except Exception as e:
        _log_exception("답변 생성 실패", e)
        st.error("답변 생성 중 오류. 우측 로그/Traceback 확인.")

# ===== [10] END OF FILE ======================================================
