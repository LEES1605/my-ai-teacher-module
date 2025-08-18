# ===== [01] TOP OF FILE ======================================================
# Streamlit AI-Teacher (slim shell using modular features) + 자동/수동 응답 선택

# ===== [02] ENV VARS =========================================================
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_RUN_ON_SAVE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ===== [03] IMPORTS ==========================================================
import re
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

# 상단 우측 아이콘(관리자 모드 진입)
_, _, _c3 = st.columns([0.8, 0.1, 0.1])
with _c3:
    if st.button("🛠️", key="admin_icon_top_bar"):
        st.session_state.admin_mode = True

# ===== [05] ADMIN AUTH =======================================================
is_admin = admin_login_flow(getattr(settings, "ADMIN_PASSWORD", ""))

# ✅ 관리자 인증되면, 전역 CSS가 숨긴 사이드바를 다시 보이도록 강제 오버라이드
#    (assets/style.css 에 [data-testid="stSidebar"] { display: none; } 가 있으므로 필요)
if is_admin:
    st.markdown(
        "<style>[data-testid='stSidebar']{display:block !important;}</style>",
        unsafe_allow_html=True
    )

# ===== [06] AUTO ATTACH/RESTORE (SILENT) =====================================
def _secret_or_str(v):
    try: return v.get_secret_value()
    except Exception: return str(v)

def _auto_attach_or_restore_silently() -> bool:
    """
    - 로컬 PERSIST_DIR이 있으면 그대로 attach
    - 없으면 Drive에서 복구 시도
    - 둘 다 실패 시 False
    """
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

# ===== [06.5] ADMIN: 응답 선택(자동/수동) UI ==================================
def _render_admin_response_selector():
    """
    관리자만 보는 '응답 선택(자동/수동)' 패널.
    - 기본: 질문 기반 자동 라우팅
    - 필요 시: 수동 오버라이드(관리자 지정)
    """
    with st.sidebar:
        st.markdown("### 🧭 응답 선택(자동/수동)")

        # 오버라이드 토글(관리자만)
        st.session_state.setdefault("use_manual_override", False)
        use_manual = st.checkbox("수동 모드(관리자 오버라이드) 사용", value=st.session_state["use_manual_override"])
        st.session_state["use_manual_override"] = use_manual

        # 수동 모드일 때, 어떤 페르소나(프롬프트)를 강제할지 선택
        st.session_state.setdefault("manual_prompt_mode", "explainer")
        manual_mode = st.selectbox(
            "수동 모드 선택",
            ["explainer(이유문법 설명)", "analyst(구문 분석)", "reader(독해/요약)"],
            index=["explainer(이유문법 설명)","analyst(구문 분석)","reader(독해/요약)"].index(
                {"explainer":"explainer(이유문법 설명)",
                 "analyst":"analyst(구문 분석)",
                 "reader":"reader(독해/요약)"}[
                    st.session_state.get("manual_prompt_mode","explainer")
                ]
            ),
            help="체크박스를 켜면 이 선택이 학생 라디오를 덮어씁니다."
        )
        # 내부 저장값은 짧은 키워드로
        if "explainer" in manual_mode: st.session_state["manual_prompt_mode"] = "explainer"
        elif "analyst" in manual_mode: st.session_state["manual_prompt_mode"] = "analyst"
        else: st.session_state["manual_prompt_mode"] = "reader"

if is_admin:
    _render_admin_response_selector()

# ===== [07] 간단한 자동 라우팅(질문 → 프롬프트 모드) ==========================
_EXPLAIN_HINTS = re.compile(r"(왜|이유|설명|원리|규칙|뜻이|무슨 의미|어떻게 동작)", re.I)
_ANALYST_HINTS = re.compile(r"(구문|분석|품사|문장 성분|관계절|도치|가정법|비교급|분열문|정확한 구조)", re.I)
_READER_HINTS  = re.compile(r"(요약|정리|해석|번역|독해|요지|주제|제목)", re.I)

def choose_prompt_mode(user_text: str) -> str:
    """
    사용자의 입력 문구로 프롬프트 모드를 자동 선택.
    반환: 'explainer' | 'analyst' | 'reader'
    """
    t = (user_text or "").strip()
    if not t:
        return "explainer"
    if _READER_HINTS.search(t):
        return "reader"
    if _ANALYST_HINTS.search(t):
        return "analyst"
    if _EXPLAIN_HINTS.search(t):
        return "explainer"
    # 길이가 길고 문단이 많으면 독해/요약으로 유도
    if len(t) > 300 or ("\n" in t and len(t.split()) > 50):
        return "reader"
    # 기본값
    return "explainer"

# ===== [08] CHAT UI ==========================================================
def render_chat_ui():
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    st.markdown("---")

    # 학생에게 보이는 학습 모드(기존 유지)
    mode = st.radio("**어떤 도움이 필요한가요?**",
                    ["💬 이유문법 설명","🔎 구문 분석","📚 독해 및 요약"],
                    horizontal=True, key="mode_select")

    prompt = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
    if not prompt:
        return

    # 대화 로그에 사용자 메시지 표시
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # --- 최종 프롬프트 모드 결정(자동이 기본 / 관리자 수동 오버라이드 허용) ---
    use_manual = bool(st.session_state.get("use_manual_override")) and is_admin
    if use_manual:
        final_mode = st.session_state.get("manual_prompt_mode","explainer")
        origin = "관리자 수동"
    else:
        # 정책: 기본은 질문 자동 선택
        final_mode = choose_prompt_mode(prompt)
        origin = "자동 추천(질문 기반)"

    # 모드 → 프롬프트 매핑
    if final_mode == "analyst":
        selected_prompt = ANALYST_PROMPT
    elif final_mode == "reader":
        selected_prompt = READER_PROMPT
    else:
        selected_prompt = EXPLAINER_PROMPT

    # 사용자에게 현재 선택 근거를 짧게 안내(디버깅/신뢰도)
    with st.status("응답 모드 선택 중…", expanded=False):
        st.write(f"선택 근거: **{origin}** → `{final_mode}`")

    # --- 답변 생성 ---
    with st.spinner("AI 선생님이 답변을 생각하고 있어요..."):
        answer = get_text_answer(st.session_state.query_engine, prompt, selected_prompt)

    st.session_state.messages.append({"role":"assistant","content":answer})
    st.rerun()

# ===== [09] MAIN =============================================================
def main():
    # 두뇌(인덱스)가 준비된 경우
    if "query_engine" in st.session_state and not st.session_state.get(STATE_KEYS.BUILD_PAUSED):
        if is_admin:
            render_admin_panels()
            render_admin_quickbar()
        render_chat_ui(); return

    # 두뇌가 아직이면: 관리자에게는 패널, 학생에게는 대기 UI
    if is_admin:
        render_admin_panels()
        render_admin_quickbar()
        return

    render_student_waiting_view()

if __name__ == "__main__":
    main()
