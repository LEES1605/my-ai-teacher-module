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
import shutil
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
st.set_page_config(page_title="나의 AI 영어 교사", layout="wide", initial_sidebar_state="expanded")
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

# ===== [06] AUTO ATTACH/RESTORE (SILENT) =====================================
def _secret_or_str(v):
    try: return v.get_secret_value()
    except Exception: return str(v)

def _attach_from_local() -> bool:
    if not os.path.exists(PERSIST_DIR):
        return False
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

def _restore_from_drive() -> bool:
    creds = _validate_sa(_normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON))
    dest = getattr(settings, "BACKUP_FOLDER_ID", None) or get_effective_gdrive_folder_id()
    if try_restore_index_from_drive(creds, PERSIST_DIR, dest):
        return _attach_from_local()
    return False

def _auto_attach_or_restore_silently() -> bool:
    try:
        return _attach_from_local() or _restore_from_drive()
    except Exception as e:
        st.session_state["_attach_error"] = repr(e)
        return False

if "query_engine" not in st.session_state:
    _auto_attach_or_restore_silently()

# ===== [06.5] 강의 준비 (메인 화면 섹션) ======================================
def render_brain_prep_main():
    st.markdown("### 🧠 강의 준비")
    c1, c2 = st.columns([0.4, 0.6])

    with c1:
        if st.button("🧠 AI 두뇌 준비(복구/연결)", type="primary"):
            ok = _auto_attach_or_restore_silently()
            if ok:
                st.success("두뇌 연결이 완료되었습니다.")
                st.rerun()
            else:
                st.error("두뇌 연결에 실패했습니다. 서비스 계정/폴더 권한을 확인하세요.")

        if st.button("📥 강의 자료 다시 불러오기 (두뇌 초기화)"):
            try:
                if os.path.exists(PERSIST_DIR):
                    shutil.rmtree(PERSIST_DIR)
                if "query_engine" in st.session_state:
                    del st.session_state["query_engine"]
                st.success("두뇌 파일이 초기화되었습니다. ‘AI 두뇌 준비’를 다시 눌러주세요.")
            except Exception as e:
                st.error("초기화 중 오류가 발생했습니다.")
                st.exception(e)

    with c2:
        st.info(
            "- ‘AI 두뇌 준비’는 로컬 저장본이 있으면 연결하고, 없으면 Drive에서 복구합니다.\n"
            "- 서비스 계정 권한과 폴더 ID가 올바른지 확인하세요."
        )

# ===== [07] 간단한 자동 라우팅(질문 → 프롬프트 모드) ==========================
_EXPLAIN_HINTS = re.compile(r"(왜|이유|설명|원리|규칙|뜻이|무슨 의미|어떻게 동작)", re.I)
_ANALYST_HINTS = re.compile(r"(구문|분석|품사|문장 성분|관계절|도치|가정법|비교급|분열문|정확한 구조)", re.I)
_READER_HINTS  = re.compile(r"(요약|정리|해석|번역|독해|요지|주제|제목)", re.I)

def choose_prompt_mode(user_text: str) -> str:
    t = (user_text or "").strip()
    if not t:
        return "explainer"
    if _READER_HINTS.search(t):
        return "reader"
    if _ANALYST_HINTS.search(t):
        return "analyst"
    if _EXPLAIN_HINTS.search(t):
        return "explainer"
    if len(t) > 300 or ("\n" in t and len(t.split()) > 50):
        return "reader"
    return "explainer"

# ===== [08] CHAT UI ==========================================================
def render_chat_ui():
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    st.markdown("---")

    # 학생에게 보이는 학습 모드 UI(표시만, 실제 분기는 자동 우선)
    st.radio("**어떤 도움이 필요한가요?**",
             ["💬 이유문법 설명","🔎 구문 분석","📚 독해 및 요약"],
             horizontal=True, key="mode_select")

    prompt = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
    if not prompt:
        return

    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)

    # 자동 우선, 관리자 수동 오버라이드 시 덮어쓰기
    if bool(st.session_state.get("use_manual_override")) and is_admin:
        final_mode = st.session_state.get("manual_prompt_mode","explainer")
        origin = "관리자 수동"
    else:
        final_mode = choose_prompt_mode(prompt)
        origin = "자동 추천(질문 기반)"

    if final_mode == "analyst":
        selected_prompt = ANALYST_PROMPT
    elif final_mode == "reader":
        selected_prompt = READER_PROMPT
    else:
        selected_prompt = EXPLAINER_PROMPT

    with st.status("응답 모드 선택 중…", expanded=False):
        st.write(f"선택 근거: **{origin}** → `{final_mode}`")

    with st.spinner("AI 선생님이 답변을 생각하고 있어요..."):
        answer = get_text_answer(st.session_state.query_engine, prompt, selected_prompt)

    st.session_state.messages.append({"role":"assistant","content":answer})
    st.rerun()

# ===== [09] MAIN =============================================================
def main():
    # 사이드바: 관리자 패널(응답 자동/수동 + 설정 요약)
    if is_admin:
        render_admin_panels()
        render_admin_quickbar()

    # 두뇌가 준비된 경우 → 채팅
    if "query_engine" in st.session_state and not st.session_state.get(STATE_KEYS.BUILD_PAUSED):
        render_chat_ui()
        return

    # 두뇌 미준비 시
    if is_admin:
        render_brain_prep_main()   # ✅ 강의 준비는 메인 화면에서
    else:
        render_student_waiting_view()

if __name__ == "__main__":
    main()
