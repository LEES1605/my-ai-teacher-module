# ===== [01] TOP OF FILE ======================================================
# Streamlit AI-Teacher — 관리자 가드/가독성/진행바/연동/더미응답 제거 통합본
import os, sys, time, traceback, datetime as dt
from pathlib import Path
from typing import Any
import streamlit as st

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_RUN_ON_SAVE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ===== [02] IMPORTS (src 우선, 루트 폴백) ====================================
from importlib import import_module

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# 동적 임포트로 no-redef 차단
try:
    _config  = import_module("src.config")
    _prompts = import_module("src.prompts")
    _rag     = import_module("src.rag_engine")
    _auth    = import_module("src.auth")
    _ui      = import_module("src.ui")
    _IMPORT_MODE = "src"
except ImportError:
    _config  = import_module("config")
    _prompts = import_module("prompts")
    _rag     = import_module("rag_engine")
    _auth    = import_module("auth")
    _ui      = import_module("ui")
    _IMPORT_MODE = "root"

# ── 단일 바인딩(여기서 한 번만 이름 확정) ─────────────────────────────────────
settings         = _config.settings
PERSIST_DIR      = _config.PERSIST_DIR

EXPLAINER_PROMPT = _prompts.EXPLAINER_PROMPT
ANALYST_PROMPT   = _prompts.ANALYST_PROMPT
READER_PROMPT    = _prompts.READER_PROMPT

get_or_build_index = _rag.get_or_build_index
init_llama_settings = _rag.init_llama_settings
_normalize_sa       = _rag._normalize_sa
_validate_sa        = _rag._validate_sa

admin_login_flow    = _auth.admin_login_flow

load_css            = _ui.load_css
ensure_progress_css = _ui.ensure_progress_css
safe_render_header  = _ui.safe_render_header

# ===== [03] SECRET/STRING HELPER ============================================
def _sec(value: Any) -> str:
    try:
        from pydantic.types import SecretStr
        if isinstance(value, SecretStr):
            return value.get_secret_value()
    except Exception:
        pass
    if value is None:
        return ""
    if isinstance(value, dict):
        import json
        return json.dumps(value, ensure_ascii=False)
    return str(value)

# ===== [04] PAGE SETUP & CSS/HEADER =========================================
st.set_page_config(page_title="나의 AI 영어 교사", layout="wide", initial_sidebar_state="expanded")
# 기본은 학생 모드
st.session_state.setdefault("admin_mode", False)

load_css("assets/style.css", use_bg=True, bg_path="assets/background_book.png")
ensure_progress_css()
safe_render_header(subtitle=f"임포트 경로: {_IMPORT_MODE}")

# ===== [05] LOG PANEL HELPERS ===============================================
def _log(msg: str):
    st.session_state.setdefault("_ui_logs", [])
    ts = dt.datetime.now().strftime("%H:%M:%S")
    st.session_state["_ui_logs"].append(f"[{ts}] {msg}")

def _log_exception(prefix: str, exc: Exception):
    _log(f"{prefix}: {exc}")
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    st.session_state["_ui_traceback"] = tb

def _log_kv(k, v):
    _log(f"{k}: {v}")

# ===== [06] ADMIN ENTRY / AUTH GUARD ========================================
# 상단 공구 아이콘(항상 보이되, 눌렀을 때만 인증 UI 등장)
_, _, _c3 = st.columns([0.82, 0.09, 0.09])
with _c3:
    if st.button("🛠️", key="admin_icon_top_bar"):
        st.session_state.admin_mode = True
        _log("관리자 버튼 클릭")

RAW_ADMIN_PW = _sec(getattr(settings, "ADMIN_PASSWORD", ""))
HAS_ADMIN_PW = bool(RAW_ADMIN_PW.strip())
# 비밀번호가 있고, 현재 admin_mode라면 인증을 수행
is_admin = admin_login_flow(RAW_ADMIN_PW) if HAS_ADMIN_PW and st.session_state.get("admin_mode") else False
# 최종 관리자 여부(학생 화면 봉인 기준)
effective_admin = bool(st.session_state.get("admin_mode") and is_admin)

# ===== [06.5] 작은 유틸: 선형 눈금 스케일 ===================================
def render_step_scale(pct: int, steps=(0, 25, 50, 75, 100)):
    pct = max(0, min(100, int(pct)))
    marks = []
    for s in steps:
        filled = pct >= s
        marks.append(
            f"<div class='step-mark{' step-filled' if filled else ''}' title='{s}%'>{s}</div>"
        )
    st.markdown(
        f"""
        <div class="step-scale">
            {''.join(marks)}
        </div>
        """,
        unsafe_allow_html=True
    )

# ===== [07] 2-COLUMN LAYOUT (오른쪽 로그는 관리자 전용) ======================
left, right = st.columns([0.66, 0.34], gap="large")
with right:
    if effective_admin:                 # ✅ 학생에겐 안 보임
        st.markdown("### 🔎 로그 / 오류 메시지")
        st.caption("진행/오류 메시지가 여기에 누적됩니다. 복붙해서 공유하세요.")
        st.code("\n".join(st.session_state.get("_ui_logs", [])) or "로그 없음", language="text")
        st.markdown("**Traceback (있다면)**")
        st.code(st.session_state.get("_ui_traceback", "") or "(없음)", language="text")

# ===== [08] SIDEBAR — 관리자 패널(가드 철저) ================================
with st.sidebar:
    if effective_admin:
        if st.button("🔒 관리자 모드 끄기"):
            st.session_state.admin_mode = False
            _log("관리자 모드 끔")
            st.rerun()

        st.markdown("## ⚙️ 관리자 패널")

        # --- 응답 모드 수동 오버라이드 --------------------------------------
        st.markdown("### 🧭 응답 모드(관리자 오버라이드)")
        st.session_state.setdefault("use_manual_override", False)
        st.session_state["use_manual_override"] = st.checkbox(
            "수동 모드 사용", value=st.session_state["use_manual_override"]
        )
        st.session_state.setdefault("manual_prompt_mode", "explainer")
        st.session_state["manual_prompt_mode"] = st.selectbox(
            "수동 모드 선택", ["explainer", "analyst", "reader"],
            index=["explainer", "analyst", "reader"].index(st.session_state["manual_prompt_mode"])
        )

        # --- LLM/RAG 파라미터 + 자동 권장값 연동 -----------------------------
        with st.expander("🤖 RAG/LLM 설정", expanded=False):
            RECOMMENDED = {
                "compact":        {"k": 5, "temp": 0.0},
                "refine":         {"k": 7, "temp": 0.2},
                "tree_summarize": {"k": 9, "temp": 0.1},
            }
            st.session_state.setdefault("response_mode", getattr(settings, "RESPONSE_MODE", "compact"))
            st.session_state.setdefault("similarity_top_k", getattr(settings, "SIMILARITY_TOP_K", 5))
            st.session_state.setdefault("temperature", 0.0)
