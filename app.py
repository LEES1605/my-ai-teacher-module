# ===== [01] TOP OF FILE ======================================================
# Streamlit AI-Teacher — 가독성 강화 + 관리자 가드(인증 전 완전 비표시) 버전
# - UI 유틸(배경/CSS/헤더/진행바) 내장
# - src 패키지 실패 시 루트 모듈 폴백
# - 다크 테마 폴백 + 사이드바 고대비 + 입력칸/본문 색 분리

# ===== [02] ENV VARS =========================================================
import os, time, re, datetime as dt, traceback, base64
from pathlib import Path
import sys
import streamlit as st

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_RUN_ON_SAVE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ===== [03] IMPORTS (with fallback) =========================================
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

_imp_err = None
try:
    # 1차: 표준 src 패키지 경로
    from src.config import settings, PERSIST_DIR
    from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
    from src.rag_engine import (
        get_or_build_index, init_llama_settings, get_text_answer,
        _normalize_sa, _validate_sa, try_restore_index_from_drive
    )
    from src.auth import admin_login_flow
    _IMPORT_MODE = "src"
except Exception as e:
    _imp_err = e
    try:
        # 2차: 루트 모듈로 폴백
        import config as _config
        settings = _config.settings
        PERSIST_DIR = _config.PERSIST_DIR

        import prompts as _prompts
        EXPLAINER_PROMPT = _prompts.EXPLAINER_PROMPT
        ANALYST_PROMPT = _prompts.ANALYST_PROMPT
        READER_PROMPT  = _prompts.READER_PROMPT

        import rag_engine as _rag
        get_or_build_index = _rag.get_or_build_index
        init_llama_settings = _rag.init_llama_settings
        get_text_answer     = _rag.get_text_answer
        _normalize_sa       = _rag._normalize_sa
        _validate_sa        = _rag._validate_sa
        try_restore_index_from_drive = _rag.try_restore_index_from_drive

        import auth as _auth
        admin_login_flow = _auth.admin_login_flow

        _IMPORT_MODE = "root"
    except Exception as ee:
        raise ImportError(
            "핵심 모듈 임포트 실패.\n"
            "1) src/ 패키지 구조를 확인하거나(src/__init__.py 포함)\n"
            "2) 또는 루트에 config.py, prompts.py, rag_engine.py, auth.py 가 있는지 확인하세요.\n"
            f"[1차:{repr(_imp_err)}]\n[2차:{repr(ee)}]"
        )

# ===== [03.5] STREAMLIT CACHE COMPAT ========================================
def _compat_cache_data(**kwargs):
    if hasattr(st, "cache_data"): return st.cache_data(**kwargs)
    if hasattr(st, "cache"):      return st.cache(**kwargs)
    def _noop(fn): return fn
    return _noop

# ===== [04] INLINE UI UTILITIES =============================================
@_compat_cache_data(show_spinner=False)
def _read_text(path_str: str) -> str:
    try: return Path(path_str).read_text(encoding="utf-8")
    except Exception: return ""

@_compat_cache_data(show_spinner=False)
def _file_b64(path_str: str) -> str:
    try: return base64.b64encode(Path(path_str).read_bytes()).decode()
    except Exception: return ""

def load_css(file_path: str, use_bg: bool = False, bg_path: str | None = None) -> None:
    css = _read_text(file_path) or ""
    bg_css = ""
    if use_bg and bg_path:
        b64 = _file_b64(bg_path)
        if b64:
            bg_css = f"""
            .stApp{{
              background-image:url("data:image/png;base64,{b64}");
              background-size:cover;background-position:center;
              background-repeat:no-repeat;background-attachment:fixed;
            }}
            """
    st.markdown(f"<style>{bg_css}\n{css}</style>", unsafe_allow_html=True)

def safe_render_header(
    title: str | None = None,
    subtitle: str | None = None,
    logo_path: str | None = "assets/academy_logo.png",
    logo_height_px: int | None = None,
) -> None:
    _title = title or getattr(settings, "TITLE_TEXT", "나의 AI 영어 교사")
    _subtitle = subtitle or getattr(settings, "SUBTITLE_TEXT", "")
    _logo_h = int(logo_height_px or getattr(settings, "LOGO_HEIGHT_PX", 56))
    logo_b64 = _file_b64(logo_path) if logo_path else ""
    st.markdown(
        f"""
        <style>
        .aihdr-wrap{{display:flex;align-items:center;gap:14px;margin:6px 0 10px;}}
        .aihdr-logo{{height:{_logo_h}px;width:auto;object-fit:contain;display:block}}
        .aihdr-title{{font-size:{getattr(settings,'TITLE_SIZE_REM',2.2)}rem;color:{getattr(settings,'BRAND_COLOR','#F8FAFC')};margin:0}}
        .aihdr-sub{{color:#C7D2FE;margin:2px 0 0 0;}}
        </style>
        """, unsafe_allow_html=True,
    )
    left, _right = st.columns([0.85, 0.15])
    with left:
        st.markdown(
            f"""
            <div class="aihdr-wrap">
              {'<img src="data:image/png;base64,'+logo_b64+'" class="aihdr-logo"/>' if logo_b64 else ''}
              <div>
                <h1 class="aihdr-title">{_title}</h1>
                {f'<div class="aihdr-sub">{_subtitle}</div>' if _subtitle else ''}
              </div>
            </div>
            """, unsafe_allow_html=True,
        )

def ensure_progress_css() -> None:
    st.markdown("""
    <style>
      .gp-wrap{ width:100%; height:28px; border-radius:12px;
        background:#1f2937; border:1px solid #334155;
        position:relative; overflow:hidden; box-shadow:0 4px 14px rgba(0,0,0,.25);
      }
      .gp-fill{ height:100%; background:linear-gradient(90deg,#7c5ad9,#9067C6); transition:width .25s ease; }
      .gp-label{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
        font-weight:800; color:#E8EDFF; text-shadow:0 1px 2px rgba(0,0,0,.5); font-size:18px; pointer-events:none;
      }
      .gp-msg{ margin-top:.5rem; color:#E8EDFF; opacity:.9; font-size:0.95rem; }
    </style>
    """, unsafe_allow_html=True)

def render_progress_bar(slot, pct: int) -> None:
    pct = max(0, min(100, int(pct)))
    slot.markdown(
        f"""
        <div class="gp-wrap">
          <div class="gp-fill" style="width:{pct}%"></div>
          <div class="gp-label">{pct}%</div>
        </div>
        """, unsafe_allow_html=True,
    )

# ===== [05] PAGE SETUP =======================================================
st.set_page_config(
    page_title="나의 AI 영어 교사",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 배경/스타일 로딩 + 가독성 폴백 + 사이드바 고대비
_BG_PATH = "assets/background_book.png"
load_css("assets/style.css", use_bg=True, bg_path=_BG_PATH)

st.markdown("""
<style>
/* 전체 폴백 다크 */
.stApp{ background:#0b1220 !important; color:#E8EDFF !important; }
h1,h2,h3,h4,h5,h6{ color:#F8FAFC !important; }

/* 헤더/툴바 투명 */
[data-testid="stHeader"],[data-testid="stToolbar"]{ background:transparent !important; }

/* ===== 사이드바 고대비 ===== */
[data-testid="stSidebar"]{ display:block!important; background:#0f172a !important; border-right:1px solid #334155; }
[data-testid="stSidebar"] *{ color:#E8EDFF !important; }
[data-testid="stSidebar"] .stButton>button{ background:#334155 !important; border:1px solid #475569 !important; }

/* ===== 입력칸(텍스트/비번/에어리어): 본문 대비 ↑ ===== */
[data-testid="stTextInput"] input, input[type="text"], input[type="password"], textarea{
  background:#111827 !important;    /* 더 어두운 입력칸 */
  border:1px solid #374151 !important;
  color:#F9FAFB !important; caret-color:#F9FAFB !important;
  border-radius:10px !important;
}
[data-testid="stTextInput"] input::placeholder, textarea::placeholder{ color:#9CA3AF !important; }

/* 알림/카드 톤 */
[data-testid="stAlert"]{ background:#111827 !important; border:1px solid #334155 !important; }
[data-testid="stAlert"] p{ color:#E8EDFF !important; }

/* 채팅 버블(본문 내용 가독성) */
[data-testid="stChatMessage"]{
  background:#0f172a !important; border:1px solid #273449 !important;
  border-radius:12px; padding:1rem; margin-bottom:1rem;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li{ color:#E8EDFF !important; }

/* 버튼 기본 */
.stButton>button{
  border-radius:999px; border:1px solid #6b7280;
  background:#4f46e5; color:#fff; font-weight:700; padding:10px 18px;
}
.stButton>button:hover{ background:#4338ca; }

/* 라디오/슬라이더 라벨 색 고정 */
[data-testid="stRadio"] label p, [data-testid="stSlider"] *{ color:#E8EDFF !important; }

/* 우측 로그 박스 코드 색 */
pre, code{ color:#CFE3FF !important; }
</style>
""", unsafe_allow_html=True)

ensure_progress_css()
safe_render_header(subtitle=f"임포트 경로: {_IMPORT_MODE}")

# ===== [06] LOG PANEL ========================================================
def _log(msg: str):
    st.session_state.setdefault("_ui_logs", [])
    ts = dt.datetime.now().strftime("%H:%M:%S")
    st.session_state["_ui_logs"].append(f"[{ts}] {msg}")

def _log_exception(prefix: str, exc: Exception):
    _log(f"{prefix}: {exc}")
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    st.session_state["_ui_traceback"] = tb

def _log_kv(k, v): _log(f"{k}: {v}")

# ===== [07] ADMIN ENTRY / GUARD =============================================
# 상단 관리자 아이콘만 항상 보이게 (패널은 인증 전 완전 숨김)
_, _, _c3 = st.columns([0.82, 0.09, 0.09])
with _c3:
    if st.button("🛠️", key="admin_icon_top_bar"):
        st.session_state.admin_mode = True
        _log("관리자 버튼 클릭")

# 인증 실행 (패널은 아래 가드로 제어)
is_admin = admin_login_flow(getattr(settings, "ADMIN_PASSWORD", "") or "")

# ===== [08] 2-COLUMN LAYOUT ==================================================
left, right = st.columns([0.66, 0.34], gap="large")
with right:
    st.markdown("### 🔎 로그 / 오류 메시지")
    st.caption("진행/오류 메시지가 여기에 누적됩니다. 복붙해서 공유하세요.")
    st.code("\n".join(st.session_state.get("_ui_logs", [])) or "로그 없음", language="text")
    st.markdown("**Traceback (있다면)**")
    st.code(st.session_state.get("_ui_traceback", "") or "(없음)", language="text")

# ===== [09] SIDEBAR (ADMIN-ONLY CONTENT) ====================================
# ✅ 관리자 인증 전에는 사이드바 관리자 패널 '전부 미표시'
with st.sidebar:
    if is_admin:
        st.markdown("## ⚙️ 관리자 패널")
        # 응답 모드
        st.markdown("### 🧭 응답 모드")
        st.session_state.setdefault("use_manual_override", False)
        st.session_state["use_manual_override"] = st.checkbox(
            "수동 모드(관리자 오버라이드) 사용", value=st.session_state["use_manual_override"])
        st.session_state.setdefault("manual_prompt_mode", "explainer")
        st.session_state["manual_prompt_mode"] = st.selectbox(
            "수동 모드 선택", ["explainer","analyst","reader"],
            index=["explainer","analyst","reader"].index(st.session_state["manual_prompt_mode"])
        )
        # RAG/LLM
        with st.expander("🤖 RAG/LLM 설정", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.setdefault("similarity_top_k", getattr(settings,"SIMILARITY_TOP_K",5))
                st.session_state["similarity_top_k"] = st.slider("similarity_top_k", 1, 12, int(st.session_state["similarity_top_k"]))
            with c2:
                st.session_state.setdefault("temperature", 0.0)
                st.session_state["temperature"] = st.slider("LLM temperature", 0.0, 1.0, float(st.session_state["temperature"]), 0.05)
            with c3:
                st.session_state.setdefault("response_mode", getattr(settings,"RESPONSE_MODE","compact"))
                st.session_state["response_mode"] = st.selectbox(
                    "response_mode", ["compact","refine","tree_summarize"],
                    index=["compact","refine","tree_summarize"].index(st.session_state["response_mode"])
                )
        # 도구
        with st.expander("🛠️ 관리자 도구", expanded=False):
            if st.button("↺ 두뇌 초기화(인덱스 삭제)"):
                import shutil
                try:
                    if os.path.exists(PERSIST_DIR): shutil.rmtree(PERSIST_DIR)
                    st.session_state.pop("query_engine", None)
                    _log("두뇌 초기화 완료"); st.success("두뇌 파일 삭제됨. 메인에서 다시 준비하세요.")
                except Exception as e:
                    _log_exception("두뇌 초기화 실패", e); st.error("초기화 중 오류. 우측 로그/Traceback 확인.")
    else:
        # 관리자 전용 요소 완전 미표시 (빈 사이드바 유지)
        pass

# ===== [10] MAIN: 강의 준비 & 진단 & 채팅 ===================================
with left:
    # --- [10-1] 두뇌 준비 ----------------------------------------------------
    if "query_engine" not in st.session_state:
        st.markdown("## 📚 강의 준비")
        st.info("‘AI 두뇌 준비’는 로컬 저장본이 있으면 연결하고, 없으면 Drive에서 복구합니다.\n서비스 계정 권한과 폴더 ID가 올바른지 확인하세요.")

        btn_col, diag_col = st.columns([0.55, 0.45])
        with btn_col:
            if st.button("🧠 AI 두뇌 준비(복구/연결)"):
                bar_slot = st.empty()
                msg_slot = st.empty()
                key = "_gp_pct"
                st.session_state[key] = 0

                def update_pct(p, m=None):
                    st.session_state[key] = max(0, min(100, int(p)))
                    render_progress_bar(bar_slot, st.session_state[key])
                    if m:
                        msg_slot.markdown(f"<div class='gp-msg'>{m}</div>", unsafe_allow_html=True)
                        _log(m)

                try:
                    # 0) 시작 메시지
                    update_pct(0, "두뇌 준비를 시작합니다…")

                    # 1) LLM 초기화(여기서 실패하는 경우가 많음)
                    try:
                        init_llama_settings(
                            api_key=settings.GEMINI_API_KEY.get_secret_value(),
                            llm_model=settings.LLM_MODEL,
                            embed_model=settings.EMBED_MODEL,
                            temperature=float(st.session_state.get("temperature", 0.0))
                        )
                        _log("LLM 설정 완료")
                        update_pct(2, "설정 확인 중…")
                    except Exception as ee:
                        # rag_engine의 사용자친화 예외라면 public_msg 노출
                        public = getattr(ee, "public_msg", str(ee))
                        _log_exception("LLM 초기화 실패", ee)
                        st.error(f"LLM 초기화 실패: {public}")
                        st.stop()

                    # 2) 인덱스 로드/복구
                    try:
                        folder_id = getattr(settings, "GDRIVE_FOLDER_ID", None) or getattr(settings, "BACKUP_FOLDER_ID", None)
                        raw_sa = getattr(settings, "GDRIVE_SERVICE_ACCOUNT_JSON", None)
                        persist_dir = PERSIST_DIR

                        # 진단 스냅샷(실패 시 우측에 함께 보이도록 선기록)
                        _log_kv("PERSIST_DIR", persist_dir)
                        _log_kv("folder_id", str(folder_id or "(empty)"))
                        _log_kv("has_service_account", "yes" if raw_sa else "no")

                        index = get_or_build_index(
                            update_pct=update_pct,
                            update_msg=lambda m: update_pct(st.session_state[key], m),
                            gdrive_folder_id=folder_id,
                            raw_sa=raw_sa,
                            persist_dir=persist_dir,
                            manifest_path=getattr(settings, "MANIFEST_PATH", None)
                        )
                    except Exception as ee:
                        public = getattr(ee, "public_msg", str(ee))
                        _log_exception("인덱스 준비 실패", ee)
                        st.error(f"두뇌 준비 실패: {public}")
                        st.stop()

                    # 3) QueryEngine 생성
                    try:
                        st.session_state.query_engine = index.as_query_engine(
                            response_mode=st.session_state.get("response_mode", getattr(settings, "RESPONSE_MODE", "compact")),
                            similarity_top_k=int(st.session_state.get("similarity_top_k", getattr(settings, "SIMILARITY_TOP_K", 5)))
                        )
                        update_pct(100, "두뇌 준비 완료!")
                        _log("query_engine 생성 완료 ✅")
                        time.sleep(0.3)
                        st.rerun()
                    except Exception as ee:
                        public = getattr(ee, "public_msg", str(ee))
                        _log_exception("QueryEngine 생성 실패", ee)
                        st.error(f"두뇌 준비는 되었으나 QueryEngine 생성에서 실패: {public}")
                        st.stop()

                except Exception as e:
                    # 최상위 가드 — 어떤 예외라도 우측 Traceback은 반드시 찍힌다
                    _log_exception("예상치 못한 오류", e)
                    st.error("두뇌 준비 중 알 수 없는 오류. 우측 로그/Traceback을 확인하세요.")
                    st.stop()

            if st.button("📥 강의 자료 다시 불러오기(두뇌 초기화)"):
                import shutil
                try:
                    if os.path.exists(PERSIST_DIR): shutil.rmtree(PERSIST_DIR)
                    st.session_state.pop("query_engine", None)
                    _log("본문에서 두뇌 초기화 실행")
                    st.success("두뇌 파일을 삭제했습니다. 다시 ‘AI 두뇌 준비’를 눌러주세요.")
                except Exception as e:
                    _log_exception("본문 초기화 실패", e)
                    st.error("초기화 중 오류. 우측 로그/Traceback 확인.")

        with diag_col:
            st.markdown("#### 🧪 연결 진단(빠름)")
            st.caption("로컬 캐시/SA/폴더 ID/Drive 복구를 검사하고 로그에 기록합니다.")
            if st.button("연결 진단 실행"):
                try:
                    _log_kv("PERSIST_DIR", PERSIST_DIR)
                    if os.path.exists(PERSIST_DIR):
                        _log_kv("local_cache", f"exists ✅, files={len(os.listdir(PERSIST_DIR))}")
                    else:
                        _log_kv("local_cache", "missing ❌")

                    try:
                        from src.rag_engine import _normalize_sa, _validate_sa  # 사용 시점 임포트
                    except Exception:
                        from rag_engine import _normalize_sa, _validate_sa

                    try:
                        sa_norm = _normalize_sa(getattr(settings,"GDRIVE_SERVICE_ACCOUNT_JSON", None))
                        creds = _validate_sa(sa_norm)
                        _log("service_account: valid ✅")
                        _log_kv("sa_client_email", creds.get("client_email","(unknown)"))
                    except Exception as se:
                        _log_exception("service_account invalid ❌", se)

                    folder_id = getattr(settings, "BACKUP_FOLDER_ID", None) or getattr(settings, "GDRIVE_FOLDER_ID", None)
                    _log_kv("folder_id", str(folder_id or "(empty)"))
                    st.success("진단 완료. 우측 로그/Traceback 확인하세요.")
                except Exception as e:
                    _log_exception("연결 진단 자체 실패", e)
                    st.error("연결 진단 중 오류. 우측 로그/Traceback 확인.")
        st.stop()

    # --- [10-2] 채팅 UI ------------------------------------------------------
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    st.markdown("---")

    mode_label = st.radio("**어떤 도움이 필요한가요?**",
                          ["💬 이유문법 설명","🔎 구문 분석","📚 독해 및 요약"],
                          horizontal=True, key="mode_select")
    prompt = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
    if not prompt: st.stop()

    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if is_admin and st.session_state.get("use_manual_override"):
        final_mode = st.session_state.get("manual_prompt_mode","explainer"); origin="관리자 수동"
    else:
        final_mode = "explainer" if mode_label.startswith("💬") else "analyst" if mode_label.startswith("🔎") else "reader"
        origin="학생 선택"

    selected_prompt = EXPLAINER_PROMPT if final_mode=="explainer" else ANALYST_PROMPT if final_mode=="analyst" else READER_PROMPT
    _log(f"모드 결정: {origin} → {final_mode}")

    try:
        with st.spinner("AI 선생님이 답변을 생각하고 있어요..."):
            answer = get_text_answer(st.session_state.query_engine, prompt, selected_prompt)
        st.session_state.messages.append({"role":"assistant","content":answer}); st.rerun()
    except Exception as e:
        _log_exception("답변 생성 실패", e); st.error("답변 생성 중 오류. 우측 로그/Traceback 확인.")

# ===== [11] END OF FILE ======================================================
