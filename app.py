# ===== [01] TOP OF FILE ======================================================
# Streamlit AI-Teacher — 핫픽스: NameError 제거 + 관리자 가드 강화 + 로그 패널 유지
import os, sys, time, traceback, base64, datetime as dt
from pathlib import Path
import streamlit as st

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_RUN_ON_SAVE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ===== [02] IMPORTS (src 우선, 루트 폴백) ====================================
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from src.config import settings, PERSIST_DIR
    from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
    from src.rag_engine import get_or_build_index, init_llama_settings, get_text_answer, _normalize_sa, _validate_sa
    from src.auth import admin_login_flow
    from src.ui import load_css, ensure_progress_css, safe_render_header, render_progress_bar
    _IMPORT_MODE = "src"
except Exception:
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

    import auth as _auth
    admin_login_flow = _auth.admin_login_flow

    from ui import load_css, ensure_progress_css, safe_render_header, render_progress_bar
    _IMPORT_MODE = "root"

# ===== [03] SECRET/STRING HELPER ============================================
def _sec(value) -> str:
    try:
        from pydantic.types import SecretStr
        if isinstance(value, SecretStr):
            return value.get_secret_value()
    except Exception:
        pass
    if value is None: return ""
    if isinstance(value, dict):
        import json; return json.dumps(value, ensure_ascii=False)
    return str(value)

# ===== [04] PAGE SETUP & CSS/HEADER =========================================
st.set_page_config(page_title="나의 AI 영어 교사", layout="wide", initial_sidebar_state="expanded")
st.session_state.setdefault("admin_mode", False)   # 기본은 학생 모드
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

def _log_kv(k, v): _log(f"{k}: {v}")

# ===== [06] ADMIN ENTRY / AUTH GUARD ========================================
# 상단 공구 아이콘: 관리자 모드 진입 트리거 (항상 표시)
_, _, _c3 = st.columns([0.82, 0.09, 0.09])
with _c3:
    if st.button("🛠️", key="admin_icon_top_bar"):
        st.session_state.admin_mode = True
        _log("관리자 버튼 클릭")

RAW_ADMIN_PW = _sec(getattr(settings, "ADMIN_PASSWORD", ""))
HAS_ADMIN_PW = bool(RAW_ADMIN_PW.strip())
is_admin = admin_login_flow(RAW_ADMIN_PW) if HAS_ADMIN_PW and st.session_state.get("admin_mode") else False

# ===== [07] 2-COLUMN LAYOUT (로그 패널은 항상 우측) ==========================
left, right = st.columns([0.66, 0.34], gap="large")
with right:
    st.markdown("### 🔎 로그 / 오류 메시지")
    st.caption("진행/오류 메시지가 여기에 누적됩니다. 복붙해서 공유하세요.")
    st.code("\n".join(st.session_state.get("_ui_logs", [])) or "로그 없음", language="text")
    st.markdown("**Traceback (있다면)**")
    st.code(st.session_state.get("_ui_traceback", "") or "(없음)", language="text")

# ===== [08] SIDEBAR — 관리자 패널(가드 철저) ================================
with st.sidebar:
    # 학생에게는 아무 관리자 UI도 보이지 않음
    if HAS_ADMIN_PW and st.session_state.get("admin_mode") and is_admin:
        if st.button("🔒 관리자 모드 끄기"):
            st.session_state.admin_mode = False
            _log("관리자 모드 끔")
            st.rerun()

        st.markdown("## ⚙️ 관리자 패널")

        st.markdown("### 🧭 응답 모드(관리자 오버라이드)")
        st.session_state.setdefault("use_manual_override", False)
        st.session_state["use_manual_override"] = st.checkbox(
            "수동 모드 사용", value=st.session_state["use_manual_override"]
        )
        st.session_state.setdefault("manual_prompt_mode", "explainer")
        st.session_state["manual_prompt_mode"] = st.selectbox(
            "수동 모드 선택", ["explainer","analyst","reader"],
            index=["explainer","analyst","reader"].index(st.session_state["manual_prompt_mode"])
        )

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

        with st.expander("🛠️ 관리자 도구", expanded=False):
            if st.button("↺ 두뇌 초기화(인덱스 삭제)"):
                import shutil
                try:
                    if os.path.exists(PERSIST_DIR): shutil.rmtree(PERSIST_DIR)
                    st.session_state.pop("query_engine", None)
                    _log("두뇌 초기화 완료"); st.success("두뇌 파일 삭제됨. 메인에서 다시 준비하세요.")
                except Exception as e:
                    _log_exception("두뇌 초기화 실패", e); st.error("초기화 중 오류. 우측 Traceback 확인.")
    else:
        # 학생 사이드바에 보여줄 항목이 있으면 여기 작성 (현재는 비워둠)
        pass

# ===== [09] MAIN — 강의 준비 & 연결 진단 & 채팅 =============================
with left:
    # --- [09-1] 두뇌 준비 ----------------------------------------------------
    if "query_engine" not in st.session_state:
        st.markdown("## 📚 강의 준비")
        st.info("‘AI 두뇌 준비’는 로컬 저장본이 있으면 연결하고, 없으면 Drive에서 복구합니다.\n서비스 계정 권한과 폴더 ID가 올바른지 확인하세요.")

        btn_col, diag_col = st.columns([0.55, 0.45])
        with btn_col:
            if st.button("🧠 AI 두뇌 준비(복구/연결)"):
                bar_slot = st.empty()
                msg_slot = st.empty()
                st.session_state["_gp_pct"] = 0

                def update_pct(pct, msg=None):
                    pct = max(0, min(100, int(pct)))
                    st.session_state["_gp_pct"] = pct
                    render_progress_bar(bar_slot, pct)
                    if msg: 
                        msg_slot.markdown(f"<div class='gp-msg'>{msg}</div>", unsafe_allow_html=True)
                        _log(msg)

                try:
                    update_pct(0, "두뇌 준비를 시작합니다…")

                    # 1) LLM 초기화
                    try:
                        init_llama_settings(
                            api_key=_sec(getattr(settings, "GEMINI_API_KEY", "")),
                            llm_model=settings.LLM_MODEL,
                            embed_model=settings.EMBED_MODEL,
                            temperature=float(st.session_state.get("temperature", 0.0))
                        )
                        _log("LLM 설정 완료"); update_pct(2, "설정 확인 중…")
                    except Exception as ee:
                        _log_exception("LLM 초기화 실패", ee)
                        st.error(getattr(ee, "public_msg", str(ee))); st.stop()

                    # 2) 인덱스 로드/복구
                    try:
                        folder_id = getattr(settings, "GDRIVE_FOLDER_ID", None) or getattr(settings, "BACKUP_FOLDER_ID", None)
                        raw_sa = getattr(settings, "GDRIVE_SERVICE_ACCOUNT_JSON", None)
                        persist_dir = PERSIST_DIR
                        _log_kv("PERSIST_DIR", persist_dir)
                        _log_kv("local_cache", "exists ✅" if os.path.exists(persist_dir) else "missing ❌")
                        _log_kv("folder_id", str(folder_id or "(empty)"))
                        _log_kv("has_service_account", "yes" if raw_sa else "no")

                        index = get_or_build_index(
                            update_pct=update_pct,
                            update_msg=lambda m: update_pct(st.session_state["_gp_pct"], m),
                            gdrive_folder_id=folder_id,
                            raw_sa=raw_sa,
                            persist_dir=persist_dir,
                            manifest_path=getattr(settings, "MANIFEST_PATH", None)
                        )
                    except Exception as ee:
                        _log_exception("인덱스 준비 실패", ee)
                        st.error(getattr(ee, "public_msg", str(ee))); st.stop()

                    # 3) QueryEngine 생성
                    try:
                        st.session_state.query_engine = index.as_query_engine(
                            response_mode=st.session_state.get("response_mode", getattr(settings,"RESPONSE_MODE","compact")),
                            similarity_top_k=int(st.session_state.get("similarity_top_k", getattr(settings,"SIMILARITY_TOP_K",5)))
                        )
                        update_pct(100, "두뇌 준비 완료!"); _log("query_engine 생성 완료 ✅")
                        time.sleep(0.2); st.rerun()
                    except Exception as ee:
                        _log_exception("QueryEngine 생성 실패", ee)
                        st.error(getattr(ee, "public_msg", str(ee))); st.stop()

                except Exception as e:
                    _log_exception("예상치 못한 오류", e)
                    st.error("두뇌 준비 중 알 수 없는 오류. 우측 로그/Traceback을 확인하세요.")
                    st.stop()

            if st.button("📥 강의 자료 다시 불러오기(두뇌 초기화)"):
                import shutil
                try:
                    if os.path.exists(PERSIST_DIR): shutil.rmtree(PERSIST_DIR)
                    st.session_state.pop("query_engine", None)
                    _log("본문에서 두뇌 초기화 실행"); st.success("두뇌 파일 삭제됨. 다시 ‘AI 두뇌 준비’를 눌러주세요.")
                except Exception as e:
                    _log_exception("본문 초기화 실패", e); st.error("초기화 중 오류. 우측 로그/Traceback 확인.")

        # --- [09-2] 빠른 연결 진단 -------------------------------------------
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

    # --- [09-3] 채팅 UI ------------------------------------------------------
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

    if HAS_ADMIN_PW and is_admin and st.session_state.get("admin_mode") and st.session_state.get("use_manual_override"):
        final_mode = st.session_state.get("manual_prompt_mode","explainer"); origin="관리자 수동"
    else:
        final_mode = "explainer" if mode_label.startswith("💬") else "analyst" if mode_label.startswith("🔎") else "reader"
        origin="학생 선택"
    _log(f"모드 결정: {origin} → {final_mode}")

    selected_prompt = EXPLAINER_PROMPT if final_mode=="explainer" else ANALYST_PROMPT if final_mode=="analyst" else READER_PROMPT

    try:
        with st.spinner("AI 선생님이 답변을 생각하고 있어요..."):
            answer = get_text_answer(st.session_state.query_engine, prompt, selected_prompt)
        st.session_state.messages.append({"role":"assistant","content":answer}); st.rerun()
    except Exception as e:
        _log_exception("답변 생성 실패", e); st.error("답변 생성 중 오류. 우측 로그/Traceback 확인.")

# ===== [10] END OF FILE ======================================================
