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
from pathlib import Path
import sys

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
            st.session_state.setdefault("_last_response_mode", st.session_state["response_mode"])
            st.session_state.setdefault("auto_tune_llm", True)

            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.session_state["response_mode"] = st.selectbox(
                    "response_mode", ["compact", "refine", "tree_summarize"],
                    index=["compact", "refine", "tree_summarize"].index(st.session_state["response_mode"])
                )
                st.session_state["auto_tune_llm"] = st.checkbox("자동 권장값 연동", value=st.session_state["auto_tune_llm"])

            if st.session_state["auto_tune_llm"]:
                if st.session_state["response_mode"] != st.session_state["_last_response_mode"]:
                    rec = RECOMMENDED[st.session_state["response_mode"]]
                    st.session_state["similarity_top_k"] = rec["k"]
                    st.session_state["temperature"] = rec["temp"]
                    st.session_state["_last_response_mode"] = st.session_state["response_mode"]

            with c2:
                st.session_state["similarity_top_k"] = st.slider(
                    "similarity_top_k", 1, 12, int(st.session_state["similarity_top_k"])
                )
            with c3:
                st.session_state["temperature"] = st.slider(
                    "LLM temperature", 0.0, 1.0, float(st.session_state["temperature"]), 0.05
                )

        with st.expander("📤 강의 자료 업로드(베타)", expanded=False):
            st.caption("원본은 Drive의 prepared 폴더에, 최적화 결과는 backup 폴더에 저장(설계 반영).")
            uf = st.file_uploader("자료 업로드", type=["pdf", "docx", "pptx", "txt", "md", "csv", "zip"], accept_multiple_files=False)
            if uf is not None:
                tmp_dir = Path("/tmp/ai_teacher_uploads"); tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / uf.name
                tmp_path.write_bytes(uf.getbuffer())
                _log(f"업로드 수신: {tmp_path}")
                st.success("업로드 파일을 받았습니다. (Drive 업로드/최적화 파이프라인 연결 필요)")

# ===== [09] MAIN — 강의 준비 / 채팅 =========================================
with left:
    # --- 관리자 전용: 두뇌 준비 워크플로우 -----------------------------------
    if ("query_engine" not in st.session_state) and effective_admin:
        st.markdown("## 📚 강의 준비")
        st.info("‘AI 두뇌 준비’는 로컬 저장본이 있으면 연결하고, 없으면 Drive에서 복구합니다.\n서비스 계정 권한과 폴더 ID가 올바른지 확인하세요.")

        btn_col, diag_col = st.columns([0.55, 0.45])
        with btn_col:
            # 진행 영역(기본 progress + 스텝눈금)
            _prog_slot = st.empty()  # (미사용 변수 경고 예방)
            scale_slot = st.empty()
            msg_slot = st.empty()
            bar = st.progress(0)
            st.session_state["_gp_pct"] = 0

            def update_pct(pct: int, msg: str | None = None):
                pct = max(0, min(100, int(pct)))
                st.session_state["_gp_pct"] = pct
                bar.progress(pct)
                with scale_slot: render_step_scale(pct)
                if msg:
                    msg_slot.markdown(f"<div class='gp-msg'>{msg}</div>", unsafe_allow_html=True)
                    _log(msg)

            if st.button("🧠 AI 두뇌 준비(복구/연결)"):
                try:
                    update_pct(0, "두뇌 준비를 시작합니다…")

                    # 1) LLM 초기화
                    init_llama_settings(
                        api_key=_sec(getattr(settings, "GEMINI_API_KEY", "")),
                        llm_model=settings.LLM_MODEL,
                        embed_model=settings.EMBED_MODEL,
                        temperature=float(st.session_state.get("temperature", 0.0))
                    )
                    _log("LLM 설정 완료"); update_pct(2, "설정 확인 중…")

                    # 2) 인덱스 로드/복구
                    folder_id = getattr(settings, "GDRIVE_FOLDER_ID", None) or getattr(settings, "BACKUP_FOLDER_ID", None)
                    raw_sa = getattr(settings, "GDRIVE_SERVICE_ACCOUNT_JSON", None)
                    persist_dir = PERSIST_DIR
                    _log_kv("PERSIST_DIR", persist_dir)
                    _log_kv("local_cache", "exists ✅" if os.path.exists(persist_dir) else "missing ❌")
                    _log_kv("folder_id", str(folder_id or "(empty)"))
                    _log_kv("has_service_account", "yes" if raw_sa else "no")

                    def _update_pct_hook(p, m=None): update_pct(p, m)
                    index = get_or_build_index(
                        update_pct=_update_pct_hook,
                        update_msg=lambda m: _update_pct_hook(st.session_state["_gp_pct"], m),
                        gdrive_folder_id=folder_id,
                        raw_sa=raw_sa,
                        persist_dir=persist_dir,
                        manifest_path=getattr(settings, "MANIFEST_PATH", None)
                    )
                    # 3) QueryEngine
                    st.session_state.query_engine = index.as_query_engine(
                        response_mode=st.session_state.get("response_mode", getattr(settings,"RESPONSE_MODE","compact")),
                        similarity_top_k=int(st.session_state.get("similarity_top_k", getattr(settings,"SIMILARITY_TOP_K",5)))
                    )
                    update_pct(100, "두뇌 준비 완료!"); _log("query_engine 생성 완료 ✅")
                    time.sleep(0.2); st.rerun()

                except Exception as e:
                    _log_exception("두뇌 준비 실패", e)
                    st.error(getattr(e, "public_msg", "두뇌 준비 중 오류. 우측 로그/Traceback을 확인하세요."))

            if st.button("📥 강의 자료 다시 불러오기(두뇌 초기화)"):
                import shutil
                try:
                    if os.path.exists(PERSIST_DIR): shutil.rmtree(PERSIST_DIR)
                    st.session_state.pop("query_engine", None)
                    _log("본문에서 두뇌 초기화 실행"); st.success("두뇌 파일 삭제됨. 다시 ‘AI 두뇌 준비’를 눌러주세요.")
                except Exception as e:
                    _log_exception("본문 초기화 실패", e); st.error("초기화 중 오류. 우측 Traceback 확인.")

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
                    # SA 검사
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
                    st.error("연결 진단 중 오류. 우측 Traceback 확인.")
        st.stop()

    # --- 학생/관리자 공통: 채팅 UI ---------------------------------------------
    if "query_engine" not in st.session_state and not effective_admin:
        st.markdown("## 👋 준비 중")
        st.info("수업 준비가 완료되면 챗이 열립니다. 잠시만 기다려 주세요.")
        st.stop()

    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    st.markdown("---")

    mode_label = st.radio("**어떤 도움이 필요한가요?**",
                          ["💬 이유문법 설명","🔎 구문 분석","📚 독해 및 요약"],
                          horizontal=True, key="mode_select")
    user_text = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
    if not user_text: st.stop()

    st.session_state.messages.append({"role":"user","content":user_text})
    with st.chat_message("user"): st.markdown(user_text)

    # 관리자 수동 오버라이드(관리자일 때만 작동)
    if effective_admin and st.session_state.get("use_manual_override"):
        final_mode = st.session_state.get("manual_prompt_mode","explainer"); origin="관리자 수동"
    else:
        final_mode = "explainer" if mode_label.startswith("💬") else "analyst" if mode_label.startswith("🔎") else "reader"
        origin="학생 선택"
    _log(f"모드 결정: {origin} → {final_mode}")

    selected_prompt = EXPLAINER_PROMPT if final_mode=="explainer" else ANALYST_PROMPT if final_mode=="analyst" else READER_PROMPT

    # [더미응답] 제거: QueryEngine 직접 호출
    try:
        with st.spinner("AI 선생님이 답변을 생각하고 있어요..."):
            qe = st.session_state.query_engine
            resp = qe.query(f"{selected_prompt}\n\n사용자 입력:\n{user_text}") if selected_prompt else qe.query(user_text)
            answer = str(resp)  # LlamaIndex Response -> str
        st.session_state.messages.append({"role":"assistant","content":answer}); st.rerun()
    except Exception as e:
        _log_exception("답변 생성 실패", e); st.error("답변 생성 중 오류. 우측 Traceback 확인.")
# ===== [10] END OF FILE ======================================================
