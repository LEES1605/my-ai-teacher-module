# ===== [01] TOP OF FILE ======================================================
# Streamlit AI-Teacher — 즉시 가독성 확보용 다크테마 강제 + 배경 폴백 + 입력창/비번 대비 강화

# ===== [02] ENV VARS =========================================================
import os, time, re, datetime as dt, traceback
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_RUN_ON_SAVE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ===== [03] IMPORTS ==========================================================
import streamlit as st
from src.config import settings, PERSIST_DIR
from src.ui import load_css, safe_render_header, ensure_progress_css, render_progress_bar
from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
from src.rag_engine import (
    get_or_build_index, init_llama_settings, get_text_answer,
    _normalize_sa, _validate_sa, try_restore_index_from_drive
)
from src.auth import admin_login_flow

# ===== [04] PAGE SETUP =======================================================
st.set_page_config(
    page_title="나의 AI 영어 교사",
    layout="wide",
    initial_sidebar_state="expanded"  # 사이드바 항상 보이기
)

# [04-1] 배경 이미지는 **강제로 사용** (없어도 폴백 다크 배경으로 가독성 확보)
_BG_PATH = "assets/background_book.png"
load_css("assets/style.css", use_bg=True, bg_path=_BG_PATH)

# [04-2] 혹시 외부 CSS가 사이드바를 숨겼던 경우를 대비해 강제 노출
st.markdown("<style>[data-testid='stSidebar']{display:block!important;}</style>", unsafe_allow_html=True)

# [04-3] 가시성 비상 CSS(필수 대비) — style.css가 실패해도 글씨/입력이 보이도록 보강
st.markdown("""
<style>
/* 배경 폴백 + 전역 전경색 */
.stApp{ background:#0B1220 !important; color:#F7FAFC !important; }
h1,h2,h3,h4,h5,h6{ color:#F7FAFC !important; }
[data-testid="stHeader"],[data-testid="stToolbar"]{ background:transparent !important; }
/* 입력/비밀번호 대비 강화 */
[data-testid="stTextInput"] input, input[type="text"], input[type="password"], textarea{
  background: rgba(255,255,255,0.12) !important;
  border: 1px solid rgba(255,255,255,0.25) !important;
  color: #FFFFFF !important; caret-color:#FFFFFF !important; border-radius:10px !important;
}
[data-testid="stTextInput"] input::placeholder, textarea::placeholder{ color: rgba(255,255,255,.6) !important; }
/* 사이드바 항상 표시 */
[data-testid="stSidebar"] *{ color:#F7FAFC !important; }
</style>
""", unsafe_allow_html=True)

ensure_progress_css()
safe_render_header()

# ===== [05] LOG PANEL (오른쪽 고정) =========================================
def _log(msg: str):
    st.session_state.setdefault("_ui_logs", [])
    ts = dt.datetime.now().strftime("%H:%M:%S")
    st.session_state["_ui_logs"].append(f"[{ts}] {msg}")

def _log_exception(prefix: str, exc: Exception):
    _log(f"{prefix}: {exc}")
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    st.session_state["_ui_traceback"] = tb

def _log_kv(k, v): _log(f"{k}: {v}")

# ===== [06] ADMIN AUTH & SIDEBAR ============================================
# 상단 관리자 아이콘
_, _, _c3 = st.columns([0.8, 0.1, 0.1])
with _c3:
    if st.button("🛠️", key="admin_icon_top_bar"):
        st.session_state.admin_mode = True
        _log("관리자 버튼 클릭")

is_admin = admin_login_flow(settings.ADMIN_PASSWORD or "")

with st.sidebar:
    st.markdown("## ⚙️ 관리자 패널")
    if is_admin:
        # [06-1] 응답 모드(자동/수동)
        st.markdown("### 🧭 응답 모드")
        st.session_state.setdefault("use_manual_override", False)
        st.session_state["use_manual_override"] = st.checkbox(
            "수동 모드(관리자 오버라이드) 사용", value=st.session_state["use_manual_override"]
        )
        st.session_state.setdefault("manual_prompt_mode", "explainer")
        st.session_state["manual_prompt_mode"] = st.selectbox(
            "수동 모드 선택", ["explainer","analyst","reader"],
            index=["explainer","analyst","reader"].index(st.session_state["manual_prompt_mode"])
        )

        # [06-2] LLM/RAG 설정 + response_mode 자동/수동
        with st.expander("🤖 RAG/LLM 설정", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.setdefault("similarity_top_k", settings.SIMILARITY_TOP_K)
                st.session_state["similarity_top_k"] = st.slider(
                    "similarity_top_k", 1, 12, int(st.session_state["similarity_top_k"])
                )
            with c2:
                st.session_state.setdefault("temperature", 0.0)
                st.session_state["temperature"] = st.slider(
                    "LLM temperature", 0.0, 1.0, float(st.session_state["temperature"]), 0.05
                )
            with c3:
                st.session_state.setdefault("response_mode", settings.RESPONSE_MODE)
                st.session_state["response_mode"] = st.selectbox(
                    "response_mode", ["compact","refine","tree_summarize"],
                    index=["compact","refine","tree_summarize"].index(st.session_state["response_mode"])
                )

        # [06-3] 관리자 도구
        with st.expander("🛠️ 관리자 도구", expanded=False):
            if st.button("↺ 두뇌 초기화(인덱스 삭제)"):
                import shutil
                try:
                    if os.path.exists(PERSIST_DIR):
                        shutil.rmtree(PERSIST_DIR)
                    st.session_state.pop("query_engine", None)
                    _log("두뇌 초기화 완료")
                    st.success("두뇌 파일 삭제됨. 메인에서 다시 준비하세요.")
                except Exception as e:
                    _log_exception("두뇌 초기화 실패", e)
                    st.error("초기화 중 오류. 우측 로그/Traceback 확인.")
    else:
        st.info("우측 상단 '🛠️' 버튼으로 관리자 인증을 진행하세요.")

# ===== [07] 2-Column LAYOUT ==================================================
left, right = st.columns([0.66, 0.34], gap="large")
with right:
    st.markdown("### 🔎 로그 / 오류 메시지")
    st.caption("오류나 진행 메시지가 여기에 누적됩니다. 복붙해서 공유하세요.")
    st.code("\n".join(st.session_state.get("_ui_logs", [])) or "로그 없음", language="text")
    st.markdown("**Traceback (있다면)**")
    st.code(st.session_state.get("_ui_traceback", "") or "(없음)", language="text")

# ===== [08] MAIN: 강의 준비 & 연결 진단 ======================================
with left:
    # [08-1] 두뇌 준비(로컬→드라이브 복구)
    if "query_engine" not in st.session_state:
        st.markdown("## 📚 강의 준비")
        st.info("AI 두뇌가 아직 준비되지 않았습니다. 아래 버튼을 눌러 주세요.")

        btn_col, diag_col = st.columns([0.55, 0.45])

        # --- 준비 버튼 ---
        with btn_col:
            if st.button("🧠 AI 두뇌 준비(복구/연결)"):
                try:
                    bar_slot = st.empty(); msg_slot = st.empty()
                    key = "_gp_pct"; st.session_state[key] = 0
                    def update_pct(p, m=None):
                        st.session_state[key] = max(0, min(100, int(p)))
                        render_progress_bar(bar_slot, st.session_state[key])
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
                        update_msg=lambda m: update_pct(st.session_state[key], m),
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

            # 보조: 초기화
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

        # --- 연결 진단(빠름) ---
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
                        sa_norm = _normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON)
                        creds = _validate_sa(sa_norm)
                        _log("service_account: valid ✅")
                    except Exception as se:
                        _log_exception("service_account invalid ❌", se)
                    folder_id = getattr(settings, "BACKUP_FOLDER_ID", None) or getattr(settings, "GDRIVE_FOLDER_ID", None)
                    _log_kv("folder_id", str(folder_id or "(empty)"))
                    if not folder_id:
                        _log("folder_id 비어있음 ❌ — secrets.toml 확인 필요")
                    if not os.path.exists(PERSIST_DIR) and folder_id:
                        try:
                            ok = try_restore_index_from_drive(creds, PERSIST_DIR, folder_id)
                            _log_kv("drive_restore", "success ✅" if ok else "not found/failed ❌")
                        except Exception as de:
                            _log_exception("drive_restore error", de)
                    st.success("진단 완료. 우측 로그/Traceback 확인하세요.")
                except Exception as e:
                    _log_exception("연결 진단 자체 실패", e)
                    st.error("연결 진단 중 오류. 우측 로그/Traceback 확인.")
        st.stop()

    # [08-2] 채팅 UI
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    st.markdown("---")

    mode_label = st.radio(
        "**어떤 도움이 필요한가요?**",
        ["💬 이유문법 설명","🔎 구문 분석","📚 독해 및 요약"],
        horizontal=True, key="mode_select"
    )

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
        st.session_state.messages.append({"role":"assistant","content":answer})
        st.rerun()
    except Exception as e:
        _log_exception("답변 생성 실패", e)
        st.error("답변 생성 중 오류. 우측 로그/Traceback 확인.")

# ===== [09] END OF FILE ======================================================
