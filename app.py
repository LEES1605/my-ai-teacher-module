# app.py — 그룹토론(항상 ON) + 인덱싱 재개(Resume) + 준비 취소/종료
#          + Google Drive 대화 로그(.jsonl) 저장

import streamlit as st
import pandas as pd
import time, re, uuid
from datetime import datetime

# ===== 페이지 설정 ============================================================
st.set_page_config(page_title="나의 AI 영어 교사", layout="wide", initial_sidebar_state="collapsed")

# ===== 기본 UI/스타일 =========================================================
from src.ui import load_css, render_header
load_css(); render_header()

st.info("✅ 인덱싱은 1번만 수행하고, 그 인덱스로 Gemini/ChatGPT 두 LLM을 준비합니다. (빠른 모드·되돌림 방지·Resume·항상 👥 그룹토론)")

# ===== Google Drive 연결 테스트 ===============================================
try:
    from src.rag_engine import smoke_test_drive, preview_drive_files
except Exception:
    st.error("`src.rag_engine` 임포트 실패")
    import traceback, os
    st.write("파일 존재 여부:", os.path.exists("src/rag_engine.py"))
    with st.expander("임포트 스택", expanded=True): st.code(traceback.format_exc())
    st.stop()

st.markdown("## 🔗 Google Drive 연결 테스트")
st.caption("서비스계정에 폴더 ‘쓰기(Writer)’ 권한이 있어야 대화 로그 저장이 됩니다.")

col1, col2 = st.columns([0.65, 0.35])
with col1:
    if st.button("폴더 파일 미리보기 (최신 10개)", use_container_width=True):
        ok, msg, rows = preview_drive_files(max_items=10)
        if ok and rows:
            df = pd.DataFrame(rows)
            df["type"] = df["mime"].str.replace("application/vnd.google-apps.", "", regex=False)
            df = df.rename(columns={"modified": "modified_at"})
            df = df[["name", "link", "type", "modified_at"]]
            st.dataframe(
                df, use_container_width=True, height=360,
                column_config={
                    "name": st.column_config.TextColumn("파일명"),
                    "link": st.column_config.LinkColumn("open", display_text="열기"),
                    "type": st.column_config.TextColumn("유형"),
                    "modified_at": st.column_config.TextColumn("수정시각"),
                }, hide_index=True
            )
        elif ok:
            st.warning("폴더에 파일이 없거나 접근할 수 없습니다.")
        else:
            st.error(msg)
with col2:
    ok, msg = smoke_test_drive()
    if ok: st.success(msg)
    else:  st.warning(msg)

# ===== 두뇌 준비 (공통 인덱스 + LLM 2개) ======================================
st.markdown("----"); st.subheader("🧠 두뇌 준비 — 인덱스 1회 + Gemini/ChatGPT")

from src.config import settings
try:
    from src.rag_engine import (
        set_embed_provider, make_llm, get_or_build_index, get_text_answer, CancelledError
    )
except Exception:
    st.error("`src.rag_engine` 임포트(LLM/RAG) 실패")
    import traceback
    with st.expander("임포트 스택", expanded=True): st.code(traceback.format_exc())
    st.stop()

# ▶ 대화 로그 저장 모듈
from src import chat_store

# ▶ 세션 기본값
ss = st.session_state
ss.setdefault("prep_both_running", False)
ss.setdefault("prep_both_done", ("qe_google" in ss) or ("qe_openai" in ss))
ss.setdefault("p_shared", 0)
ss.setdefault("prep_cancel_requested", False)
ss.setdefault("session_terminated", False)
ss.setdefault("session_id", uuid.uuid4().hex[:12])  # 대화 세션 ID
ss.setdefault("save_logs", True)                    # 로그 저장 ON/OFF

# ▶ 학습 모드(페르소나)
from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
mode = st.radio("학습 모드", ["💬 이유문법 설명", "🔎 구문 분석", "📚 독해 및 요약"],
                horizontal=True, key="mode_select")
def _persona():
    return EXPLAINER_PROMPT if mode == "💬 이유문법 설명" else (ANALYST_PROMPT if mode == "🔎 구문 분석" else READER_PROMPT)

# ▶ 로그 저장 대상 폴더
CHAT_FOLDER_ID = getattr(settings, "CHATLOG_FOLDER_ID", None) or settings.GDRIVE_FOLDER_ID

# ▶ ‘세션 종료’ 버튼
with st.container():
    colx, coly = st.columns([0.72, 0.28])
    with coly:
        if st.button("⏹ 세션 종료", use_container_width=True, type="secondary"):
            ss.session_terminated = True
            ss.prep_both_running = False
            ss.prep_cancel_requested = False
            st.warning("세션이 종료되었습니다. 새로고침(Ctrl/⌘+Shift+R)으로 다시 시작하세요.")
            st.stop()

# ▶ 진행률 렌더 & 단조증가
def _render_progress(slot_bar, slot_msg, pct: int, msg: str | None = None):
    p = max(0, min(100, int(pct)))
    slot_bar.markdown(f"""
<div class="gp-wrap"><div class="gp-fill" style="width:{p}%"></div><div class="gp-label">{p}%</div></div>
""", unsafe_allow_html=True)
    if msg is not None:
        slot_msg.markdown(f"<div class='gp-msg'>{msg}</div>", unsafe_allow_html=True)
def _bump_max(key: str, pct: int) -> int:
    now = int(pct); prev = int(ss.get(key, 0))
    if now < prev: now = prev
    ss[key] = now; return now

# ▶ 옵션(빠른 모드)
with st.expander("⚙️ 옵션", expanded=False):
    fast = st.checkbox("⚡ 빠른 준비 (처음 N개 문서만 인덱싱)", value=True,
                       disabled=ss.prep_both_running or ss.prep_both_done)
    max_docs = st.number_input("N (빠른 모드일 때만)", min_value=5, max_value=500, value=40, step=5,
                               disabled=ss.prep_both_running or ss.prep_both_done)
    ss.save_logs = st.checkbox("💾 대화 로그를 Google Drive에 저장하기", value=ss.save_logs,
                               help="Writer 권한 필요. 일자별 chat_log_YYYY-MM-DD.jsonl 로 저장됩니다.",
                               disabled=False)

st.markdown("### 🚀 인덱싱 1번 + 두 LLM 준비")
c_g, c_o = st.columns(2)
with c_g: st.caption("Gemini 진행"); g_bar = st.empty(); g_msg = st.empty()
with c_o: st.caption("ChatGPT 진행"); o_bar = st.empty(); o_msg = st.empty()

def _is_cancelled() -> bool:
    return bool(ss.get("prep_cancel_requested", False))

def run_prepare_both():
    _render_progress(g_bar, g_msg, ss.p_shared, "대기 중…")
    _render_progress(o_bar, o_msg, ss.p_shared, "대기 중…")

    # 1) 임베딩 공급자
    embed_provider = "openai"
    embed_api = getattr(settings, "OPENAI_API_KEY", None).get_secret_value() if hasattr(settings, "OPENAI_API_KEY") else ""
    embed_model = getattr(settings, "OPENAI_EMBED_MODEL", "text-embedding-3-small")
    if not embed_api:
        embed_provider = "google"
        embed_api = settings.GEMINI_API_KEY.get_secret_value()
        embed_model = getattr(settings, "EMBED_MODEL", "text-embedding-004")
    persist_dir = f"{getattr(settings, 'PERSIST_DIR', '/tmp/my_ai_teacher/storage_gdrive')}_shared"

    # 2) 임베딩 설정
    try:
        if _is_cancelled(): raise CancelledError("사용자 취소")
        p = _bump_max("p_shared", 5)
        _render_progress(g_bar, g_msg, p, f"임베딩 설정({embed_provider})")
        _render_progress(o_bar, o_msg, p, f"임베딩 설정({embed_provider})")
        set_embed_provider(embed_provider, embed_api, embed_model)
    except CancelledError:
        ss.prep_both_running = False; ss.prep_cancel_requested = False
        _render_progress(g_bar, g_msg, ss.p_shared, "사용자 취소")
        _render_progress(o_bar, o_msg, ss.p_shared, "사용자 취소"); st.stop()
    except Exception as e:
        p = _bump_max("p_shared", 100)
        _render_progress(g_bar, g_msg, p, f"임베딩 실패: {e}")
        _render_progress(o_bar, o_msg, p, f"임베딩 실패: {e}")
        ss.prep_both_running = False; st.stop()

    # 3) 인덱스(Resume 지원)
    try:
        def upd(pct: int, msg: str | None = None):
            if _is_cancelled(): raise CancelledError("사용자 취소(진행 중)")
            p = _bump_max("p_shared", pct)
            _render_progress(g_bar, g_msg, p, msg); _render_progress(o_bar, o_msg, p, msg)
        def umsg(m: str):
            p = ss.p_shared
            _render_progress(g_bar, g_msg, p, m); _render_progress(o_bar, o_msg, p, m)

        index = get_or_build_index(
            update_pct=upd, update_msg=umsg,
            gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
            raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
            persist_dir=persist_dir,
            manifest_path=getattr(settings, "MANIFEST_PATH", "/tmp/my_ai_teacher/drive_manifest.json"),
            max_docs=(max_docs if fast else None),
            is_cancelled=_is_cancelled,   # 재개/취소 지원
        )
    except CancelledError:
        ss.prep_both_running = False; ss.prep_cancel_requested = False
        _render_progress(g_bar, g_msg, ss.p_shared, "사용자 취소"); _render_progress(o_bar, o_msg, ss.p_shared, "사용자 취소")
        st.stop()
    except Exception as e:
        p = _bump_max("p_shared", 100)
        _render_progress(g_bar, g_msg, p, f"인덱스 실패: {e}")
        _render_progress(o_bar, o_msg, p, f"인덱스 실패: {e}")
        ss.prep_both_running = False; st.stop()

    # 4) LLM 두 개 준비
    try:
        g_llm = make_llm("google", settings.GEMINI_API_KEY.get_secret_value(),
                         getattr(settings, "LLM_MODEL", "gemini-1.5-pro"),
                         float(ss.get("temperature", 0.0)))
        ss["qe_google"] = index.as_query_engine(
            llm=g_llm,
            response_mode=ss.get("response_mode", getattr(settings, "RESPONSE_MODE", "compact")),
            similarity_top_k=int(ss.get("similarity_top_k", getattr(settings, "SIMILARITY_TOP_K", 5))),
        )
        _render_progress(g_bar, g_msg, 100, "완료!")
    except Exception as e:
        _render_progress(g_bar, g_msg, 100, f"Gemini 준비 실패: {e}")

    try:
        if hasattr(settings, "OPENAI_API_KEY") and settings.OPENAI_API_KEY.get_secret_value():
            if _is_cancelled(): raise CancelledError("사용자 취소")
            o_llm = make_llm("openai", settings.OPENAI_API_KEY.get_secret_value(),
                             getattr(settings, "OPENAI_LLM_MODEL", "gpt-4o-mini"),
                             float(ss.get("temperature", 0.0)))
            ss["qe_openai"] = index.as_query_engine(
                llm=o_llm,
                response_mode=ss.get("response_mode", getattr(settings, "RESPONSE_MODE", "compact")),
                similarity_top_k=int(ss.get("similarity_top_k", getattr(settings, "SIMILARITY_TOP_K", 5))),
            )
            _render_progress(o_bar, o_msg, 100, "완료!")
        else:
            _render_progress(o_bar, o_msg, 100, "키 누락 — OPENAI_API_KEY 필요")
    except CancelledError:
        ss.prep_both_running = False; ss.prep_cancel_requested = False
        _render_progress(o_bar, o_msg, ss.p_shared, "사용자 취소"); st.stop()
    except Exception as e:
        _render_progress(o_bar, o_msg, 100, f"ChatGPT 준비 실패: {e}")

    ss.prep_both_running = False; ss.prep_both_done = True
    time.sleep(0.2); st.rerun()

# ▶ 실행/취소 버튼
left, right = st.columns([0.7, 0.3])
with left:
    clicked = st.button("🚀 한 번에 준비하기", key="prepare_both", use_container_width=True,
                        disabled=ss.prep_both_running or ss.prep_both_done)
with right:
    cancel_clicked = st.button("⛔ 준비 취소", key="cancel_prepare", use_container_width=True, type="secondary",
                               disabled=not ss.prep_both_running)

if cancel_clicked and ss.prep_both_running:
    ss.prep_cancel_requested = True; st.rerun()

if clicked and not (ss.prep_both_running or ss.prep_both_done):
    ss.p_shared = 0; ss.prep_cancel_requested = False; ss.prep_both_running = True; st.rerun()

if ss.prep_both_running:
    run_prepare_both()

st.caption("준비 버튼을 다시 활성화하려면 아래 재설정 버튼을 누르세요.")
if st.button("🔧 재설정(버튼 다시 활성화)", disabled=not ss.prep_both_done):
    ss.prep_both_done = False
    ss.p_shared = 0
    st.rerun()  # ← 최신 Streamlit

# ===== 대화 UI — 항상 👥 그룹토론 + 로그 저장 ==================================
st.markdown("---"); st.subheader("💬 그룹토론 대화 (사용자 → Gemini 1차 → ChatGPT 보완/검증)")

ready_google = "qe_google" in ss
ready_openai = "qe_openai" in ss
if ss.session_terminated:
    st.warning("세션이 종료된 상태입니다. 새로고침으로 다시 시작하세요."); st.stop()
if not ready_google:
    st.info("먼저 **[🚀 한 번에 준비하기]** 를 클릭해 두뇌를 준비하세요. (OpenAI 키가 없으면 Gemini만 응답)"); st.stop()

# 대화 기록
ss.setdefault("messages", [])
for m in ss.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

def _strip_sources(text: str) -> str:
    return re.sub(r"\n+---\n\*참고 자료:.*$", "", text, flags=re.DOTALL)

# 공통: 로그 저장 함수(실패해도 앱 중단 X)
def _log_try(items):
    if not ss.save_logs: return
    try:
        chat_store.append_jsonl(
            folder_id=CHAT_FOLDER_ID,
            sa_json=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
            items=items
        )
        st.toast("대화 로그 저장 완료", icon="💾")
    except Exception as e:
        st.caption(f"⚠️ 대화 로그 저장 실패: {e}")

# 입력창
user_input = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
if user_input:
    # 0) 사용자 메시지
    ss.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    # 로그: 사용자 발화
    _log_try([chat_store.make_entry(ss.session_id, "user", "user", user_input, mode, model="user")])

    # 1) Gemini 1차
    with st.spinner("🤖 Gemini 선생님이 먼저 답변합니다…"):
        ans_g = get_text_answer(ss["qe_google"], user_input, _persona())
    content_g = f"**🤖 Gemini**\n\n{ans_g}"
    ss.messages.append({"role": "assistant", "content": content_g})
    with st.chat_message("assistant"): st.markdown(content_g)

    # 로그: Gemini 답변
    _log_try([chat_store.make_entry(ss.session_id, "assistant", "Gemini", content_g, mode, model=getattr(settings, "LLM_MODEL", "gemini"))])

    # 2) ChatGPT 보완/검증 (있을 때)
    if ready_openai:
        review_directive = (
            "당신은 동료 AI 교사입니다. 아래 [학생 질문]과 [동료의 1차 답변]을 읽고 "
            "부족한 부분을 보완/교정하고 예시를 추가한 뒤, 마지막에 '최종 정리'를 제시하세요. "
            "가능하면 근거(자료 파일명)를 유지하거나 보강하세요."
        )
        augmented = f"[학생 질문]\n{user_input}\n\n[동료의 1차 답변(Gemini)]\n{_strip_sources(ans_g)}"
        with st.spinner("🤝 ChatGPT 선생님이 보완/검증 중…"):
            ans_o = get_text_answer(ss["qe_openai"], augmented, _persona() + "\n" + review_directive)
        content_o = f"**🤖 ChatGPT**\n\n{ans_o}"
        ss.messages.append({"role": "assistant", "content": content_o})
        with st.chat_message("assistant"): st.markdown(content_o)

        # 로그: ChatGPT 답변
        _log_try([chat_store.make_entry(ss.session_id, "assistant", "ChatGPT", content_o, mode,
                                        model=getattr(settings, "OPENAI_LLM_MODEL", "gpt-4o-mini"))])
    else:
        with st.chat_message("assistant"):
            st.info("ChatGPT 키가 없어 Gemini만 응답했습니다. OPENAI_API_KEY를 추가하면 보완/검증이 활성화됩니다.")

    st.rerun()
