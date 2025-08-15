# app.py — 스텝 인덱싱(중간 취소/재개) + 두 LLM 준비 + 그룹토론 UI
#          + Google Drive 대화 로그(.jsonl / Markdown) 저장

# ===== Imports =====
import os
import time
import uuid
import re
import json
import pandas as pd
import streamlit as st

# Drive Markdown 로그 유틸
from src.drive_log import save_chatlog_markdown, get_chatlog_folder_id
# 기본 UI
from src.ui import load_css, render_header
# 설정
from src.config import settings

# RAG/Index & LLM
from src.rag_engine import (
    CancelledError,
    smoke_test_drive,
    preview_drive_files,
    set_embed_provider,
    make_llm,
    start_index_builder,   # ← 스텝 인덱서
    get_text_answer,
    llm_complete,
)

# ===== 환경 변수 설정: 런타임 안정화 =====
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ===== Streamlit 페이지 설정 (첫 호출만 허용) =====
st.set_page_config(
    page_title="나의 AI 영어 교사",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===== 세션 상태 =====
ss = st.session_state
ss.setdefault("session_id", uuid.uuid4().hex[:12])   # 대화 세션 ID
ss.setdefault("messages", [])                        # 채팅 메시지 렌더용
ss.setdefault("auto_save_chatlog", True)             # Markdown 자동 저장
ss.setdefault("save_logs", True)                     # JSONL 저장 ON/OFF
ss.setdefault("prep_both_running", False)
ss.setdefault("prep_both_done", ("qe_google" in ss) or ("qe_openai" in ss))
ss.setdefault("prep_cancel_requested", False)
ss.setdefault("p_shared", 0)                         # 진행 퍼센트
ss.setdefault("files_per_step", 6)                   # 스텝당 처리 파일 수
ss.setdefault("index_builder", None)                 # 스텝 인덱서 객체

# ===== 기본 UI/스타일 =====
load_css()
render_header()
st.info("✅ 인덱싱은 스텝 방식으로 수행되어 **중간 취소/재개**가 가능합니다. (Resume·chat_log 제외·증분 저장)")

# ===== 사이드바 =====
with st.sidebar:
    ss.auto_save_chatlog = st.toggle("대화 자동 저장(Drive)", value=ss.auto_save_chatlog)

# ===== Google Drive 연결 테스트 =====
st.markdown("## 🔗 Google Drive 연결 테스트")
st.caption("서비스계정에 폴더 ‘쓰기(Writer)’ 권한이 있어야 대화 로그 저장이 됩니다.")

col1, col2 = st.columns([0.65, 0.35])
with col1:
    if st.button("폴더 파일 미리보기 (최신 10개)", use_container_width=True, disabled=ss.prep_both_running):
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
    st.success(msg) if ok else st.warning(msg)

# ===== (선택) 인덱싱 보고서 표시 =====
rep = st.session_state.get("indexing_report")
if rep:
    with st.expander("🧾 인덱싱 보고서(스킵된 파일 보기)", expanded=False):
        st.write(f"총 파일(매니페스트): {rep.get('total_manifest')}, "
                 f"로딩된 문서 수: {rep.get('loaded_docs')}, "
                 f"스킵: {rep.get('skipped_count')}")
        skipped = rep.get("skipped", [])
        if skipped:
            st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)
        else:
            st.caption("스킵된 파일이 없습니다 🎉")

# ▶ 대화 로그 저장 모듈(JSONL)
from src import chat_store

# ▶ 학습 모드(페르소나)
from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
mode = st.radio(
    "학습 모드", ["💬 이유문법 설명", "🔎 구문 분석", "📚 독해 및 요약"],
    horizontal=True, key="mode_select"
)
def _persona():
    return EXPLAINER_PROMPT if mode == "💬 이유문법 설명" else (ANALYST_PROMPT if mode == "🔎 구문 분석" else READER_PROMPT)

# ▶ 로그 저장 대상 폴더
CHAT_FOLDER_ID = getattr(settings, "CHATLOG_FOLDER_ID", None) or settings.GDRIVE_FOLDER_ID

# ▶ ‘세션 종료’ 버튼
with st.container():
    _, coly = st.columns([0.72, 0.28])
    with coly:
        if st.button("⏹ 세션 종료", use_container_width=True, type="secondary"):
            ss.session_terminated = True
            ss.prep_both_running = False
            ss.prep_cancel_requested = False
            ss.index_builder = None
            st.warning("세션이 종료되었습니다. 새로고침(Ctrl/⌘+Shift+R)으로 다시 시작하세요.")
            st.stop()

# ===== 진행률 UI =====
st.markdown("----"); st.subheader("🧠 두뇌 준비 — 스텝 인덱싱 + Gemini/ChatGPT")
st.markdown("### 🚀 인덱싱 + 두 LLM 준비")
c_g, c_o = st.columns(2)
with c_g: st.caption("Gemini 진행"); g_bar = st.empty(); g_msg = st.empty()
with c_o: st.caption("ChatGPT 진행"); o_bar = st.empty(); o_msg = st.empty()

def _render_progress(slot_bar, slot_msg, pct: int, msg: str | None = None):
    p = max(0, min(100, int(pct)))
    slot_bar.markdown(
        f"<div class='gp-wrap'><div class='gp-fill' style='width:{p}%'></div>"
        f"<div class='gp-label'>{p}%</div></div>", unsafe_allow_html=True
    )
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
    ss.files_per_step = st.number_input("스텝당 처리 파일 수", min_value=1, max_value=50, value=ss.files_per_step, step=1,
                                        disabled=ss.prep_both_running or ss.prep_both_done)
    ss.save_logs = st.checkbox("💾 대화 로그를 Google Drive에 저장하기", value=ss.save_logs,
                               help="Writer 권한 필요. 일자별 chat_log_YYYY-MM-DD.jsonl 로 저장됩니다.",
                               disabled=False)

# ===== 스텝 인덱싱 러너 =====
def _make_embed_provider():
    """임베딩 공급자/모델 선택 및 Settings 셋업"""
    embed_provider = "openai"
    embed_api = getattr(settings, "OPENAI_API_KEY", None)
    embed_api = embed_api.get_secret_value() if embed_api else ""
    embed_model = getattr(settings, "OPENAI_EMBED_MODEL", "text-embedding-3-small")
    if not embed_api:
        embed_provider = "google"
        embed_api = settings.GEMINI_API_KEY.get_secret_value()
        embed_model = getattr(settings, "EMBED_MODEL", "text-embedding-004")
    set_embed_provider(embed_provider, embed_api, embed_model)
    return embed_provider

def _normalize_sa(v):
    if isinstance(v, str):
        try: return json.loads(v)
        except Exception: return {}
    return v

def _step_on_pct(pct: int, msg: str | None = None):
    ss.p_shared = _bump_max("p_shared", pct)
    _render_progress(g_bar, g_msg, ss.p_shared, msg)
    _render_progress(o_bar, o_msg, ss.p_shared, msg)

def _step_on_msg(m: str):
    _render_progress(g_bar, g_msg, ss.p_shared, m)
    _render_progress(o_bar, o_msg, ss.p_shared, m)

def _tick_prepare_both_step():
    """한 번에 조금씩 처리하고, 끝나면 LLM들을 준비한다."""
    try:
        # 0) 준비: 임베딩 세팅 & IndexBuilder 생성
        if ss.index_builder is None:
            _render_progress(g_bar, g_msg, ss.p_shared, "대기 중…")
            _render_progress(o_bar, o_msg, ss.p_shared, "대기 중…")

            provider = _make_embed_provider()
            _step_on_pct(5, f"임베딩 설정({provider})")

            sa = _normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON)
            persist_dir = f"{getattr(settings, 'PERSIST_DIR', '/tmp/my_ai_teacher/storage_gdrive')}_shared"
            ss.index_builder = start_index_builder(
                gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
                gcp_creds=sa,
                persist_dir=persist_dir,
                exclude_folder_names=["chat_log"],
                max_docs=(max_docs if fast else None),
            )
            _step_on_msg("Resume/증분 준비 완료")

        # 1) 스텝 실행
        status = ss.index_builder.step(
            max_files=int(ss.files_per_step),
            per_file_timeout_s=40,
            is_cancelled=lambda: bool(ss.get("prep_cancel_requested", False)),
            on_pct=_step_on_pct,
            on_msg=_step_on_msg,
        )

        if status == "running":
            # 다음 스텝으로 바로 진행
            time.sleep(0.05)
            st.rerun()
            return

        # 2) 인덱싱 완료 → LLM 준비
        _step_on_pct(92, "인덱스 저장 중…")
        # ss.index_builder 내부에서 persist 완료됨

        # 인덱싱 보고서 저장
        b = ss.index_builder
        st.session_state["indexing_report"] = {
            "total_manifest": b.total,
            "loaded_docs": b.processed,
            "skipped_count": len(b.skipped),
            "skipped": b.skipped,
        }

        # LLM 두 개 준비
        try:
            g_llm = make_llm(
                "google",
                settings.GEMINI_API_KEY.get_secret_value(),
                getattr(settings, "LLM_MODEL", "gemini-1.5-pro"),
                float(ss.get("temperature", 0.0)),
            )
            ss["llm_google"] = g_llm
            ss["qe_google"] = b.index.as_query_engine(
                llm=g_llm,
                response_mode=ss.get("response_mode", getattr(settings, "RESPONSE_MODE", "compact")),
                similarity_top_k=int(ss.get("similarity_top_k", getattr(settings, "SIMILARITY_TOP_K", 5))),
            )
            _render_progress(g_bar, g_msg, 100, "완료!")
        except Exception as e:
            _render_progress(g_bar, g_msg, 100, f"Gemini 준비 실패: {e}")

        try:
            if getattr(settings, "OPENAI_API_KEY", None) and settings.OPENAI_API_KEY.get_secret_value():
                o_llm = make_llm(
                    "openai",
                    settings.OPENAI_API_KEY.get_secret_value(),
                    getattr(settings, "OPENAI_LLM_MODEL", "gpt-4o-mini"),
                    float(ss.get("temperature", 0.0)),
                )
                ss["llm_openai"] = o_llm
                ss["qe_openai"] = b.index.as_query_engine(
                    llm=o_llm,
                    response_mode=ss.get("response_mode", getattr(settings, "RESPONSE_MODE", "compact")),
                    similarity_top_k=int(ss.get("similarity_top_k", getattr(settings, "SIMILARITY_TOP_K", 5))),
                )
                _render_progress(o_bar, o_msg, 100, "완료!")
            else:
                _render_progress(o_bar, o_msg, 100, "키 누락 — OPENAI_API_KEY 필요")
        except Exception as e:
            _render_progress(o_bar, o_msg, 100, f"ChatGPT 준비 실패: {e}")

        ss.prep_both_running = False
        ss.prep_both_done = True
        ss.prep_cancel_requested = False
        ss.index_builder = None
        time.sleep(0.2)
        st.rerun()

    except CancelledError:
        ss.prep_both_running = False
        ss.prep_cancel_requested = False
        ss.index_builder = None
        _step_on_msg("사용자 취소")
        st.stop()
    except Exception as e:
        ss.prep_both_running = False
        ss.index_builder = None
        _step_on_msg(f"오류: {e}")
        st.stop()

# ▶ 실행/취소 버튼
left, right = st.columns([0.7, 0.3])
with left:
    clicked = st.button("🚀 준비 시작(스텝)", key="prepare_step", use_container_width=True,
                        disabled=ss.prep_both_running or ss.prep_both_done)
with right:
    cancel_clicked = st.button("⛔ 준비 취소", key="cancel_prepare", use_container_width=True, type="secondary",
                               disabled=not ss.prep_both_running)

if cancel_clicked and ss.prep_both_running:
    ss.prep_cancel_requested = True
    st.rerun()

if clicked and not (ss.prep_both_running or ss.prep_both_done):
    ss.p_shared = 0
    ss.prep_cancel_requested = False
    ss.prep_both_running = True
    ss.index_builder = None
    st.rerun()

if ss.prep_both_running:
    _tick_prepare_both_step()

st.caption("준비 버튼을 다시 활성화하려면 아래 재설정 버튼을 누르세요.")
if st.button("🔧 재설정(버튼 다시 활성화)", disabled=not ss.prep_both_done):
    ss.prep_both_done = False
    ss.p_shared = 0
    st.rerun()

# ===== 대화 UI — 항상 👥 그룹토론 + 로그 저장 =====
st.markdown("---")
st.subheader("💬 그룹토론 대화 (사용자 → Gemini 1차 → ChatGPT 보완/검증)")

ready_google = "qe_google" in ss
ready_openai = "qe_openai" in ss
if ss.get("session_terminated"):
    st.warning("세션이 종료된 상태입니다. 새로고침으로 다시 시작하세요.")
    st.stop()
if not ready_google:
    st.info("먼저 **[🚀 준비 시작(스텝)]** 을 클릭해 두뇌를 준비하세요. (OpenAI 키가 없으면 Gemini만 응답)")
    st.stop()

# 이미 쌓인 메시지 렌더
for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def _strip_sources(text: str) -> str:
    # 하단 참고자료 블록 제거
    return re.sub(r"\n+---\n\*참고 자료:.*$", "", text, flags=re.DOTALL)

# 최근 대화 맥락 구성기
def _build_context_for_models(messages: list[dict], limit_pairs: int = 2, max_chars: int = 2000) -> str:
    pairs = []
    buf_user = None
    for m in reversed(messages):
        role, content = m.get("role"), str(m.get("content", "")).strip()
        if role == "assistant":
            content = re.sub(r"^\*\*🤖 .*?\*\*\s*\n+", "", content).strip()
            if buf_user is not None:
                pairs.append((buf_user, content))
                buf_user = None
                if len(pairs) >= limit_pairs:
                    break
        elif role == "user":
            if buf_user is None:
                buf_user = content
    pairs = list(reversed(pairs))
    blocks = [f"[학생]\n{u}\n\n[교사]\n{a}" for u, a in pairs]
    ctx = "\n\n---\n\n".join(blocks).strip()
    if len(ctx) > max_chars:
        ctx = ctx[-max_chars:]
    return ctx

# 공통: JSONL 로그 저장(실패해도 앱 중단 X) — 항상 chat_log/ 서브폴더에 저장
def _log_try(items):
    if not ss.save_logs: return
    try:
        parent_id = (getattr(settings, "CHATLOG_FOLDER_ID", None) or settings.GDRIVE_FOLDER_ID)
        # SA JSON 정규화
        sa = settings.GDRIVE_SERVICE_ACCOUNT_JSON
        if isinstance(sa, str):
            try: sa = json.loads(sa)
            except Exception: pass
        # chat_log 서브폴더 ID 보장
        sub_id = get_chatlog_folder_id(parent_folder_id=parent_id, sa_json=sa)
        # JSONL 저장
        chat_store.append_jsonl(folder_id=sub_id, sa_json=sa, items=items)
        st.toast("대화 로그 저장 완료", icon="💾")
    except Exception as e:
        st.caption(f"⚠️ 대화 로그 저장 실패: {e}")

# ===== 입력창 & 처리 =====
user_input = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
if user_input:
    # 0) 사용자 메시지
    ss.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    # JSONL 로그: 사용자
    _log_try([chat_store.make_entry(ss.session_id, "user", "user", user_input, mode, model="user")])

    # 1) Gemini 1차 (최근 맥락 + 현재 질문)
    with st.spinner("🤖 Gemini 선생님이 먼저 답변합니다…"):
        prev_ctx = _build_context_for_models(ss.messages[:-1], limit_pairs=2, max_chars=2000)
        gemini_query = (f"[이전 대화]\n{prev_ctx}\n\n" if prev_ctx else "") + f"[학생 질문]\n{user_input}"
        ans_g = get_text_answer(ss["qe_google"], gemini_query, _persona())
    content_g = f"**🤖 Gemini**\n\n{ans_g}"
    ss.messages.append({"role": "assistant", "content": content_g})
    with st.chat_message("assistant"): st.markdown(content_g)

    # JSONL 로그: Gemini
    _log_try([chat_store.make_entry(
        ss.session_id, "assistant", "Gemini", content_g, mode,
        model=getattr(settings, "LLM_MODEL", "gemini")
    )])

    # 2) ChatGPT 보완/검증 (RAG 없이, Gemini 답변을 읽고 보완)
    if ready_openai:
        review_directive = (
            "역할: 당신은 동료 AI 영어교사입니다.\n"
            "목표: [이전 대화], [학생 질문], [동료의 1차 답변]을 읽고, 사실오류/빠진점/모호함을 교정·보완합니다.\n"
            "지침:\n"
            "1) 핵심만 간결히 재정리\n"
            "2) 틀린 부분은 근거와 함께 바로잡기\n"
            "3) 이해를 돕는 예문 2~3개 추가 (가능하면 학습자의 모국어 대비 포인트)\n"
            "4) 마지막에 <최종 정리> 섹션으로 한눈 요약\n"
            "금지: 새로운 외부 검색/RAG. 제공된 내용과 교사 지식만 사용.\n"
        )
        prev_ctx2 = _build_context_for_models(ss.messages, limit_pairs=2, max_chars=2000)
        augmented = (
            (f"[이전 대화]\n{prev_ctx2}\n\n" if prev_ctx2 else "") +
            f"[학생 질문]\n{user_input}\n\n"
            f"[동료의 1차 답변(Gemini)]\n{_strip_sources(ans_g)}\n\n"
            f"[당신의 작업]\n위 기준으로만 보완/검증하라."
        )
        with st.spinner("🤝 ChatGPT 선생님이 보완/검증 중…"):
            ans_o = llm_complete(ss.get("llm_openai"),
                                 _persona() + "\n\n" + review_directive + "\n\n" + augmented)
        content_o = f"**🤖 ChatGPT**\n\n{ans_o}"
        ss.messages.append({"role": "assistant", "content": content_o})
        with st.chat_message("assistant"): st.markdown(content_o)
    else:
        with st.chat_message("assistant"):
            st.info("ChatGPT 키가 없어 Gemini만 응답했습니다. OPENAI_API_KEY를 추가하면 보완/검증이 활성화됩니다.")

    # ✅ Drive Markdown 대화 로그 자동 저장 (공유드라이브 데이터 폴더 내 chat_log/)
    if ss.auto_save_chatlog and ss.messages:
        try:
            parent_id = (getattr(settings, "CHATLOG_FOLDER_ID", None) or settings.GDRIVE_FOLDER_ID)
            sa = settings.GDRIVE_SERVICE_ACCOUNT_JSON
            if isinstance(sa, str):
                try: sa = json.loads(sa)
                except Exception: pass
            save_chatlog_markdown(
                ss.session_id, ss.messages,
                parent_folder_id=parent_id, sa_json=sa
            )
            st.toast("Drive에 대화 저장 완료 (chat_log/)", icon="💾")
        except Exception as e:
            st.caption(f"⚠️ Drive 저장 실패: {e}")

    st.rerun()
