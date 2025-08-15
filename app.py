# app.py — 스텝 인덱싱(중간취소/재개) + 두뇌준비 안정화
#        + 인덱싱 보고서(스킵 파일 표시) + Drive 대화로그(JSONL/Markdown, chat_log/ 저장)
#        + 대화 페르소나(친절한 Gemini, 유머러스한 ChatGPT)

# ===== Imports =====
import os
import time
import uuid
import re
import json
import pandas as pd
import streamlit as st

# UI
from src.ui import load_css, render_header

# Drive 로그 유틸
from src.drive_log import save_chatlog_markdown_oauth

if ss.auto_save_chatlog and ss.messages:
    try:
        if is_signed_in():
            svc = build_drive_service()
            parent_id = (st.secrets.get("OAUTH_CHAT_PARENT_ID") or "").strip() or None
            _fid = save_chatlog_markdown_oauth(ss.session_id, ss.messages, svc, parent_id)
            st.toast("내 드라이브에 대화 저장 완료 ✅", icon="💾")
        else:
            st.info("구글 계정으로 로그인하면 대화가 **내 드라이브**에 저장됩니다.")
    except Exception as e:
        st.warning(f"OAuth 저장 실패: {e}")


# 설정
from src.config import settings

# RAG/인덱싱 유틸 (스텝 빌더 사용)
from src.rag_engine import (
    set_embed_provider, make_llm, get_text_answer, CancelledError,
    start_index_builder, resume_index_builder, cancel_index_builder, get_index_progress
)

# JSONL 로그 스토어
from src import chat_store

# 페르소나 프롬프트
from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT


# ===== 런타임 안정화 환경변수 =====
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ===== 페이지 설정 (첫 호출만) =====
st.set_page_config(
    page_title="나의 AI 영어 교사",
    layout="wide",
    initial_sidebar_state="collapsed",
)
from src.google_oauth import finish_oauth_if_redirected
finish_oauth_if_redirected()

# ===== 세션 상태 =====
ss = st.session_state
ss.setdefault("session_id", uuid.uuid4().hex[:12])      # 대화 세션 ID
ss.setdefault("messages", [])                           # {"role": "user"|"assistant", "content": str}
ss.setdefault("auto_save_chatlog", True)                # Markdown 자동 저장
ss.setdefault("save_logs", True)                        # JSONL 저장 ON/OFF
ss.setdefault("prep_both_running", False)
ss.setdefault("prep_both_done", ("qe_google" in ss) or ("qe_openai" in ss))
ss.setdefault("p_shared", 0)
ss.setdefault("prep_cancel_requested", False)
ss.setdefault("session_terminated", False)
ss.setdefault("index_job", None)                        # 스텝 빌더 상태


# ===== 기본 UI / 헤더 =====
load_css()
render_header()
st.info("✅ 인덱싱은 변경이 있을 때만 다시 수행합니다. 저장된 두뇌가 있으면 즉시 불러옵니다. (중간 취소/재개 가능)")

# ===== 사이드바 =====
with st.sidebar:
    ss.auto_save_chatlog = st.toggle("대화 자동 저장(Drive/Markdown)", value=ss.auto_save_chatlog)
    ss.save_logs = st.toggle("대화 JSONL 저장(Drive/chat_log/)", value=ss.save_logs)

from src.google_oauth import start_oauth, is_signed_in, build_drive_service, get_user_email, sign_out

with st.sidebar:
    ss.auto_save_chatlog = st.toggle("대화 자동 저장(Drive)", value=ss.auto_save_chatlog)

    st.markdown("---")
    st.markdown("### Google 로그인 (내 드라이브 저장)")

    if not is_signed_in():
        if st.button("🔐 Google로 로그인"):
            url = start_oauth()
            st.markdown(f"[여기를 눌러 로그인하세요]({url})")
    else:
        st.success(f"로그인됨: {get_user_email() or '알 수 없음'}")
        if st.button("로그아웃"):
            sign_out()
            st.experimental_rerun()


# ===== Google Drive 연결 테스트 =====
st.markdown("## 🔗 Google Drive 연결 테스트")
st.caption("서비스계정에 **공유 드라이브(Shared Drive)**의 폴더에 대해 Writer 권한이 있어야 저장이 됩니다.")

try:
    from src.rag_engine import smoke_test_drive, preview_drive_files
except Exception:
    st.error("`src.rag_engine` 임포트 실패")
    import traceback, os as _os
    st.write("파일 존재 여부:", _os.path.exists("src/rag_engine.py"))
    with st.expander("임포트 스택", expanded=True):
        st.code(traceback.format_exc())
    st.stop()

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

# ===== 인덱싱 보고서(스킵 파일 확인) =====
rep = st.session_state.get("indexing_report")
if rep:
    with st.expander("🧾 인덱싱 보고서 (스킵된 파일 보기)", expanded=False):
        st.write(
            f"총 파일(매니페스트): {rep.get('total_manifest')}, "
            f"로딩된 문서 수: {rep.get('loaded_docs')}, "
            f"스킵: {rep.get('skipped_count')}"
        )
        skipped = rep.get("skipped", [])
        if skipped:
            st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)
        else:
            st.caption("스킵된 파일이 없습니다 🎉")

# ===== 두뇌 준비(스텝) =====
st.markdown("---")
st.subheader("🧠 두뇌 준비 — 저장본 로드 ↔ 변경 시 증분 인덱싱 (중간 취소/재개)")

# 옵션
with st.expander("⚙️ 옵션", expanded=False):
    ss.fast = st.checkbox(
        "⚡ 빠른 준비 (처음 N개 문서만 인덱싱)", value=ss.get("fast", True),
        disabled=ss.prep_both_running or ss.prep_both_done
    )
    ss.max_docs = st.number_input(
        "N (빠른 모드일 때만)", min_value=5, max_value=500, value=int(ss.get("max_docs", 40)), step=5,
        disabled=ss.prep_both_running or ss.prep_both_done
    )

# 진행률 바
c_g, c_o = st.columns(2)
with c_g:
    st.caption("Gemini 진행")
    g_bar = st.empty(); g_msg = st.empty()
with c_o:
    st.caption("ChatGPT 진행")
    o_bar = st.empty(); o_msg = st.empty()

def _render_progress(slot_bar, slot_msg, pct: int, msg: str | None = None):
    p = max(0, min(100, int(pct)))
    slot_bar.markdown(
        f"""
<div class="gp-wrap"><div class="gp-fill" style="width:{p}%"></div><div class="gp-label">{p}%</div></div>
""", unsafe_allow_html=True
    )
    if msg is not None:
        slot_msg.markdown(f"<div class='gp-msg'>{msg}</div>", unsafe_allow_html=True)

def _is_cancelled() -> bool:
    return bool(ss.get("prep_cancel_requested", False))

# 스텝 러너
def run_prepare_both_step():
    # 1) 임베딩 공급자 선택/설정
    embed_provider = "openai"
    embed_api = (getattr(settings, "OPENAI_API_KEY", None).get_secret_value()
                 if getattr(settings, "OPENAI_API_KEY", None) else "")
    embed_model = getattr(settings, "OPENAI_EMBED_MODEL", "text-embedding-3-small")
    if not embed_api:
        embed_provider = "google"
        embed_api = settings.GEMINI_API_KEY.get_secret_value()
        embed_model = getattr(settings, "EMBED_MODEL", "text-embedding-004")

    try:
        _render_progress(g_bar, g_msg, 3, f"임베딩 설정({embed_provider})")
        _render_progress(o_bar, o_msg, 3, f"임베딩 설정({embed_provider})")
        set_embed_provider(embed_provider, embed_api, embed_model)
    except Exception as e:
        _render_progress(g_bar, g_msg, 100, f"임베딩 실패: {e}")
        _render_progress(o_bar, o_msg, 100, f"임베딩 실패: {e}")
        ss.prep_both_running = False
        return

    # 2) 인덱스 스텝 진행
    def upd(p, m=None):
        _render_progress(g_bar, g_msg, p, m)
        _render_progress(o_bar, o_msg, p, m)

    def umsg(m):
        # 메시지는 현재 퍼센트 유지
        _render_progress(g_bar, g_msg, ss.get("p_shared", 0), m)
        _render_progress(o_bar, o_msg, ss.get("p_shared", 0), m)

    job = ss.get("index_job")
    persist_dir = f"{getattr(settings,'PERSIST_DIR','/tmp/my_ai_teacher/storage_gdrive')}_shared"

    if job is None:
        # 처음 시작
        res = start_index_builder(
            update_pct=upd, update_msg=umsg,
            gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
            raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
            persist_dir=persist_dir,
            manifest_path=getattr(settings, "MANIFEST_PATH", "/tmp/my_ai_teacher/drive_manifest.json"),
            max_docs=(int(ss.max_docs) if ss.fast else None),
            is_cancelled=_is_cancelled,
        )
        status = res.get("status")
        if status == "done":
            index = res["index"]
        elif status == "running":
            ss.index_job = res["job"]
            _render_progress(g_bar, g_msg, res.get("pct", 8), res.get("msg", "진행 중…"))
            _render_progress(o_bar, o_msg, res.get("pct", 8), res.get("msg", "진행 중…"))
            time.sleep(0.2); st.rerun()
            return
        else:
            _render_progress(g_bar, g_msg, 100, "인덱싱 시작 실패")
            _render_progress(o_bar, o_msg, 100, "인덱싱 시작 실패")
            ss.prep_both_running = False
            return
    else:
        # 재개
        res = resume_index_builder(
            job=job, update_pct=upd, update_msg=umsg,
            is_cancelled=_is_cancelled, batch_size=6,
        )
        status = res.get("status")
        if status == "running":
            _render_progress(g_bar, g_msg, res.get("pct", 8), res.get("msg", "진행 중…"))
            _render_progress(o_bar, o_msg, res.get("pct", 8), res.get("msg", "진행 중…"))
            time.sleep(0.15); st.rerun()
            return
        elif status == "cancelled":
            ss.prep_both_running = False
            ss.prep_cancel_requested = False
            ss.index_job = None
            _render_progress(g_bar, g_msg, ss.get("p_shared", 0), "사용자 취소")
            _render_progress(o_bar, o_msg, ss.get("p_shared", 0), "사용자 취소")
            return
        elif status == "done":
            index = res["index"]
            ss.index_job = None
        else:
            _render_progress(g_bar, g_msg, 100, "인덱싱 실패")
            _render_progress(o_bar, o_msg, 100, "인덱싱 실패")
            ss.prep_both_running = False
            return

    # 3) 인덱스 준비 완료 → LLM 2개 준비 + QE 생성
    try:
        g_llm = make_llm("google", settings.GEMINI_API_KEY.get_secret_value(),
                         getattr(settings, "LLM_MODEL", "gemini-1.5-pro"),
                         float(ss.get("temperature", 0.0)))
        ss["llm_google"] = g_llm
        ss["qe_google"] = index.as_query_engine(
            llm=g_llm,
            response_mode=ss.get("response_mode", getattr(settings, "RESPONSE_MODE", "compact")),
            similarity_top_k=int(ss.get("similarity_top_k", getattr(settings, "SIMILARITY_TOP_K", 5))),
        )
        _render_progress(g_bar, g_msg, 100, "완료!")
    except Exception as e:
        _render_progress(g_bar, g_msg, 100, f"Gemini 준비 실패: {e}")

    try:
        if getattr(settings, "OPENAI_API_KEY", None) and settings.OPENAI_API_KEY.get_secret_value():
            o_llm = make_llm("openai", settings.OPENAI_API_KEY.get_secret_value(),
                             getattr(settings, "OPENAI_LLM_MODEL", "gpt-4o-mini"),
                             float(ss.get("temperature", 0.0)))
            ss["llm_openai"] = o_llm
            ss["qe_openai"] = index.as_query_engine(
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
    time.sleep(0.2); st.rerun()

# 실행/취소 버튼
left, right = st.columns([0.7, 0.3])
with left:
    clicked = st.button("🚀 한 번에 준비하기", key="prepare_both", use_container_width=True,
                        disabled=ss.prep_both_running or ss.prep_both_done)
with right:
    cancel_clicked = st.button("⛔ 준비 취소", key="cancel_prepare", use_container_width=True, type="secondary",
                               disabled=not ss.prep_both_running)

if cancel_clicked and ss.prep_both_running:
    ss.prep_cancel_requested = True
    if ss.get("index_job"):
        cancel_index_builder(ss.index_job)
    st.rerun()

if clicked and not (ss.prep_both_running or ss.prep_both_done):
    ss.p_shared = 0
    ss.prep_cancel_requested = False
    ss.prep_both_running = True
    ss.index_job = None
    st.rerun()

if ss.prep_both_running:
    run_prepare_both_step()

st.caption("준비 버튼을 다시 활성화하려면 아래 재설정 버튼을 누르세요.")
if st.button("🔧 재설정(버튼 다시 활성화)", disabled=not ss.prep_both_done):
    ss.prep_both_done = False
    ss.p_shared = 0
    st.rerun()


# ===== 대화 UI (그룹토론) =====
st.markdown("---")
st.subheader("💬 그룹토론 — 학생 ↔ 🤖Gemini(친절/꼼꼼) ↔ 🤖ChatGPT(유머러스/보완)")

ready_google = "qe_google" in ss
ready_openai = "qe_openai" in ss

if ss.session_terminated:
    st.warning("세션이 종료된 상태입니다. 새로고침으로 다시 시작하세요.")
    st.stop()

if not ready_google:
    st.info("먼저 **[🚀 한 번에 준비하기]** 를 클릭해 두뇌를 준비하세요. (OpenAI 키가 없으면 Gemini만 응답)")
    st.stop()

# 이미 쌓인 메시지 렌더
for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# ===== 유틸: 참고자료 꼬리 제거 & 맥락 구성 =====
def _strip_sources(text: str) -> str:
    return re.sub(r"\n+---\n\*참고 자료:.*$", "", text, flags=re.DOTALL)

def _build_context_for_models(messages: list[dict], limit_pairs: int = 2, max_chars: int = 2000) -> str:
    """최근 user/assistant 쌍을 limit_pairs개까지 모아 맥락을 만든다."""
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
    blocks = []
    for u, a in pairs:
        blocks.append(f"[학생]\n{u}\n\n[교사]\n{a}")
    ctx = "\n\n---\n\n".join(blocks).strip()
    if len(ctx) > max_chars:
        ctx = ctx[-max_chars:]
    return ctx


# ===== 페르소나 합성 =====
def _persona():
    # 기본 모드(설명/분석/독해)
    mode = st.session_state.get("mode_select", "💬 이유문법 설명")
    base = EXPLAINER_PROMPT if mode == "💬 이유문법 설명" else (ANALYST_PROMPT if mode == "🔎 구문 분석" else READER_PROMPT)
    # 공통 스타일
    common = (
        "역할: 학생의 영어 실력을 돕는 AI 교사.\n"
        "규칙: 근거가 불충분하면 그 사실을 명확히 밝힌다. 예시 문장은 짧고 점진적으로.\n"
    )
    return base + "\n" + common

GEMINI_STYLE = (
    "당신은 착하고 똑똑한 친구 같은 교사입니다. 지나치게 어렵게 말하지 말고, "
    "칭찬과 격려를 곁들여 차분히 안내하세요. 핵심 규칙은 정확성입니다."
)

CHATGPT_REVIEW_STYLE = (
    "당신은 유머러스하지만 정확한 동료 교사입니다. 동료(Gemini)의 답을 읽고 "
    "빠진 부분을 보완/교정하고, 마지막에 <최종 정리>를 제시하세요. 과한 농담은 피하고, "
    "짧고 명료한 유머 한두 줄만 허용됩니다."
)

# 모드 스위처(상단에 두기보단 대화 위젯 위에 배치)
mode = st.radio(
    "학습 모드", ["💬 이유문법 설명", "🔎 구문 분석", "📚 독해 및 요약"],
    horizontal=True, key="mode_select"
)

# ===== JSONL 로그 저장 (chat_log/ 서브폴더) =====
def _log_try(items):
    if not ss.save_logs:
        return
    try:
        parent_id = (getattr(settings, "CHATLOG_FOLDER_ID", None) or settings.GDRIVE_FOLDER_ID)
        sa = settings.GDRIVE_SERVICE_ACCOUNT_JSON
        if isinstance(sa, str):
            try:
                sa = json.loads(sa)
            except Exception:
                pass
        sub_id = get_chatlog_folder_id(parent_folder_id=parent_id, sa_json=sa)
        chat_store.append_jsonl(folder_id=sub_id, sa_json=sa, items=items)
        st.toast("대화 JSONL 저장 완료", icon="💾")
    except Exception as e:
        # 실패는 눈에 보이도록 error로 표시
        st.error(f"대화 JSONL 저장 실패: {e}")

# ===== 입력창 =====
user_input = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")

if user_input:
    # 0) 사용자 메시지
    ss.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # JSONL 로그: 사용자
    _log_try([chat_store.make_entry(ss.session_id, "user", "user", user_input, mode, model="user")])

    # 1) Gemini 1차 (이전 맥락 + 현재 질문)
    with st.spinner("🤖 Gemini 선생님이 먼저 답변합니다…"):
        prev_ctx = _build_context_for_models(ss.messages[:-1], limit_pairs=2, max_chars=2000)
        gemini_query = (f"[이전 대화]\n{prev_ctx}\n\n" if prev_ctx else "") + f"[학생 질문]\n{user_input}"
        ans_g = get_text_answer(ss["qe_google"], gemini_query, _persona() + "\n" + GEMINI_STYLE)

    content_g = f"**🤖 Gemini**\n\n{ans_g}"
    ss.messages.append({"role": "assistant", "content": content_g})
    with st.chat_message("assistant"):
        st.markdown(content_g)

    # JSONL 로그: Gemini
    _log_try([chat_store.make_entry(
        ss.session_id, "assistant", "Gemini", content_g, mode,
        model=getattr(settings, "LLM_MODEL", "gemini")
    )])

    # 2) ChatGPT 보완/검증 — RAG 없이 LLM 직답(동료 답변 읽고 보완)
    if ready_openai:
        from src.rag_engine import llm_complete

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

        prev_ctx_all = _build_context_for_models(ss.messages, limit_pairs=2, max_chars=2000)  # Gemini 방금 답 포함
        augmented = (
            (f"[이전 대화]\n{prev_ctx_all}\n\n" if prev_ctx_all else "") +
            f"[학생 질문]\n{user_input}\n\n"
            f"[동료의 1차 답변(Gemini)]\n{_strip_sources(ans_g)}\n\n"
            f"[당신의 작업]\n위 기준으로만 보완/검증하라."
        )

        with st.spinner("🤝 ChatGPT 선생님이 보완/검증 중…"):
            ans_o = llm_complete(
                ss.get("llm_openai"),
                _persona() + "\n" + CHATGPT_REVIEW_STYLE + "\n\n" + review_directive + "\n\n" + augmented
            )

        content_o = f"**🤖 ChatGPT**\n\n{ans_o}"
        ss.messages.append({"role": "assistant", "content": content_o})
        with st.chat_message("assistant"):
            st.markdown(content_o)
    else:
        with st.chat_message("assistant"):
            st.info("ChatGPT 키가 없어 Gemini만 응답했습니다. OPENAI_API_KEY를 추가하면 보완/검증이 활성화됩니다.")

    # ✅ Drive Markdown 대화 로그 자동 저장 (공유드라이브의 데이터 폴더 내 chat_log/)
    if ss.auto_save_chatlog and ss.messages:
        try:
            parent_id = (getattr(settings, "CHATLOG_FOLDER_ID", None) or settings.GDRIVE_FOLDER_ID)
            sa = settings.GDRIVE_SERVICE_ACCOUNT_JSON
            if isinstance(sa, str):
                try:
                    sa = json.loads(sa)
                except Exception:
                    pass
            save_chatlog_markdown(
                ss.session_id,
                ss.messages,
                parent_folder_id=parent_id,
                sa_json=sa,   # 서비스계정 dict 전달 필수
            )
            st.toast("Drive에 대화 저장 완료 (chat_log/)", icon="💾")
        except Exception as e:
            st.error(f"Drive Markdown 저장 실패: {e}")

    st.rerun()
