# app.py — 스텝 인덱싱(중간취소/재개) + 두뇌준비 안정화
#        + 인덱싱 보고서(스킵 파일 표시)
#        + Drive 대화로그 저장(❶ OAuth: Markdown / ❷ 서비스계정: JSONL, chat_log/)
#        + 페르소나: 🤖Gemini(친절/꼼꼼), 🤖ChatGPT(유머러스/보완)
#        + 양쪽 모두 RAG 우선, 부족 시 "(자료 밖 지식)"으로 최소 보완

# ===== Imports =====
import os
import time
import uuid
import re
import json
import pandas as pd
import streamlit as st

# ---- Streamlit query_params 호환 패치 (deprecation 배너 제거) ----
# old -> st.experimental_get_query_params / st.experimental_set_query_params
# new -> st.query_params
if not hasattr(st, "_qp_compat_patched"):
    def _compat_get_query_params():
        try:
            raw = dict(st.query_params)      # {'k': 'v'} or {'k': ['a','b']}
            return {k: (v if isinstance(v, list) else [v]) for k, v in raw.items()}
        except Exception:
            return {}

    def _compat_set_query_params(**kwargs):
        try:
            qp = st.query_params
            qp.clear()
            for k, v in kwargs.items():
                qp[k] = v
        except Exception:
            pass

    st.experimental_get_query_params = _compat_get_query_params
    st.experimental_set_query_params = _compat_set_query_params
    st._qp_compat_patched = True
# ---------------------------------------------------------------

# 기본 UI
from src.ui import load_css, render_header

# 설정
from src.config import settings

# RAG/인덱싱 유틸(스텝 빌더)
from src.rag_engine import (
    set_embed_provider, make_llm, get_text_answer, CancelledError,
    start_index_builder, resume_index_builder, cancel_index_builder,
    smoke_test_drive, preview_drive_files,
)

# JSONL 로그 스토어(서비스계정 경로)
from src import chat_store

# Drive 로그 유틸
# ❶ OAuth로 Markdown 저장
from src.drive_log import save_chatlog_markdown_oauth
# ❷ 서비스계정으로 JSONL 저장 시 chat_log/ 보장
from src.drive_log import get_chatlog_folder_id

# 페르소나 프롬프트
from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT

# ===== 공통 규칙(업로드 자료 최우선 + 부족분만 보강) =====
RAG_FIRST_RULES = (
    "규칙: 반드시 업로드된 학습 자료(문서/슬라이드 등)에서 근거를 먼저 찾는다. "
    "자료에 직접 근거가 없으면 그때에만 일반 지식으로 짧게 보완하되, "
    "보강한 문장 앞에 '(자료 밖 지식)'이라고 표시한다. "
    "자료 근거가 불충분하면 그 사실을 명확히 밝힌다."
)

# 두 캐릭터 스타일
GEMINI_STYLE = (
    "당신은 착하고 똑똑한 친구 같은 교사입니다. 지나치게 어렵게 말하지 말고, "
    "칭찬과 격려를 곁들여 차분히 안내하세요. 핵심 규칙은 정확성입니다."
)
CHATGPT_REVIEW_STYLE = (
    "당신은 유머러스하지만 정확한 동료 교사입니다. 동료(Gemini)의 답을 읽고 "
    "빠진 부분을 보완/교정하고, 마지막에 <최종 정리>를 제시하세요. 과한 농담은 피하고, "
    "짧고 명료한 유머 한두 줄만 허용됩니다."
)

# ===== 런타임 안정화 환경변수 =====
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ===== 페이지 설정 (첫 호출만!) =====
st.set_page_config(
    page_title="나의 AI 영어 교사",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# OAuth 리다이렉트 처리 (페이지 설정 직후)
from src.google_oauth import finish_oauth_if_redirected
finish_oauth_if_redirected()

# ===== 세션 상태(먼저 정의! 이후에 ss 참조) =====
ss = st.session_state
ss.setdefault("session_id", uuid.uuid4().hex[:12])   # 대화 세션 ID
ss.setdefault("messages", [])                        # {"role": "user"|"assistant", "content": str}
ss.setdefault("auto_save_chatlog", True)             # OAuth Markdown 자동 저장
ss.setdefault("save_logs", False)                    # 서비스계정 JSONL 저장(기본 False; 쿼터 이슈 회피)
ss.setdefault("prep_both_running", False)
ss.setdefault("prep_both_done", ("qe_google" in ss) or ("qe_openai" in ss))
ss.setdefault("p_shared", 0)
ss.setdefault("prep_cancel_requested", False)
ss.setdefault("session_terminated", False)
ss.setdefault("index_job", None)                     # 스텝 빌더 상태

# ===== 기본 UI / 헤더 =====
load_css()
render_header()
st.info("✅ 변경이 있을 때만 인덱싱합니다. 저장된 두뇌가 있으면 즉시 불러옵니다. (중간 취소/재개 지원)")

# ===== 사이드바: 자동저장 토글 + OAuth 로그인 =====
from src.google_oauth import start_oauth, is_signed_in, build_drive_service, get_user_email, sign_out

with st.sidebar:
    ss.auto_save_chatlog = st.toggle("대화 자동 저장 (OAuth/내 드라이브, Markdown)", value=ss.auto_save_chatlog)
    ss.save_logs = st.toggle("대화 JSONL 저장 (서비스계정/chat_log/)", value=ss.save_logs,
                             help="공유드라이브 Writer 권한 필요. 쿼터 문제 시 끄기 권장.")
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
            st.rerun()

# ===== Google Drive 연결 테스트 =====
st.markdown("## 🔗 Google Drive 연결 테스트")
st.caption("서비스계정은 **공유 드라이브**에 Writer 권한이 있어야(저장 시) 오류 없이 동작합니다. 인덱싱은 Readonly면 충분.")

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

# 진행률 바
c_g, c_o = st.columns(2)
with c_g:
    st.caption("Gemini 진행"); g_bar = st.empty(); g_msg = st.empty()
with c_o:
    st.caption("ChatGPT 진행"); o_bar = st.empty(); o_msg = st.empty()

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
        _render_progress(g_bar, g_msg, p, m); _render_progress(o_bar, o_msg, p, m)

    def umsg(m):
        _render_progress(g_bar, g_msg, ss.get("p_shared", 0), m)
        _render_progress(o_bar, o_msg, ss.get("p_shared", 0), m)

    job = ss.get("index_job")
    persist_dir = f"{getattr(settings,'PERSIST_DIR','/tmp/my_ai_teacher/storage_gdrive')}_shared"

    if job is None:
        # 처음 시작 (빠른 모드 제거 → 전체 인덱싱)
        res = start_index_builder(
            update_pct=upd, update_msg=umsg,
            gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
            raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
            persist_dir=persist_dir,
            manifest_path=getattr(settings, "MANIFEST_PATH", "/tmp/my_ai_teacher/drive_manifest.json"),
            max_docs=None,                 # 🔸 빠른 모드 제거
            is_cancelled=_is_cancelled,
        )
        status = res.get("status")
        if status == "done":
            index = res["index"]
        elif status == "running":
            ss.index_job = res["job"]
            _render_progress(g_bar, g_msg, res.get("pct", 8), res.get("msg", "진행 중…"))
            _render_progress(o_bar, o_msg, res.get("pct", 8), res.get("msg", "진행 중…"))
            time.sleep(0.2); st.rerun(); return
        else:
            _render_progress(g_bar, g_msg, 100, "인덱싱 시작 실패")
            _render_progress(o_bar, o_msg, 100, "인덱싱 시작 실패")
            ss.prep_both_running = False; return
    else:
        # 재개
        res = resume_index_builder(job=job, update_pct=upd, update_msg=umsg,
                                   is_cancelled=_is_cancelled, batch_size=6)
        status = res.get("status")
        if status == "running":
            _render_progress(g_bar, g_msg, res.get("pct", 8), res.get("msg", "진행 중…"))
            _render_progress(o_bar, o_msg, res.get("pct", 8), res.get("msg", "진행 중…"))
            time.sleep(0.15); st.rerun(); return
        elif status == "cancelled":
            ss.prep_both_running = False; ss.prep_cancel_requested = False; ss.index_job = None
            _render_progress(g_bar, g_msg, ss.get("p_shared", 0), "사용자 취소")
            _render_progress(o_bar, o_msg, ss.get("p_shared", 0), "사용자 취소")
            return
        elif status == "done":
            index = res["index"]
            ss.index_job = None
        else:
            _render_progress(g_bar, g_msg, 100, "인덱싱 실패")
            _render_progress(o_bar, o_msg, 100, "인덱싱 실패")
            ss.prep_both_running = False; return

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
ready_openai = "llm_openai" in ss  # RAG가 없어도 폴백 보완은 가능

if ss.session_terminated:
    st.warning("세션이 종료된 상태입니다. 새로고침으로 다시 시작하세요.")
    st.stop()

if "qe_google" not in ss:
    st.info("먼저 **[🚀 한 번에 준비하기]** 를 클릭해 두뇌를 준비하세요. (OpenAI 키가 없으면 Gemini만 응답)")
    st.stop()

# 이미 쌓인 메시지 렌더
for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 유틸: 참고자료 꼬리 제거 & 맥락 구성
def _strip_sources(text: str) -> str:
    return re.sub(r"\n+---\n\*참고 자료:.*$", "", text, flags=re.DOTALL)

def _build_context_for_models(messages: list[dict], limit_pairs: int = 2, max_chars: int = 2000) -> str:
    pairs, buf_user = [], None
    for m in reversed(messages):
        role, content = m.get("role"), str(m.get("content", "")).strip()
        if role == "assistant":
            content = re.sub(r"^\*\*🤖 .*?\*\*\s*\n+", "", content).strip()
            if buf_user is not None:
                pairs.append((buf_user, content)); buf_user = None
                if len(pairs) >= limit_pairs: break
        elif role == "user" and buf_user is None:
            buf_user = content
    pairs = list(reversed(pairs))
    blocks = [f"[학생]\n{u}\n\n[교사]\n{a}" for u, a in pairs]
    ctx = "\n\n---\n\n".join(blocks).strip()
    return ctx[-max_chars:] if len(ctx) > max_chars else ctx

# 페르소나 합성
def _persona():
    mode = st.session_state.get("mode_select", "💬 이유문법 설명")
    base = EXPLAINER_PROMPT if mode == "💬 이유문법 설명" else (ANALYST_PROMPT if mode == "🔎 구문 분석" else READER_PROMPT)
    common = (
        "역할: 학생의 영어 실력을 돕는 AI 교사.\n"
        "규칙: 근거가 불충분하면 그 사실을 명확히 밝힌다. 예시 문장은 짧고 점진적으로.\n"
    )
    return base + "\n" + common

# 모드 스위처
mode = st.radio("학습 모드", ["💬 이유문법 설명", "🔎 구문 분석", "📚 독해 및 요약"],
                horizontal=True, key="mode_select")

# JSONL 로그 저장(서비스계정, chat_log/)
def _log_try(items):
    if not ss.save_logs:
        return
    try:
        parent_id = (getattr(settings, "CHATLOG_FOLDER_ID", None) or settings.GDRIVE_FOLDER_ID)
        sa = settings.GDRIVE_SERVICE_ACCOUNT_JSON
        if isinstance(sa, str):
            try: sa = json.loads(sa)
            except Exception: pass
        sub_id = get_chatlog_folder_id(parent_folder_id=parent_id, sa_json=sa)
        chat_store.append_jsonl(folder_id=sub_id, sa_json=sa, items=items)
        st.toast("대화 JSONL 저장 완료", icon="💾")
    except Exception as e:
        st.warning(f"대화 JSONL 저장 실패: {e}")

# ===== 입력창 =====
user_input = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
if user_input:
    # 0) 사용자 메시지
    ss.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # JSONL 로그: 사용자
    _log_try([chat_store.make_entry(ss.session_id, "user", "user", user_input, mode, model="user")])

    # 1) Gemini 1차 (RAG 우선)
    with st.spinner("🤖 Gemini 선생님이 먼저 답변합니다…"):
        prev_ctx = _build_context_for_models(ss.messages[:-1], limit_pairs=2, max_chars=2000)
        gemini_query = (f"[이전 대화]\n{prev_ctx}\n\n" if prev_ctx else "") + f"[학생 질문]\n{user_input}"
        ans_g = get_text_answer(
            ss["qe_google"],
            gemini_query,
            _persona() + "\n" + GEMINI_STYLE + "\n" + RAG_FIRST_RULES
        )

    content_g = f"**🤖 Gemini**\n\n{ans_g}"
    ss.messages.append({"role": "assistant", "content": content_g})
    with st.chat_message("assistant"):
        st.markdown(content_g)

    # JSONL 로그: Gemini
    _log_try([chat_store.make_entry(
        ss.session_id, "assistant", "Gemini", content_g, mode,
        model=getattr(settings, "LLM_MODEL", "gemini")
    )])

    # 2) ChatGPT 보완/검증 — RAG 우선(OpenAI QueryEngine) → 실패 시 LLM 폴백
    if ready_openai:
        review_directive = (
            "역할: 당신은 유머러스하지만 정확한 동료 영어교사다. "
            "동료(Gemini)의 1차 답변을 바탕으로 교정/보완하고 마무리한다.\n"
            "출력 지침:\n"
            "1) 잘못된 부분은 바로잡고 누락 포인트를 채운다.\n"
            "2) 예문 2~3개(간단/점진적) 추가.\n"
            "3) 마지막에 <최종 정리>를 불릿 3~5개로 제시.\n"
        )

        prev_ctx_all = _build_context_for_models(ss.messages, limit_pairs=2, max_chars=2000)  # Gemini 방금 답 포함
        rag_query_for_openai = (
            (f"[이전 대화]\n{prev_ctx_all}\n\n" if prev_ctx_all else "") +
            f"[학생 질문]\n{user_input}\n\n"
            f"[동료의 1차 답변(Gemini)]\n{_strip_sources(ans_g)}\n\n"
            "[작업]\n위 내용을 바탕으로 업로드 자료를 최우선으로 점검·보완하고, "
            "누락/오류를 고친 뒤 간결하게 마무리하라."
        )

        used_rag = False
        ans_o = None
        try:
            if "qe_openai" in ss:
                ans_o = get_text_answer(
                    ss["qe_openai"],
                    rag_query_for_openai,
                    _persona() + "\n" + CHATGPT_REVIEW_STYLE + "\n" + RAG_FIRST_RULES + "\n" + review_directive,
                )
                used_rag = True
        except Exception as e:
            used_rag = False
            ans_o = f"(RAG 보완 실패로 폴백합니다) 에러: {e}"

        if not used_rag:
            from src.rag_engine import llm_complete
            augmented = (
                (f"[이전 대화]\n{prev_ctx_all}\n\n" if prev_ctx_all else "") +
                f"[학생 질문]\n{user_input}\n\n"
                f"[동료의 1차 답변(Gemini)]\n{_strip_sources(ans_g)}\n\n"
                f"[당신의 작업]\n업로드 자료를 우선한다는 전제를 유지하되, 자료 참조가 불가능하면 "
                f"간단한 일반 지식으로만 보완하라. 보강 문장 앞에는 '(자료 밖 지식)'을 붙여라."
            )
            try:
                ans_o = llm_complete(
                    ss.get("llm_openai"),
                    _persona() + "\n" + CHATGPT_REVIEW_STYLE + "\n" + review_directive + "\n\n" + augmented
                )
            except Exception as e:
                ans_o = f"(보완 단계 오류로 Gemini 답변만 제공합니다)\n\n에러: {e}"

        content_o = f"**🤖 ChatGPT**\n\n{ans_o}"
        ss.messages.append({"role": "assistant", "content": content_o})
        with st.chat_message("assistant"):
            st.markdown(content_o)
    else:
        with st.chat_message("assistant"):
            st.info("ChatGPT 키가 없어 Gemini만 응답했습니다. OPENAI_API_KEY를 추가하면 보완/검증이 활성화됩니다.")

    # ❶ OAuth: Markdown 자동 저장 (내 드라이브, my-ai-teacher-data/chat_log/)
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

    st.rerun()
