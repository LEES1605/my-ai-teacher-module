# app.py — 스텝 인덱싱(중간취소/재개) + 두뇌준비 안정화
#        + 인덱싱 보고서(스킵 표시)
#        + Drive 대화로그 저장(❶ OAuth: Markdown / ❷ 서비스계정: JSONL, chat_log/)
#        + 페르소나: 🤖Gemini(친절/꼼꼼), 🤖ChatGPT(유머러스/보완)

import os, time, uuid, re, json
import pandas as pd
import streamlit as st

# ===== 페이지 설정 (첫 호출만) =====
st.set_page_config(page_title="나의 AI 영어 교사", layout="wide", initial_sidebar_state="collapsed")

# ===== 런타임 안정화 =====
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ===== 세션 상태 =====
ss = st.session_state
ss.setdefault("session_id", uuid.uuid4().hex[:12])
ss.setdefault("messages", [])
ss.setdefault("auto_save_chatlog", True)    # OAuth Markdown 저장
ss.setdefault("save_logs", False)           # SA JSONL 저장(기본 꺼둠: 쿼터 이슈 회피)
ss.setdefault("prep_both_running", False)
ss.setdefault("prep_both_done", ("qe_google" in ss) or ("qe_openai" in ss))
ss.setdefault("prep_cancel_requested", False)
ss.setdefault("session_terminated", False)
ss.setdefault("index_job", None)

# ===== (진단) 부팅 하트비트 =====
st.caption(f"heartbeat ✅ keys={list(ss.keys())[:8]}")

# ===== 기본 UI =====
from src.ui import load_css, render_header
load_css(); render_header()
st.info("✅ 변경이 있을 때만 인덱싱합니다. 저장된 두뇌가 있으면 즉시 로드합니다. (중간 취소/재개 지원)")

# ===== OAuth 리다이렉트 처리 (한 번만) =====
try:
    from src.google_oauth import finish_oauth_if_redirected
    if not st.secrets.get("OAUTH_DISABLE_FINISH"):
        if not ss.get("_oauth_finalized", False):
            finalized = finish_oauth_if_redirected()  # code/state 있으면 토큰 교환
            if finalized:
                ss["_oauth_finalized"] = True
                # URL 파라미터 제거
                try:
                    st.query_params.clear()
                except Exception:
                    st.experimental_set_query_params()
                st.rerun()
except Exception as e:
    st.warning(f"OAuth finalize skipped: {e}")

# ===== 사이드바: OAuth 로그인/로그아웃, 저장 옵션 =====
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
            sign_out(); st.rerun()

# ===== Drive 연결 테스트 =====
st.markdown("## 🔗 Google Drive 연결 테스트")
st.caption("서비스계정 저장은 공유드라이브 Writer 권한이 필요. 인덱싱은 Readonly면 충분합니다.")
try:
    # ── replace this block inside app.py (Google Drive 연결 테스트 카드) ──
# ==== Google Drive 연결/진단 유틸 임포트 ====
try:
    from src.rag_engine import (
        smoke_test_drive,
        preview_drive_files,
        drive_diagnostics,
    )

ok, headline, details = drive_diagnostics(settings.GDRIVE_FOLDER_ID)
if ok:
    st.success(headline)
else:
    st.error(headline)

with st.expander("🔎 연결/권한 진단 상세", expanded=not ok):
    for line in details:
        st.write("• ", line)

# (원하시면 '폴더 파일 미리보기' 버튼은 기존 코드 유지)
# ─────────────────────────────────────────────────────────────────

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
            df = df.rename(columns={"modified": "modified_at"})[["name","link","type","modified_at"]]
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

# ===== 인덱싱 보고서 =====
rep = ss.get("indexing_report")
if rep:
    with st.expander("🧾 인덱싱 보고서 (스킵된 파일 보기)", expanded=False):
        st.write(f"총 파일(매니페스트): {rep.get('total_manifest')}, "
                 f"로딩된 문서 수: {rep.get('loaded_docs')}, "
                 f"스킵: {rep.get('skipped_count')}")
        skipped = rep.get("skipped", [])
        if skipped:
            st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)
        else:
            st.caption("스킵된 파일이 없습니다 🎉")

# ===== 두뇌 준비(스텝) =====
st.markdown("---"); st.subheader("🧠 두뇌 준비 — 저장본 로드 ↔ 변경 시 증분 인덱싱 (중간 취소/재개)")

# 진행률 UI
c_g, c_o = st.columns(2)
with c_g: st.caption("Gemini 진행"); g_bar = st.empty(); g_msg = st.empty()
with c_o: st.caption("ChatGPT 진행"); o_bar = st.empty(); o_msg = st.empty()

def _render_progress(slot_bar, slot_msg, pct: int, msg: str | None = None):
    p = max(0, min(100, int(pct)))
    slot_bar.markdown(f"""
<div class="gp-wrap"><div class="gp-fill" style="width:{p}%"></div><div class="gp-label">{p}%</div></div>
""", unsafe_allow_html=True)
    if msg is not None:
        slot_msg.markdown(f"<div class='gp-msg'>{msg}</div>", unsafe_allow_html=True)

def _is_cancelled() -> bool:
    return bool(ss.get("prep_cancel_requested", False))

from src.config import settings
from src.rag_engine import (
    set_embed_provider, make_llm, get_text_answer, CancelledError,
    start_index_builder, resume_index_builder, cancel_index_builder
)

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

    persist_dir = f"{getattr(settings,'PERSIST_DIR','/tmp/my_ai_teacher/storage_gdrive')}_shared"
    job = ss.get("index_job")

    try:
        if job is None:
            res = start_index_builder(
                update_pct=upd, update_msg=umsg,
                gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
                raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
                persist_dir=persist_dir,
                manifest_path=getattr(settings, "MANIFEST_PATH", "/tmp/my_ai_teacher/drive_manifest.json"),
                max_docs=None,                      # 빠른모드 제거 → 전체
                is_cancelled=_is_cancelled,
            )
        else:
            res = resume_index_builder(
                job=job, update_pct=upd, update_msg=umsg,
                is_cancelled=_is_cancelled, batch_size=6
            )

        status = res.get("status")
        if status == "running":
            ss.index_job = res["job"]
            _render_progress(g_bar, g_msg, res.get("pct", 8), res.get("msg", "진행 중…"))
            _render_progress(o_bar, o_msg, res.get("pct", 8), res.get("msg", "진행 중…"))
            time.sleep(0.15); st.rerun(); return

        if status == "cancelled":
            ss.prep_both_running = False; ss.prep_cancel_requested = False; ss.index_job = None
            _render_progress(g_bar, g_msg, res.get("pct", 0), "사용자 취소")
            _render_progress(o_bar, o_msg, res.get("pct", 0), "사용자 취소")
            return

        if status != "done":
            _render_progress(g_bar, g_msg, 100, "인덱싱 실패")
            _render_progress(o_bar, o_msg, 100, "인덱싱 실패")
            ss.prep_both_running = False
            return

        index = res["index"]
        ss.index_job = None

    except Exception as e:
        ss.prep_both_running = False; ss.index_job = None
        _render_progress(g_bar, g_msg, 100, f"에러: {e}")
        _render_progress(o_bar, o_msg, 100, f"에러: {e}")
        return

    # 3) QE 생성
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
    ss.prep_cancel_requested = False
    ss.prep_both_running = True
    ss.index_job = None
    st.rerun()

if ss.prep_both_running:
    run_prepare_both_step()

st.caption("준비 버튼을 다시 활성화하려면 아래 재설정 버튼을 누르세요.")
if st.button("🔧 재설정(버튼 다시 활성화)", disabled=not ss.prep_both_done):
    ss.prep_both_done = False
    st.rerun()

# ===== 대화 UI (그룹토론) =====
st.markdown("---")
st.subheader("💬 그룹토론 — 학생 ↔ 🤖Gemini(친절/꼼꼼) ↔ 🤖ChatGPT(유머러스/보완)")

ready_google = "qe_google" in ss
ready_openai = "qe_openai" in ss
if ss.session_terminated:
    st.warning("세션이 종료된 상태입니다. 새로고침으로 다시 시작하세요."); st.stop()
if not ready_google:
    st.info("먼저 **[🚀 한 번에 준비하기]**로 두뇌를 준비하세요. (OpenAI 키 없으면 Gemini만 응답)"); st.stop()

# 과거 메시지 렌더
for m in ss.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# 맥락/도우미
def _strip_sources(text: str) -> str:
    return re.sub(r"\n+---\n\*참고 자료:.*$", "", text, flags=re.DOTALL)

def _build_context(messages, limit_pairs=2, max_chars=2000) -> str:
    pairs, buf_user = [], None
    for m in reversed(messages):
        role, content = m.get("role"), str(m.get("content","")).strip()
        if role == "assistant":
            content = re.sub(r"^\*\*🤖 .*?\*\*\s*\n+", "", content).strip()
            if buf_user is not None:
                pairs.append((buf_user, content)); buf_user = None
                if len(pairs) >= limit_pairs: break
        elif role == "user" and buf_user is None:
            buf_user = content
    pairs = list(reversed(pairs))
    blocks = [f"[학생]\n{u}\n\n[교사]\n{a}" for u,a in pairs]
    ctx = "\n\n---\n\n".join(blocks).strip()
    return ctx[-max_chars:] if len(ctx) > max_chars else ctx

# 페르소나
from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
def _persona():
    mode = ss.get("mode_select", "💬 이유문법 설명")
    base = EXPLAINER_PROMPT if mode=="💬 이유문법 설명" else (ANALYST_PROMPT if mode=="🔎 구문 분석" else READER_PROMPT)
    common = "역할: 학생의 영어 실력을 돕는 AI 교사.\n규칙: 근거가 불충분하면 그 사실을 명확히 밝힌다. 예시는 짧고 점진적으로."
    return base + "\n" + common

GEMINI_STYLE = "당신은 착하고 똑똑한 친구 같은 교사입니다. 칭찬과 격려, 정확한 설명."
CHATGPT_STYLE = ("당신은 유머러스하지만 정확한 동료 교사입니다. 동료(Gemini)의 답을 읽고 "
                 "빠진 부분을 보완/교정하고 마지막에 <최종 정리>로 요약하세요. 과한 농담 금지.")

# 모드 스위치
mode = st.radio("학습 모드", ["💬 이유문법 설명", "🔎 구문 분석", "📚 독해 및 요약"],
                horizontal=True, key="mode_select")

# 서비스계정 JSONL 저장 (chat_log/)
from src import chat_store
from src.drive_log import get_chatlog_folder_id, save_chatlog_markdown_oauth

def _jsonl_log(items):
    if not ss.save_logs: return
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

# 입력
user_input = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
if user_input:
    ss.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"): st.markdown(user_input)

    # JSONL 사용자 로그
    _jsonl_log([chat_store.make_entry(ss.session_id, "user", "user", user_input, mode, model="user")])

    # Gemini 1차
    with st.spinner("🤖 Gemini 선생님이 먼저 답합니다…"):
        prev_ctx = _build_context(ss.messages[:-1], limit_pairs=2, max_chars=2000)
        gemini_query = (f"[이전 대화]\n{prev_ctx}\n\n" if prev_ctx else "") + f"[학생 질문]\n{user_input}"
        ans_g = get_text_answer(ss["qe_google"], gemini_query, _persona() + "\n" + GEMINI_STYLE)

    content_g = f"**🤖 Gemini**\n\n{ans_g}"
    ss.messages.append({"role":"assistant","content":content_g})
    with st.chat_message("assistant"): st.markdown(content_g)

    _jsonl_log([chat_store.make_entry(
        ss.session_id, "assistant", "Gemini", content_g, mode, model=getattr(settings,"LLM_MODEL","gemini")
    )])

    # ChatGPT 보완 (있을 때)
    if ready_openai:
        from src.rag_engine import llm_complete
        review_directive = (
            "역할: 동료 AI 영어교사\n"
            "목표: [이전 대화], [학생 질문], [동료의 1차 답변]을 읽고 사실오류/빠진점/모호함을 보완.\n"
            "지침: 1)핵심 간결 재정리 2)틀린 부분 근거와 교정 3)예문 2~3개 4)<최종 정리>로 요약. 외부검색 금지."
        )
        prev_all = _build_context(ss.messages, limit_pairs=2, max_chars=2000)
        augmented = ((f"[이전 대화]\n{prev_all}\n\n" if prev_all else "") +
                     f"[학생 질문]\n{user_input}\n\n"
                     f"[동료의 1차 답변(Gemini)]\n{_strip_sources(ans_g)}\n\n"
                     "[당신의 작업]\n위 기준으로만 보완/검증.")
        with st.spinner("🤝 ChatGPT 선생님이 보완/검증 중…"):
            ans_o = llm_complete(ss.get("llm_openai"),
                                 _persona() + "\n" + CHATGPT_STYLE + "\n\n" + review_directive + "\n\n" + augmented)
        content_o = f"**🤖 ChatGPT**\n\n{ans_o}"
        ss.messages.append({"role":"assistant","content":content_o})
        with st.chat_message("assistant"): st.markdown(content_o)
    else:
        with st.chat_message("assistant"):
            st.info("ChatGPT 키가 없어 Gemini만 응답했습니다. OPENAI_API_KEY를 추가하면 보완/검증이 활성화됩니다.")

    # OAuth Markdown 저장(내 드라이브)
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
