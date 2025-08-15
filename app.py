# app.py — 그룹토론 + 증분 인덱싱(Resume) + 준비 취소/재개 + 로그 상태 고정표시
import os, time, uuid, re, json
import pandas as pd
import streamlit as st

# 기본 UI
from src.ui import load_css, render_header
# Drive 로그 유틸
from src.drive_log import save_chatlog_markdown, get_chatlog_folder_id
# 프롬프트/페르소나
from src.prompts import (
    EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT,
    GEMINI_TONE, CHATGPT_TONE, REVIEW_DIRECTIVE,
)
# 설정
from src.config import settings
# JSONL 저장 모듈
from src import chat_store

# ========== 런타임 안정화 ==========
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ========== 페이지 ==========
st.set_page_config(page_title="나의 AI 영어 교사", layout="wide", initial_sidebar_state="collapsed")
load_css(); render_header()

# ========== 세션 ==========
ss = st.session_state
ss.setdefault("session_id", uuid.uuid4().hex[:12])
ss.setdefault("messages", [])
ss.setdefault("auto_save_chatlog", True)
ss.setdefault("save_logs", True)
ss.setdefault("prep_both_running", False)
ss.setdefault("prep_both_done", ("qe_google" in ss) or ("qe_openai" in ss))
ss.setdefault("p_shared", 0)
ss.setdefault("prep_cancel_requested", False)
ss.setdefault("session_terminated", False)
# 로그 상태(지속 표시)
ss.setdefault("log_status", None)   # "ok" | "err" | None
ss.setdefault("log_error", "")

# 사이드바: 자동 저장 토글 + 로그 상태 박스
with st.sidebar:
    ss.auto_save_chatlog = st.toggle("대화 자동 저장(Drive)", value=ss.auto_save_chatlog)
    log_box = st.container()
def _render_log_status():
    log_box.empty()
    if ss.log_status == "ok":
        log_box.success("💾 Drive 저장 완료 (chat_log/)")
    elif ss.log_status == "err":
        log_box.error(f"⚠️ Drive 저장 실패: {ss.log_error}")

st.info("✅ 인덱싱은 1번만 수행하고, 저장된 인덱스를 재사용합니다. (스텝 실행·취소/재개 지원)")

# --- rag_engine 임포트 가드 (정확한 오류 표시) ---
def _import_rag_engine_or_die():
    import traceback
    try:
        import src.rag_engine as re_mod
        required = [
            "set_embed_provider","make_llm","get_or_build_index","get_text_answer","CancelledError",
            "start_index_builder","resume_index_builder","cancel_index_builder","get_index_progress"
        ]
        missing = [n for n in required if not hasattr(re_mod, n)]
        if missing:
            st.error("rag_engine가 최신이 아닙니다. 누락: " + ", ".join(missing))
            st.stop()
        return re_mod
    except Exception as e:
        st.error(f"rag_engine 임포트 오류: {e}")
        st.code(traceback.format_exc()); st.stop()
re_mod = _import_rag_engine_or_die()

# ===== Google Drive 연결 테스트 =====
st.markdown("## 🔗 Google Drive 연결 테스트")
st.caption("서비스계정에 폴더 ‘쓰기(Writer)’ 권한이 있어야 대화 로그 저장이 됩니다.")
col1, col2 = st.columns([0.65, 0.35])

with col1:
    if st.button("폴더 파일 미리보기 (최신 10개)", use_container_width=True):
        ok, msg, rows = re_mod.preview_drive_files(max_items=10)
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
                }, hide_index=True,
            )
        elif ok:
            st.warning("폴더에 파일이 없거나 접근할 수 없습니다.")
        else:
            st.error(msg)

with col2:
    ok, msg = re_mod.smoke_test_drive()
    if ok:
        st.success(msg)
    else:
        st.warning(msg)

# ===== 인덱싱 보고서(스킵 파일) =====
rep = st.session_state.get("indexing_report")
if rep:
    with st.expander("🧾 인덱싱 보고서(스킵된 파일 보기)", expanded=False):
        st.write(
            f"총 파일(매니페스트): {rep.get('total_manifest')} · "
            f"로딩된 문서: {rep.get('loaded_docs')} · "
            f"스킵: {rep.get('skipped_count')}"
        )
        if rep.get("skipped"):
            st.dataframe(pd.DataFrame(rep["skipped"]), use_container_width=True, hide_index=True)
        else:
            st.caption("스킵된 파일이 없습니다 🎉")

# ===== 두뇌 준비(스텝 실행/취소/재개) =====
st.markdown("----")
st.subheader("🧠 두뇌 준비 — 인덱스 1회 + Gemini/ChatGPT")

def _render_progress(slot_bar, slot_msg, pct: int, msg: str | None = None):
    p = max(0, min(100, int(pct)))
    slot_bar.markdown(
        f"""<div class="gp-wrap"><div class="gp-fill" style="width:{p}%"></div>
        <div class="gp-label">{p}%</div></div>""", unsafe_allow_html=True)
    if msg is not None: slot_msg.markdown(f"<div class='gp-msg'>{msg}</div>", unsafe_allow_html=True)

def _bump_max(key: str, pct: int) -> int:
    now = int(pct); prev = int(ss.get(key, 0))
    if now < prev: now = prev
    ss[key] = now; return now

with st.expander("⚙️ 옵션", expanded=False):
    fast = st.checkbox("⚡ 빠른 준비 (처음 N개 문서만 인덱싱)", value=True,
                       disabled=ss.prep_both_running or ss.prep_both_done)
    max_docs = st.number_input("N (빠른 모드일 때만)", min_value=5, max_value=500, value=40, step=5,
                               disabled=ss.prep_both_running or ss.prep_both_done)
    ss.save_logs = st.checkbox("💾 JSONL 로그 저장", value=ss.save_logs, disabled=False)

st.markdown("### 🚀 인덱싱 1번 + 두 LLM 준비")
c_g, c_o = st.columns(2)
with c_g: st.caption("Gemini 진행"); g_bar = st.empty(); g_msg = st.empty()
with c_o: st.caption("ChatGPT 진행"); o_bar = st.empty(); o_msg = st.empty()

def _is_cancelled() -> bool:
    return bool(ss.get("prep_cancel_requested", False))

def run_prepare_both_step():
    # A. 임베딩 설정
    embed_provider = "openai"
    embed_api = (settings.OPENAI_API_KEY.get_secret_value()
                 if getattr(settings,"OPENAI_API_KEY",None) else "")
    embed_model = getattr(settings, "OPENAI_EMBED_MODEL", "text-embedding-3-small")
    if not embed_api:
        embed_provider = "google"
        embed_api = settings.GEMINI_API_KEY.get_secret_value()
        embed_model = getattr(settings, "EMBED_MODEL", "text-embedding-004")

    try:
        if _is_cancelled(): raise re_mod.CancelledError("사용자 취소")
        p = _bump_max("p_shared", 5)
        _render_progress(g_bar, g_msg, p, f"임베딩 설정({embed_provider})")
        _render_progress(o_bar, o_msg, p, f"임베딩 설정({embed_provider})")
        re_mod.set_embed_provider(embed_provider, embed_api, embed_model)
    except re_mod.CancelledError:
        ss.prep_both_running=False; ss.prep_cancel_requested=False
        _render_progress(g_bar,g_msg,ss.p_shared,"사용자 취소")
        _render_progress(o_bar,o_msg,ss.p_shared,"사용자 취소"); st.stop()
    except Exception as e:
        _render_progress(g_bar,g_msg,100,f"임베딩 실패: {e}")
        _render_progress(o_bar,o_msg,100,f"임베딩 실패: {e}")
        ss.prep_both_running=False; st.stop()

    # B. 인덱스 준비(Resume/취소 지원)
    persist_dir = f"{getattr(settings,'PERSIST_DIR','/tmp/my_ai_teacher/storage_gdrive')}_shared"
    re_mod.start_index_builder(
        update_pct=lambda pct,msg=None: (
            _render_progress(g_bar,g_msg,_bump_max('p_shared',pct),msg),
            _render_progress(o_bar,o_msg,ss.p_shared,msg)
        ),
        update_msg=lambda m: (
            _render_progress(g_bar,g_msg,ss.p_shared,m),
            _render_progress(o_bar,o_msg,ss.p_shared,m)
        ),
        gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
        raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
        persist_dir=persist_dir,
        manifest_path=getattr(settings, "MANIFEST_PATH", "/tmp/my_ai_teacher/drive_manifest.json"),
        max_docs=(max_docs if fast else None),
        is_cancelled=_is_cancelled,
    )

    # C. 스텝 루프
    while True:
        if _is_cancelled():
            re_mod.cancel_index_builder()
            ss.prep_both_running=False; ss.prep_cancel_requested=False
            _render_progress(g_bar,g_msg,ss.p_shared,"사용자 취소")
            _render_progress(o_bar,o_msg,ss.p_shared,"사용자 취소")
            st.stop()
        state = re_mod.resume_index_builder()
        for b in state["bursts"]:
            pct, msg = b.get("pct", ss.p_shared), b.get("msg")
            _render_progress(g_bar,g_msg,_bump_max("p_shared",pct),msg)
            _render_progress(o_bar,o_msg,ss.p_shared,msg)
        if state.get("done"): break
        time.sleep(0.2)

    index = state["index"]

    # D. LLM 두 개 준비
    try:
        g_llm = re_mod.make_llm("google", settings.GEMINI_API_KEY.get_secret_value(),
                                getattr(settings, "LLM_MODEL","gemini-1.5-pro"),
                                float(ss.get("temperature",0.0)))
        ss["llm_google"] = g_llm
        ss["qe_google"] = index.as_query_engine(
            llm=g_llm,
            response_mode=ss.get("response_mode", getattr(settings,"RESPONSE_MODE","compact")),
            similarity_top_k=int(ss.get("similarity_top_k", getattr(settings,"SIMILARITY_TOP_K",5))),
        ); _render_progress(g_bar,g_msg,100,"완료!")
    except Exception as e:
        _render_progress(g_bar,g_msg,100,f"Gemini 준비 실패: {e}")

    try:
        if getattr(settings,"OPENAI_API_KEY",None) and settings.OPENAI_API_KEY.get_secret_value():
            o_llm = re_mod.make_llm("openai", settings.OPENAI_API_KEY.get_secret_value(),
                                    getattr(settings,"OPENAI_LLM_MODEL","gpt-4o-mini"),
                                    float(ss.get("temperature",0.0)))
            ss["llm_openai"] = o_llm
            ss["qe_openai"] = index.as_query_engine(
                llm=o_llm,
                response_mode=ss.get("response_mode", getattr(settings,"RESPONSE_MODE","compact")),
                similarity_top_k=int(ss.get("similarity_top_k", getattr(settings,"SIMILARITY_TOP_K",5))),
            ); _render_progress(o_bar,o_msg,100,"완료!")
        else:
            _render_progress(o_bar,o_msg,100,"키 누락 — OPENAI_API_KEY 필요")
    except Exception as e:
        _render_progress(o_bar,o_msg,100,f"ChatGPT 준비 실패: {e}")

    ss.prep_both_running=False; ss.prep_both_done=True

left, right = st.columns([0.7,0.3])
with left:
    clicked = st.button("🚀 한 번에 준비하기", key="prepare_both",
                        use_container_width=True, disabled=ss.prep_both_running or ss.prep_both_done)
with right:
    cancel_clicked = st.button("⛔ 준비 취소", key="cancel_prepare",
                               use_container_width=True, type="secondary",
                               disabled=not ss.prep_both_running)

if cancel_clicked and ss.prep_both_running:
    ss.prep_cancel_requested=True; st.rerun()
if clicked and not (ss.prep_both_running or ss.prep_both_done):
    ss.p_shared=0; ss.prep_cancel_requested=False; ss.prep_both_running=True
    run_prepare_both_step(); st.experimental_rerun()  # 한번 더 안정 리프레시

st.caption("준비 버튼을 다시 활성화하려면 아래 재설정 버튼을 누르세요.")
if st.button("🔧 재설정(버튼 다시 활성화)", disabled=not ss.prep_both_done):
    ss.prep_both_done=False; ss.p_shared=0; st.rerun()

# ===== 대화 UI =====
st.markdown("---"); st.subheader("💬 그룹토론 (학생 → Gemini → ChatGPT)")

ready_google = "qe_google" in ss
ready_openai = "qe_openai" in ss
if ss.session_terminated:
    st.warning("세션이 종료된 상태입니다. 새로고침으로 다시 시작하세요."); st.stop()
if not ready_google:
    st.info("먼저 **[🚀 한 번에 준비하기]** 를 클릭해 두뇌를 준비하세요. (OpenAI 키가 없으면 Gemini만 응답)")
    st.stop()

# 기존 메시지 렌더
for m in ss.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

def _strip_sources(text: str) -> str:
    return re.sub(r"\n+---\n\*참고 자료:.*$", "", text, flags=re.DOTALL)

def _build_context_for_models(messages: list[dict], limit_pairs: int = 2, max_chars: int = 2000) -> str:
    pairs, buf_user = [], None
    for m in reversed(messages):
        role, content = m.get("role"), str(m.get("content","")).strip()
        if role == "assistant":
            content = re.sub(r"^\*\*🤖 .*?\*\*\s*\n+","",content).strip()
            if buf_user is not None:
                pairs.append((buf_user, content)); buf_user=None
                if len(pairs) >= limit_pairs: break
        elif role == "user" and buf_user is None:
            buf_user = content
    pairs=list(reversed(pairs))
    blocks=[f"[학생]\n{u}\n\n[교사]\n{a}" for u,a in pairs]
    ctx="\n\n---\n\n".join(blocks).strip()
    return ctx[-max_chars:] if len(ctx)>max_chars else ctx

# JSONL 저장(지속 상태 표시)
def _log_try(items):
    if not ss.save_logs: return
    try:
        parent_id = (getattr(settings,"CHATLOG_FOLDER_ID",None) or settings.GDRIVE_FOLDER_ID)
        sa = settings.GDRIVE_SERVICE_ACCOUNT_JSON
        if isinstance(sa,str):
            try: sa=json.loads(sa)
            except Exception: pass
        sub_id = get_chatlog_folder_id(parent_folder_id=parent_id, sa_json=sa)
        chat_store.append_jsonl(folder_id=sub_id, sa_json=sa, items=items)
        ss.log_status="ok"; ss.log_error=""
    except Exception as e:
        ss.log_status="err"; ss.log_error=str(e)
    _render_log_status()

# ===== 입력 처리 =====
user_input = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
if user_input:
    ss.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"): st.markdown(user_input)
    _log_try([chat_store.make_entry(ss.session_id, "user", "user", user_input, "group", model="user")])

    # 1) Gemini (친절한 친구)
    with st.spinner("🤖 Gemini(착하고 똑똑한 친구)가 먼저 답합니다…"):
        prev_ctx = _build_context_for_models(ss.messages[:-1], limit_pairs=2, max_chars=2000)
        gemini_query = (f"[이전 대화]\n{prev_ctx}\n\n" if prev_ctx else "") + f"[학생 질문]\n{user_input}"
        gemini_system = EXPLAINER_PROMPT + "\n" + GEMINI_TONE + """
[팀플레이 출력형식]
- 본 답변
- 넘겨받을_포인트: 다음 화자에게 넘기고 싶은 핵심 2~3줄
"""
        ans_g = re_mod.get_text_answer(ss["qe_google"], gemini_query, gemini_system)
    content_g = f"**🤖 Gemini**\n\n{ans_g}"
    ss.messages.append({"role":"assistant","content":content_g})
    with st.chat_message("assistant"): st.markdown(content_g)
    _log_try([chat_store.make_entry(ss.session_id,"assistant","Gemini",content_g,"group",
                                    model=getattr(settings,"LLM_MODEL","gemini"))])

    # 2) ChatGPT (유머러스한 친구)
    if ready_openai:
        from src.rag_engine import llm_complete
        prev_ctx = _build_context_for_models(ss.messages, limit_pairs=2, max_chars=2000)
        augmented = (
            (f"[이전 대화]\n{prev_ctx}\n\n" if prev_ctx else "") +
            f"[학생 질문]\n{user_input}\n\n"
            f"[동료의 1차 답변(Gemini)]\n{_strip_sources(ans_g)}\n\n"
            f"[당신의 작업]\n위 기준(REVIEW_DIRECTIVE)에 따라 보완/검증하라."
        )
        chatgpt_system = EXPLAINER_PROMPT + "\n" + CHATGPT_TONE + "\n" + REVIEW_DIRECTIVE
        with st.spinner("🤝 ChatGPT(유머러스한 친구)가 보완/검증 중…"):
            ans_o = llm_complete(ss.get("llm_openai"), chatgpt_system + "\n\n" + augmented)
        content_o = f"**🤖 ChatGPT**\n\n{ans_o}"
        ss.messages.append({"role":"assistant","content":content_o})
        with st.chat_message("assistant"): st.markdown(content_o)
        _log_try([chat_store.make_entry(ss.session_id,"assistant","ChatGPT",content_o,"group",
                                        model=getattr(settings,"OPENAI_LLM_MODEL","gpt-4o-mini"))])
    else:
        with st.chat_message("assistant"):
            st.info("ChatGPT 키가 없어 Gemini만 응답했습니다. OPENAI_API_KEY를 추가하면 보완/검증이 활성화됩니다.")

    # Markdown 대화 로그 저장(지속 상태)
    if ss.auto_save_chatlog and ss.messages:
        try:
            parent_id = (getattr(settings,"CHATLOG_FOLDER_ID",None) or settings.GDRIVE_FOLDER_ID)
            sa = settings.GDRIVE_SERVICE_ACCOUNT_JSON
            if isinstance(sa,str):
                try: sa=json.loads(sa)
                except Exception: pass
            save_chatlog_markdown(ss.session_id, ss.messages, parent_folder_id=parent_id, sa_json=sa)
            ss.log_status="ok"; ss.log_error=""
        except Exception as e:
            ss.log_status="err"; ss.log_error=str(e)
        _render_log_status()
    st.rerun()
