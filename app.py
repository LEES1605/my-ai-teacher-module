# app.py — 인덱싱 1회 + 두 LLM(Gemini/ChatGPT) 준비 + 각자 진행바
#          ▶ 버튼 클릭 후 자동 비활성화 / 완료 시 다시 활성화

import streamlit as st
import pandas as pd
import time

# ===== 페이지 설정 ============================================================
st.set_page_config(page_title="나의 AI 영어 교사", layout="wide", initial_sidebar_state="collapsed")

# ===== 기본 UI/스타일 =========================================================
from src.ui import load_css, render_header
load_css()
render_header()

st.info("✅ 이제 인덱싱은 1번만 수행하고, 그 인덱스로 Gemini/ChatGPT 두 LLM을 준비합니다. (빠른 모드 지원)")

# ===== Google Drive 연결 테스트 ===============================================
try:
    from src.rag_engine import smoke_test_drive, preview_drive_files
except Exception:
    st.error("`src.rag_engine` 임포트에 실패했습니다.")
    import traceback, os
    st.write("파일 존재 여부:", os.path.exists("src/rag_engine.py"))
    with st.expander("임포트 스택(원인)", expanded=True):
        st.code(traceback.format_exc())
    st.stop()

st.markdown("## 🔗 Google Drive 연결 테스트")
st.caption("버튼을 눌러 Drive 폴더 연결이 정상인지 확인하세요. (서비스계정에 Viewer 이상 공유 필요)")

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
                df,
                use_container_width=True,
                height=360,
                column_config={
                    "name": st.column_config.TextColumn("파일명"),
                    "link": st.column_config.LinkColumn("open", display_text="열기"),
                    "type": st.column_config.TextColumn("유형"),
                    "modified_at": st.column_config.TextColumn("수정시각"),
                },
                hide_index=True,
            )
        elif ok:
            st.warning("폴더에 파일이 없거나 접근할 수 없습니다.")
        else:
            st.error(msg)
with col2:
    ok, msg = smoke_test_drive()
    if ok:
        st.success(msg)
    else:
        st.warning(msg)

# ===== 두뇌 준비 (실전) — 공통 인덱스 + LLM 2개 ===============================
st.markdown("----")
st.subheader("🧠 두뇌 준비 — 인덱스 1회 + Gemini/ChatGPT LLM")

from src.config import settings
try:
    from src.rag_engine import set_embed_provider, make_llm, get_or_build_index, get_text_answer
except Exception:
    st.error("`src.rag_engine` 임포트(LLM/RAG) 단계에서 오류가 발생했습니다.")
    import traceback
    with st.expander("임포트 스택(원인)", expanded=True):
        st.code(traceback.format_exc())
    st.stop()

def _render_progress(slot_bar, slot_msg, pct: int, msg: str | None = None):
    p = max(0, min(100, int(pct)))
    slot_bar.markdown(f"""
<div class="gp-wrap"><div class="gp-fill" style="width:{p}%"></div><div class="gp-label">{p}%</div></div>
""", unsafe_allow_html=True)
    if msg is not None:
        slot_msg.markdown(f"<div class='gp-msg'>{msg}</div>", unsafe_allow_html=True)

# 옵션: 빠른 모드
with st.expander("⚙️ 옵션", expanded=False):
    fast = st.checkbox("⚡ 빠른 준비 (처음 N개 문서만 인덱싱)", value=True)
    max_docs = st.number_input("N (빠른 모드일 때만 적용)", min_value=5, max_value=500, value=40, step=5)

# 진행바 자리 미리 확보
st.markdown("### 🚀 인덱싱 1번 + 두 LLM 준비")
c_g, c_o = st.columns(2)
with c_g: st.caption("Gemini 진행"); g_bar = st.empty(); g_msg = st.empty()
with c_o: st.caption("ChatGPT 진행"); o_bar = st.empty(); o_msg = st.empty()

# ▶ 버튼 상태를 세션에서 관리 (눌린 뒤 비활성화)
if "prep_both_running" not in st.session_state:
    st.session_state.prep_both_running = False

def run_prepare_both():
    """한 번 클릭으로 공통 인덱스 + 두 LLM을 준비. 끝나면 버튼 활성화 복구."""
    # 0) 초기 상태
    _render_progress(g_bar, g_msg, 0, "대기 중…")
    _render_progress(o_bar, o_msg, 0, "대기 중…")

    # 1) 임베딩 공급자 결정 (OpenAI 키가 있으면 가성비 빠른 OpenAI 임베딩 사용)
    embed_provider = "openai"
    embed_api = getattr(settings, "OPENAI_API_KEY", None).get_secret_value() if hasattr(settings, "OPENAI_API_KEY") else ""
    embed_model = getattr(settings, "OPENAI_EMBED_MODEL", "text-embedding-3-small")
    if not embed_api:
        embed_provider = "google"
        embed_api = settings.GEMINI_API_KEY.get_secret_value()
        embed_model = getattr(settings, "EMBED_MODEL", "text-embedding-004")

    # 2) 공통 persist 경로
    persist_dir = f"{getattr(settings, 'PERSIST_DIR', '/tmp/my_ai_teacher/storage_gdrive')}_shared"

    # 3) 임베딩 설정 (두 진행바 동시 갱신)
    try:
        _render_progress(g_bar, g_msg, 5, f"임베딩 설정({embed_provider})")
        _render_progress(o_bar, o_msg, 5, f"임베딩 설정({embed_provider})")
        set_embed_provider(embed_provider, embed_api, embed_model)
    except Exception as e:
        _render_progress(g_bar, g_msg, 100, f"임베딩 설정 실패: {e}")
        _render_progress(o_bar, o_msg, 100, f"임베딩 설정 실패: {e}")
        st.session_state.prep_both_running = False
        st.stop()

    # 4) 인덱스 로딩/빌드 (공통 1회)
    try:
        prog = {"pct": 10}
        def upd(pct: int, msg: str | None = None):
            prog["pct"] = int(pct)
            _render_progress(g_bar, g_msg, prog["pct"], msg)
            _render_progress(o_bar, o_msg, prog["pct"], msg)
        def umsg(m: str):
            _render_progress(g_bar, g_msg, prog["pct"], m)
            _render_progress(o_bar, o_msg, prog["pct"], m)

        index = get_or_build_index(
            update_pct=upd,
            update_msg=umsg,
            gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
            raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
            persist_dir=persist_dir,
            manifest_path=getattr(settings, "MANIFEST_PATH", "/tmp/my_ai_teacher/drive_manifest.json"),
            max_docs=(max_docs if fast else None),
        )
    except Exception as e:
        _render_progress(g_bar, g_msg, 100, f"인덱스 실패: {e}")
        _render_progress(o_bar, o_msg, 100, f"인덱스 실패: {e}")
        st.session_state.prep_both_running = False
        st.stop()

    # 5) LLM 두 개 준비
    # 5-1) Gemini
    try:
        g_llm = make_llm(
            provider="google",
            api_key=settings.GEMINI_API_KEY.get_secret_value(),
            llm_model=getattr(settings, "LLM_MODEL", "gemini-1.5-pro"),
            temperature=float(st.session_state.get("temperature", 0.0)),
        )
        qe_g = index.as_query_engine(
            llm=g_llm,
            response_mode=st.session_state.get("response_mode", getattr(settings, "RESPONSE_MODE", "compact")),
            similarity_top_k=int(st.session_state.get("similarity_top_k", getattr(settings, "SIMILARITY_TOP_K", 5))),
        )
        st.session_state["qe_google"] = qe_g
        _render_progress(g_bar, g_msg, 100, "완료!")
    except Exception as e:
        _render_progress(g_bar, g_msg, 100, f"Gemini 준비 실패: {e}")

    # 5-2) ChatGPT
    try:
        if hasattr(settings, "OPENAI_API_KEY") and settings.OPENAI_API_KEY.get_secret_value():
            o_llm = make_llm(
                provider="openai",
                api_key=settings.OPENAI_API_KEY.get_secret_value(),
                llm_model=getattr(settings, "OPENAI_LLM_MODEL", "gpt-4o-mini"),
                temperature=float(st.session_state.get("temperature", 0.0)),
            )
            qe_o = index.as_query_engine(
                llm=o_llm,
                response_mode=st.session_state.get("response_mode", getattr(settings, "RESPONSE_MODE", "compact")),
                similarity_top_k=int(st.session_state.get("similarity_top_k", getattr(settings, "SIMILARITY_TOP_K", 5))),
            )
            st.session_state["qe_openai"] = qe_o
            _render_progress(o_bar, o_msg, 100, "완료!")
        else:
            _render_progress(o_bar, o_msg, 100, "키 누락 — OPENAI_API_KEY 필요")
    except Exception as e:
        _render_progress(o_bar, o_msg, 100, f"ChatGPT 준비 실패: {e}")

    # 6) 버튼 다시 활성화 → UI 새로고침
    st.session_state.prep_both_running = False
    time.sleep(0.2)
    st.rerun()

# ▶ 버튼: 눌린 동안 disabled=True
clicked = st.button(
    "🚀 한 번에 준비하기",
    key="prepare_both",
    use_container_width=True,
    disabled=st.session_state.prep_both_running,
)
if clicked and not st.session_state.prep_both_running:
    st.session_state.prep_both_running = True
    st.rerun()

# ▶ 플래그가 True 면(= 방금 클릭해서 재실행된 상태) 실제 준비 루틴 수행
if st.session_state.prep_both_running:
    run_prepare_both()

# ===== 대화 UI — 답변할 AI 선택 후 질문 ========================================
st.markdown("---")
st.subheader("💬 대화")

ready_google = "qe_google" in st.session_state
ready_openai = "qe_openai" in st.session_state
if not (ready_google or ready_openai):
    st.info("먼저 위의 **[🚀 한 번에 준비하기]** 를 클릭해 주세요. (OpenAI 키가 없으면 Gemini만 준비됩니다)")
    st.stop()

# 대화 기록
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 메시지 렌더
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 답변할 AI 선택
choices = []
if ready_google: choices.append("Gemini")
if ready_openai: choices.append("ChatGPT")
answer_with = st.radio("답변할 AI 선택", choices, horizontal=True, index=0)

# 프롬프트들
from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
mode = st.radio("모드를 선택하세요", ["💬 이유문법 설명", "🔎 구문 분석", "📚 독해 및 요약"], horizontal=True, key="mode_select")

# 입력창
user_input = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    system_prompt = EXPLAINER_PROMPT if mode == "💬 이유문법 설명" else (ANALYST_PROMPT if mode == "🔎 구문 분석" else READER_PROMPT)
    qe = st.session_state.get("qe_google" if answer_with == "Gemini" else "qe_openai")
    if qe is None:
        st.warning(f"{answer_with} 두뇌가 아직 준비되지 않았어요. 위에서 먼저 준비 버튼을 눌러주세요.")
    else:
        with st.spinner(f"{answer_with}가 답변을 생각하고 있어요..."):
            answer = get_text_answer(qe, user_input, system_prompt)

        label = "🤖 Gemini" if answer_with == "Gemini" else "🤖 ChatGPT"
        content = f"**{label}**\n\n{answer}"
        st.session_state.messages.append({"role": "assistant", "content": content})
        with st.chat_message("assistant"): st.markdown(content)

    st.rerun()
