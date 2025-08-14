# app.py — 두 엔진(🧠Gemini / 🧠ChatGPT) 준비 + 선택 답변 UI

import streamlit as st
import pandas as pd
import time

# ===== 페이지 설정 ============================================================
st.set_page_config(page_title="나의 AI 영어 교사", layout="wide", initial_sidebar_state="collapsed")

# ===== 기본 UI/스타일 =========================================================
from src.ui import load_css, render_header
load_css()
render_header()

st.info("✅ 베이스라인 확인용 화면입니다. 이제부터 Gemini/ChatGPT 두 엔진을 각각 준비하고, 답변 시 선택할 수 있어요.")

# ===== Google Drive 연결 테스트 ===============================================
# 임포트 실패 시 상세 오류 보여주기
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
    st.success(msg) if ok else st.warning(msg)

# ===== 두뇌 준비 (시뮬레이션) ==================================================
st.markdown("----")
st.subheader("🧠 두뇌 준비 (시뮬레이션)")
if st.button("두뇌 준비 시뮬레이션 시작"):
    bar_slot = st.empty(); msg_slot = st.empty()
    def render_progress(pct: int, msg: str | None = None):
        p = max(0, min(100, int(pct)))
        bar_slot.markdown(f"""
<div class="gp-wrap"><div class="gp-fill" style="width:{p}%"></div><div class="gp-label">{p}%</div></div>
""", unsafe_allow_html=True)
        if msg: msg_slot.markdown(f"<div class='gp-msg'>{msg}</div>", unsafe_allow_html=True)
    render_progress(5, "시작…"); time.sleep(0.2)
    render_progress(25, "비밀키 점검…")
    missing = [k for k in ("GEMINI_API_KEY", "GDRIVE_FOLDER_ID") if not str(st.secrets.get(k, "")).strip()]
    if missing: render_progress(100, "실패"); st.error("필수 Secrets 없음: " + ", ".join(missing)); st.stop()
    render_progress(60, "환경 준비…"); time.sleep(0.2)
    render_progress(90, "마무리…"); time.sleep(0.2)
    render_progress(100, "완료!"); time.sleep(0.2)
    bar_slot.empty(); msg_slot.empty()
    st.success("시뮬레이션 완료 — UI/진행 흐름 정상입니다.")

# ===== 두뇌 준비 (실전) — Gemini / ChatGPT 각각 준비 ============================
st.markdown("----")
st.subheader("🧠 두뇌 준비 (실전) — Gemini & ChatGPT")

from src.config import settings
try:
    from src.rag_engine import init_llama_settings, get_or_build_index, get_text_answer
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

# 공급자별 기본 모델 (secrets 없을 때 안전 기본값)
DEFAULTS = {
    "google": {"llm": "gemini-1.5-pro", "embed": "text-embedding-004"},
    "openai": {"llm": "gpt-4o-mini",     "embed": "text-embedding-3-small"},
}

def build_brain(provider: str):
    provider = provider.lower()
    bar = st.empty(); msg = st.empty()
    _render_progress(bar, msg, 0, f"{provider.title()} 두뇌 준비 시작…")

    # 모델 선택 (secrets에 LLM_MODEL/EMBED_MODEL가 있으면 참고하되, 공급자에 맞지 않으면 기본값 사용)
    if provider == "google":
        api_key = settings.GEMINI_API_KEY.get_secret_value()
        llm_model = DEFAULTS["google"]["llm"] if "gemini" not in getattr(settings, "LLM_MODEL", "") else settings.LLM_MODEL
        embed_model = DEFAULTS["google"]["embed"] if "embedding" not in getattr(settings, "EMBED_MODEL", "") else settings.EMBED_MODEL
        persist_dir = f"{getattr(settings, 'PERSIST_DIR', '/tmp/my_ai_teacher/storage_gdrive')}_google"
    else:
        api_key = getattr(settings, "OPENAI_API_KEY", None).get_secret_value() if hasattr(settings, "OPENAI_API_KEY") else ""
        llm_model = getattr(settings, "OPENAI_LLM_MODEL", DEFAULTS["openai"]["llm"])
        embed_model = getattr(settings, "OPENAI_EMBED_MODEL", DEFAULTS["openai"]["embed"])
        persist_dir = f"{getattr(settings, 'PERSIST_DIR', '/tmp/my_ai_teacher/storage_gdrive')}_openai"

    if not api_key:
        _render_progress(bar, msg, 100, "키 누락")
        st.error(f"{provider.title()} API 키가 없습니다. secrets.toml을 확인하세요.")
        st.stop()

    # LLM/임베딩 초기화 (임베딩은 Settings에 설정, LLM 객체는 반환)
    try:
        llm = init_llama_settings(
            provider=provider,
            api_key=api_key,
            llm_model=llm_model,
            embed_model=embed_model,
            temperature=float(st.session_state.get("temperature", 0.0)),
        )
    except Exception as e:
        _render_progress(bar, msg, 100, "LLM/임베딩 설정 오류")
        st.error(f"LLM/임베딩 초기화 중 오류: {e}")
        st.stop()

    # 인덱스 로딩/빌드
    try:
        progress = {"pct": 0}
        def update_pct(pct: int, m: str | None = None):
            progress["pct"] = int(pct); _render_progress(bar, msg, progress["pct"], m)
        def update_msg(m: str):
            _render_progress(bar, msg, progress["pct"], m)

        index = get_or_build_index(
            update_pct=update_pct,
            update_msg=update_msg,
            gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
            raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
            persist_dir=persist_dir,
            manifest_path=getattr(settings, "MANIFEST_PATH", "/tmp/my_ai_teacher/drive_manifest.json"),
        )
    except Exception as e:
        _render_progress(bar, msg, 100, "인덱스 준비 실패")
        st.error("인덱스 준비 중 오류가 발생했습니다. 폴더 권한/네트워크/requirements를 확인하세요.")
        with st.expander("오류 상세 보기"):
            st.exception(e)
        st.stop()

    # 공급자별 QueryEngine 생성(해당 LLM을 명시 주입)
    qe = index.as_query_engine(
        llm=llm,
        response_mode=st.session_state.get("response_mode", getattr(settings, "RESPONSE_MODE", "compact")),
        similarity_top_k=int(st.session_state.get("similarity_top_k", getattr(settings, "SIMILARITY_TOP_K", 5))),
    )

    # 세션에 저장
    key = "qe_google" if provider == "google" else "qe_openai"
    st.session_state[key] = qe

    _render_progress(bar, msg, 100, "완료!")
    time.sleep(0.2); bar.empty(); msg.empty()
    st.success(f"{provider.title()} 두뇌 준비 완료!")

# 버튼 2개 (좌: Gemini / 우: ChatGPT)
c1, c2 = st.columns(2)
with c1:
    if st.button("🧠 Gemini 두뇌 준비", use_container_width=True):
        build_brain("google")
with c2:
    if st.button("🧠 ChatGPT 두뇌 준비", use_container_width=True):
        build_brain("openai")

# ===== 대화 UI — 답변할 AI 선택 후 질문 ========================================
st.markdown("---")
st.subheader("💬 대화")

# 준비 상태 안내
ready_google = "qe_google" in st.session_state
ready_openai = "qe_openai" in st.session_state
if not (ready_google or ready_openai):
    st.info("먼저 위에서 **Gemini** 또는 **ChatGPT** 중 하나 이상을 준비해 주세요.")
    st.stop()

# 대화 기록
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 메시지 렌더
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 어떤 AI로 답변 받을지 선택
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

    # 프롬프트 선택
    system_prompt = EXPLAINER_PROMPT if mode == "💬 이유문법 설명" else (ANALYST_PROMPT if mode == "🔎 구문 분석" else READER_PROMPT)

    # 선택된 엔진
    qe = st.session_state.get("qe_google" if answer_with == "Gemini" else "qe_openai")
    if qe is None:
        st.warning(f"{answer_with} 두뇌가 아직 준비되지 않았어요. 위에서 먼저 준비 버튼을 눌러주세요.")
    else:
        with st.spinner(f"{answer_with}가 답변을 생각하고 있어요..."):
            answer = get_text_answer(qe, user_input, system_prompt)

        # 어느 AI인지 라벨링하여 출력
        label = "🤖 Gemini" if answer_with == "Gemini" else "🤖 ChatGPT"
        content = f"**{label}**\n\n{answer}"
        st.session_state.messages.append({"role": "assistant", "content": content})
        with st.chat_message("assistant"): st.markdown(content)

    st.rerun()
