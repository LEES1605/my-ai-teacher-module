# app.py — 인덱싱 1회 + 두 LLM(Gemini/ChatGPT) 준비 + 각자 진행바
#          ▶ 실행 중/완료 후 버튼 비활성화 + 진행률 단조증가(되돌림 방지)
#          ▶ 항상 👥 그룹토론: 사용자 → Gemini(1차) → ChatGPT(보완/검증)
#          ▶ '⛔ 준비 취소' 버튼으로만 중단 가능(실수 클릭으로 중단 X)
#          ▶ '⏹ 세션 종료' 버튼으로 앱 사용 중에도 안전 종료

import streamlit as st
import pandas as pd
import time
import re

# ===== 페이지 설정 ============================================================
st.set_page_config(page_title="나의 AI 영어 교사", layout="wide", initial_sidebar_state="collapsed")

# ===== 기본 UI/스타일 =========================================================
from src.ui import load_css, render_header
load_css()
render_header()

st.info("✅ 인덱싱은 1번만 수행하고, 그 인덱스로 Gemini/ChatGPT 두 LLM을 준비합니다. (빠른 모드·진행률 되돌림 방지·항상 👥 그룹토론)")

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
    from src.rag_engine import (
        set_embed_provider, make_llm, get_or_build_index, get_text_answer, CancelledError
    )
except Exception:
    st.error("`src.rag_engine` 임포트(LLM/RAG) 단계에서 오류가 발생했습니다.")
    import traceback
    with st.expander("임포트 스택(원인)", expanded=True):
        st.code(traceback.format_exc())
    st.stop()

# ▶ 세션 상태 기본값
if "prep_both_running" not in st.session_state:
    st.session_state.prep_both_running = False
if "prep_both_done" not in st.session_state:
    st.session_state.prep_both_done = ("qe_google" in st.session_state) or ("qe_openai" in st.session_state)
if "p_shared" not in st.session_state:
    st.session_state.p_shared = 0  # 진행률 최대값(단조증가)
if "prep_cancel_requested" not in st.session_state:
    st.session_state.prep_cancel_requested = False
if "session_terminated" not in st.session_state:
    st.session_state.session_terminated = False

# ▶ '세션 종료' 버튼 (언제든 누르면 세션 종료)
with st.container():
    colx, coly = st.columns([0.75, 0.25])
    with coly:
        if st.button("⏹ 세션 종료", use_container_width=True, type="secondary"):
            st.session_state.session_terminated = True
            st.session_state.prep_both_running = False
            st.session_state.prep_cancel_requested = False
            st.warning("세션이 종료되었습니다. 페이지를 새로고침하면 다시 시작합니다.")
            st.stop()

# ▶ 진행률 렌더
def _render_progress(slot_bar, slot_msg, pct: int, msg: str | None = None):
    p = max(0, min(100, int(pct)))
    slot_bar.markdown(f"""
<div class="gp-wrap"><div class="gp-fill" style="width:{p}%"></div><div class="gp-label">{p}%</div></div>
""", unsafe_allow_html=True)
    if msg is not None:
        slot_msg.markdown(f"<div class='gp-msg'>{msg}</div>", unsafe_allow_html=True)

# ▶ 진행률 단조증가(되돌림 방지)
def _bump_max(key: str, pct: int) -> int:
    now = int(pct)
    prev = int(st.session_state.get(key, 0))
    if now < prev:
        now = prev
    st.session_state[key] = now
    return now

# ▶ 옵션(빠른 모드) — 실행 중/완료 시 비활성화
with st.expander("⚙️ 옵션", expanded=False):
    fast = st.checkbox("⚡ 빠른 준비 (처음 N개 문서만 인덱싱)", value=True,
                       disabled=st.session_state.prep_both_running or st.session_state.prep_both_done)
    max_docs = st.number_input("N (빠른 모드일 때만 적용)", min_value=5, max_value=500, value=40, step=5,
                               disabled=st.session_state.prep_both_running or st.session_state.prep_both_done)

st.markdown("### 🚀 인덱싱 1번 + 두 LLM 준비")
c_g, c_o = st.columns(2)
with c_g: st.caption("Gemini 진행"); g_bar = st.empty(); g_msg = st.empty()
with c_o: st.caption("ChatGPT 진행"); o_bar = st.empty(); o_msg = st.empty()

def _is_cancelled() -> bool:
    """준비 중 사용자가 '⛔ 준비 취소'를 눌렀는지 확인"""
    return bool(st.session_state.get("prep_cancel_requested", False))

def run_prepare_both():
    """공통 인덱스 1회 + 두 LLM 준비. 오직 '⛔ 준비 취소'로만 중단 가능."""
    # 0) 초기 메시지
    _render_progress(g_bar, g_msg, st.session_state.p_shared, "대기 중…")
    _render_progress(o_bar, o_msg, st.session_state.p_shared, "대기 중…")

    # 내부 체크 함수
    def _check_cancel():
        if _is_cancelled():
            raise CancelledError("사용자 취소")

    # 1) 임베딩 공급자 결정
    embed_provider = "openai"
    embed_api = getattr(settings, "OPENAI_API_KEY", None).get_secret_value() if hasattr(settings, "OPENAI_API_KEY") else ""
    embed_model = getattr(settings, "OPENAI_EMBED_MODEL", "text-embedding-3-small")
    if not embed_api:
        embed_provider = "google"
        embed_api = settings.GEMINI_API_KEY.get_secret_value()
        embed_model = getattr(settings, "EMBED_MODEL", "text-embedding-004")

    persist_dir = f"{getattr(settings, 'PERSIST_DIR', '/tmp/my_ai_teacher/storage_gdrive')}_shared"

    # 2) 임베딩 설정 (양쪽 바 동시 갱신)
    try:
        _check_cancel()
        p = _bump_max("p_shared", 5)
        _render_progress(g_bar, g_msg, p, f"임베딩 설정({embed_provider})")
        _render_progress(o_bar, o_msg, p, f"임베딩 설정({embed_provider})")
        set_embed_provider(embed_provider, embed_api, embed_model)
    except CancelledError:
        # 취소 처리
        st.session_state.prep_both_running = False
        st.session_state.prep_cancel_requested = False
        _render_progress(g_bar, g_msg, st.session_state.p_shared, "사용자 취소")
        _render_progress(o_bar, o_msg, st.session_state.p_shared, "사용자 취소")
        st.stop()
    except Exception as e:
        p = _bump_max("p_shared", 100)
        _render_progress(g_bar, g_msg, p, f"임베딩 설정 실패: {e}")
        _render_progress(o_bar, o_msg, p, f"임베딩 설정 실패: {e}")
        st.session_state.prep_both_running = False
        st.stop()

    # 3) 인덱스 로딩/빌드 (공통 1회)
    try:
        def upd(pct: int, msg: str | None = None):
            if _is_cancelled():
                raise CancelledError("사용자 취소(진행 중)")
            p = _bump_max("p_shared", pct)
            _render_progress(g_bar, g_msg, p, msg)
            _render_progress(o_bar, o_msg, p, msg)

        def umsg(m: str):
            p = st.session_state.p_shared
            _render_progress(g_bar, g_msg, p, m)
            _render_progress(o_bar, o_msg, p, m)

        index = get_or_build_index(
            update_pct=upd,
            update_msg=umsg,
            gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
            raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
            persist_dir=persist_dir,
            manifest_path=getattr(settings, "MANIFEST_PATH", "/tmp/my_ai_teacher/drive_manifest.json"),
            max_docs=(max_docs if fast else None),
            is_cancelled=_is_cancelled,  # 🔴 취소 콜백 전달
        )
    except CancelledError:
        st.session_state.prep_both_running = False
        st.session_state.prep_cancel_requested = False
        _render_progress(g_bar, g_msg, st.session_state.p_shared, "사용자 취소")
        _render_progress(o_bar, o_msg, st.session_state.p_shared, "사용자 취소")
        st.stop()
    except Exception as e:
        p = _bump_max("p_shared", 100)
        _render_progress(g_bar, g_msg, p, f"인덱스 실패: {e}")
        _render_progress(o_bar, o_msg, p, f"인덱스 실패: {e}")
        st.session_state.prep_both_running = False
        st.stop()

    # 4) LLM 두 개 준비
    # 4-1) Gemini
    try:
        _check_cancel()
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
    except CancelledError:
        st.session_state.prep_both_running = False
        st.session_state.prep_cancel_requested = False
        _render_progress(g_bar, g_msg, st.session_state.p_shared, "사용자 취소")
        st.stop()
    except Exception as e:
        _render_progress(g_bar, g_msg, 100, f"Gemini 준비 실패: {e}")

    # 4-2) ChatGPT
    try:
        if hasattr(settings, "OPENAI_API_KEY") and settings.OPENAI_API_KEY.get_secret_value():
            _check_cancel()
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
    except CancelledError:
        st.session_state.prep_both_running = False
        st.session_state.prep_cancel_requested = False
        _render_progress(o_bar, o_msg, st.session_state.p_shared, "사용자 취소")
        st.stop()
    except Exception as e:
        _render_progress(o_bar, o_msg, 100, f"ChatGPT 준비 실패: {e}")

    # 5) 완료 처리: 버튼 영구 비활성화 (재설정 전까지)
    st.session_state.prep_both_running = False
    st.session_state.prep_both_done = True
    time.sleep(0.2)
    st.rerun()

# ▶ 버튼/취소 버튼 UI
left, right = st.columns([0.7, 0.3])
with left:
    clicked = st.button(
        "🚀 한 번에 준비하기",
        key="prepare_both",
        use_container_width=True,
        disabled=st.session_state.prep_both_running or st.session_state.prep_both_done,
    )
with right:
    # 준비 중일 때만 '⛔ 준비 취소' 버튼 노출 (실수 클릭으로는 취소되지 않음!)
    cancel_clicked = st.button(
        "⛔ 준비 취소",
        key="cancel_prepare",
        use_container_width=True,
        type="secondary",
        disabled=not st.session_state.prep_both_running,
    )

if cancel_clicked and st.session_state.prep_both_running:
    st.session_state.prep_cancel_requested = True
    st.rerun()

if clicked and not (st.session_state.prep_both_running or st.session_state.prep_both_done):
    st.session_state.p_shared = 0  # 진행률 최대값 리셋
    st.session_state.prep_cancel_requested = False
    st.session_state.prep_both_running = True
    st.rerun()

# ▶ 플래그가 True면 실제 준비 루틴 수행
if st.session_state.prep_both_running:
    run_prepare_both()

# ▶ 재설정(다시 준비 허용)
st.caption("준비 버튼을 다시 활성화하려면 아래 재설정 버튼을 누르세요.")
if st.button("🔧 재설정(버튼 다시 활성화)", disabled=not st.session_state.prep_both_done):
    st.session_state.prep_both_done = False
    st.session_state.p_shared = 0
    st.experimental_rerun()

# ===== 대화 UI — 항상 👥 그룹토론 ==============================================
st.markdown("---")
st.subheader("💬 그룹토론 대화 (사용자 → Gemini 1차 → ChatGPT 보완/검증)")

ready_google = "qe_google" in st.session_state
ready_openai = "qe_openai" in st.session_state

if st.session_state.session_terminated:
    st.warning("세션이 종료된 상태입니다. 페이지 새로고침(Ctrl/⌘+Shift+R)으로 다시 시작하세요.")
    st.stop()

if not ready_google:
    st.info("먼저 위의 **[🚀 한 번에 준비하기]**를 클릭해 두뇌를 준비하세요. (OpenAI 키가 없으면 Gemini만 응답)")
    st.stop()

# 대화 기록
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 메시지 렌더
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 페르소나(학습 모드)
from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
mode = st.radio("학습 모드", ["💬 이유문법 설명", "🔎 구문 분석", "📚 독해 및 요약"], horizontal=True, key="mode_select")

def _persona():
    return EXPLAINER_PROMPT if mode == "💬 이유문법 설명" else (ANALYST_PROMPT if mode == "🔎 구문 분석" else READER_PROMPT)

def _strip_sources(text: str) -> str:
    """우리 get_text_answer가 뒤에 붙이는 '*참고 자료: ...*' 꼬리를 제거."""
    return re.sub(r"\n+---\n\*참고 자료:.*$", "", text, flags=re.DOTALL)

# 입력창 (항상 그룹토론 실행)
user_input = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
if user_input:
    # 0) 사용자 메시지
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 1) Gemini 1차 답변
    with st.spinner("🤖 Gemini 선생님이 먼저 답변합니다…"):
        ans_g = get_text_answer(st.session_state["qe_google"], user_input, _persona())
    content_g = f"**🤖 Gemini**\n\n{ans_g}"
    st.session_state.messages.append({"role": "assistant", "content": content_g})
    with st.chat_message("assistant"):
        st.markdown(content_g)

    # 2) ChatGPT 보완/검증 (있을 때)
    if ready_openai:
        review_directive = (
            "당신은 동료 AI 교사입니다. 아래 [학생 질문]과 [동료의 1차 답변]을 읽고 "
            "부족한 부분을 보완/교정하고 예시를 추가한 뒤, 마지막에 '최종 정리'를 제시하세요. "
            "가능하면 근거(자료 파일명)를 유지하거나 보강하세요."
        )
        augmented_question = (
            f"[학생 질문]\n{user_input}\n\n"
            f"[동료의 1차 답변(Gemini)]\n{_strip_sources(ans_g)}"
        )
        with st.spinner("🤝 ChatGPT 선생님이 보완/검증 중…"):
            ans_o = get_text_answer(st.session_state["qe_openai"],
                                    augmented_question,
                                    _persona() + "\n" + review_directive)
        content_o = f"**🤖 ChatGPT**\n\n{ans_o}"
        st.session_state.messages.append({"role": "assistant", "content": content_o})
        with st.chat_message("assistant"):
            st.markdown(content_o)
    else:
        with st.chat_message("assistant"):
            st.info("ChatGPT 키가 없어 Gemini만 응답했습니다. OPENAI_API_KEY를 추가하면 보완/검증이 활성화됩니다.")

    st.rerun()
