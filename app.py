# app.py — 최소 동작 + Drive 테스트 + (실전) 두뇌 준비 + 챗 UI

import streamlit as st
import pandas as pd  # 링크 컬럼 표시용 DataFrame
import time

# ===== 페이지 설정(항상 최상단) ================================================
st.set_page_config(
    page_title="나의 AI 영어 교사",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== 기본 UI/스타일 =========================================================
from src.ui import load_css, render_header  # 기존 프로젝트 함수 그대로 사용
load_css()
render_header()

# (운영 전환: 디버그 박스 주석 처리)
# with st.expander("🔐 디버그: 현재 보이는 secrets 키"):
#     st.write(sorted(st.secrets.keys()))

st.info("✅ 베이스라인 확인용 화면입니다. 이 화면이 보이면 모듈 구조가 정상입니다.")
st.write("이제 여기서부터 RAG/Drive/관리자 기능을 단계적으로 붙여갑니다.")

# ===== Google Drive 연결 테스트 ===============================================
# ✅ 진단용: rag_engine 임포트 실패 시 실제 에러를 화면에 표시
try:
    from src.rag_engine import smoke_test_drive, preview_drive_files
except Exception:
    st.error("`src.rag_engine` 임포트에 실패했습니다. 아래 상세 오류를 참고하세요.")
    import os, traceback
    st.write("파일 존재 여부:", os.path.exists("src/rag_engine.py"))
    with st.expander("임포트 스택(원인)", expanded=True):
        st.code(traceback.format_exc())
    st.stop()

st.markdown("## 🔗 Google Drive 연결 테스트")
st.caption("버튼을 눌러 Drive 폴더 연결이 정상인지 확인하세요. 먼저 Secrets 설정과 폴더 공유(서비스계정 이메일 Viewer 이상)가 필요합니다.")

col1, col2 = st.columns([0.65, 0.35])  # 왼쪽(표) 공간을 넓게
with col1:
    if st.button("폴더 파일 미리보기 (최신 10개)", use_container_width=True):
        ok, msg, rows = preview_drive_files(max_items=10)
        if ok:
            if rows:
                # rows → DataFrame으로 변환하고, 열 순서/폭 최적화
                df = pd.DataFrame(rows)
                # 긴 MIME을 짧은 유형으로 변환
                df["type"] = df["mime"].str.replace("application/vnd.google-apps.", "", regex=False)
                df = df.rename(columns={"modified": "modified_at"})
                # 열 순서를 '파일명, 열기, 유형, 수정시각'으로 (열기가 앞쪽에 보이도록)
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
            else:
                st.warning("폴더에 파일이 없거나 접근할 수 없습니다.")
        else:
            st.error(msg)
            with st.expander("문제 해결 가이드"):
                st.write("- `GDRIVE_FOLDER_ID` / 서비스계정 JSON(secrets) 값을 확인하세요.")
                st.write("- Google Drive에서 **서비스계정 이메일(client_email)** 에 폴더 ‘보기 권한’을 공유하세요.")
                st.write("- `requirements.txt`에 Drive 관련 라이브러리를 추가하고 다시 배포하세요.")

with col2:
    ok, msg = smoke_test_drive()
    if ok:
        st.success(msg)
    else:
        st.warning(msg)
# === /Google Drive 연결 테스트 ===

# ===== 두뇌 준비 (시뮬레이션) ==================================================
st.markdown("----")
st.subheader("🧠 두뇌 준비 (시뮬레이션)")

start_sim = st.button("두뇌 준비 시뮬레이션 시작")
if start_sim:
    # 진행바 슬롯
    bar_slot = st.empty()
    msg_slot = st.empty()

    def render_progress(pct: int, msg: str | None = None):
        p = max(0, min(100, int(pct)))
        bar_slot.markdown(f"""
<div class="gp-wrap">
  <div class="gp-fill" style="width:{p}%"></div>
  <div class="gp-label">{p}%</div>
</div>
""", unsafe_allow_html=True)
        if msg is not None:
            msg_slot.markdown(f"<div class='gp-msg'>{msg}</div>", unsafe_allow_html=True)

    # 1) 시작
    render_progress(5, "시작…")
    time.sleep(0.3)

    # 2) Secrets 간단 점검(값 존재 여부만 확인)
    render_progress(25, "비밀키 점검…")
    missing = []
    for k in ("GEMINI_API_KEY", "GDRIVE_FOLDER_ID"):
        if k not in st.secrets or not str(st.secrets[k]).strip():
            missing.append(k)
    if missing:
        render_progress(100, "실패")
        st.error("필수 Secrets가 없습니다: " + ", ".join(missing))
        st.stop()

    # 3) 의존성/환경 체크(가벼운 수면으로 시뮬레이션)
    render_progress(60, "환경 준비…")
    time.sleep(0.4)

    # 4) 저장/정리 시뮬레이션
    render_progress(90, "마무리…")
    time.sleep(0.3)

    # 5) 완료
    render_progress(100, "완료!")
    time.sleep(0.4)
    bar_slot.empty(); msg_slot.empty()
    st.success("시뮬레이션 완료 — UI/진행 흐름 정상입니다.")

# ===== 두뇌 준비 (실전) + 챗 UI ===============================================
st.markdown("----")
st.subheader("🧠 두뇌 준비 (실전) & 대화")

# 필요한 엔진/설정 유틸들
from src.config import settings  # ← import 단순화 (상수는 settings.*로 접근)

# ✅ 진단용: rag_engine 임포트 실패 시 상세 오류 표시
try:
    from src.rag_engine import init_llama_settings, get_or_build_index, get_text_answer
except Exception:
    st.error("`src.rag_engine` 임포트(LLM/RAG 유틸) 단계에서 오류가 발생했습니다.")
    import os, traceback
    st.write("파일 존재 여부:", os.path.exists("src/rag_engine.py"))
    with st.expander("임포트 스택(원인)", expanded=True):
        st.code(traceback.format_exc())
    st.stop()

from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT

# 진행 표시용 공통 함수(시뮬레이션과 동일 UI)
def _render_progress(slot_bar, slot_msg, pct: int, msg: str | None = None):
    p = max(0, min(100, int(pct)))
    slot_bar.markdown(f"""
<div class="gp-wrap">
  <div class="gp-fill" style="width:{p}%"></div>
  <div class="gp-label">{p}%</div>
</div>
""", unsafe_allow_html=True)
    if msg is not None:
        slot_msg.markdown(f"<div class='gp-msg'>{msg}</div>", unsafe_allow_html=True)

# 1) 아직 query_engine이 없으면 준비 버튼 노출
if "query_engine" not in st.session_state:
    st.info("AI 교사를 시작하려면 아래 버튼을 눌러주세요. 처음에는 학습량에 따라 시간이 소요될 수 있습니다.")

    if st.button("🧠 AI 두뇌 준비 시작하기", key="start_brain_real"):
        bar_slot = st.empty()
        msg_slot = st.empty()
        _render_progress(bar_slot, msg_slot, 0, "두뇌 준비를 시작합니다…")

        # LLM/임베딩 설정 (키 점검 포함)
        try:
            init_llama_settings(
                api_key=settings.GEMINI_API_KEY.get_secret_value(),
                llm_model=settings.LLM_MODEL,
                embed_model=settings.EMBED_MODEL,
                temperature=float(st.session_state.get("temperature", 0.0)),
            )
        except Exception as e:
            _render_progress(bar_slot, msg_slot, 100, "LLM/임베딩 설정 오류")
            st.error(f"LLM/임베딩 초기화 중 오류: {e}")
            st.stop()

        # 인덱스 로딩/빌드
        try:
            # session_state 대신 가변 컨테이너로 현재 진행률 공유
            progress = {"pct": 0}

            def update_pct(pct: int, msg: str | None = None):
                progress["pct"] = int(pct)
                _render_progress(bar_slot, msg_slot, progress["pct"], msg)

            def update_msg(msg: str):
                _render_progress(bar_slot, msg_slot, progress["pct"], msg)

            index = get_or_build_index(
                update_pct=update_pct,
                update_msg=update_msg,
                gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
                raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
                persist_dir=getattr(settings, "PERSIST_DIR", "/tmp/my_ai_teacher/storage_gdrive"),
                manifest_path=getattr(settings, "MANIFEST_PATH", "/tmp/my_ai_teacher/drive_manifest.json"),
            )


        except Exception as e:
            _render_progress(bar_slot, msg_slot, 100, "인덱스 준비 실패")
            st.error("인덱스 준비 중 오류가 발생했습니다. 폴더 권한/네트워크/requirements를 확인하세요.")
            with st.expander("오류 상세 보기"):
                st.exception(e)
            st.stop()

        # 질의 엔진 준비
        st.session_state.query_engine = index.as_query_engine(
            response_mode=st.session_state.get("response_mode", getattr(settings, "RESPONSE_MODE", "compact")),
            similarity_top_k=int(st.session_state.get("similarity_top_k", getattr(settings, "SIMILARITY_TOP_K", 5))),
        )

        _render_progress(bar_slot, msg_slot, 100, "완료!")
        time.sleep(0.4)
        bar_slot.empty(); msg_slot.empty()
        st.rerun()

    # 버튼을 누르지 않았다면 여기서 종료(아래 챗 UI 미노출)
    st.stop()

# 2) === 여기부터 챗 UI =========================================================
# 대화 기록 상태
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 메시지 렌더
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

st.markdown("---")

# 모드 선택
mode = st.radio(
    "모드를 선택하세요",
    ["💬 이유문법 설명", "🔎 구문 분석", "📚 독해 및 요약"],
    horizontal=True,
    key="mode_select",
)

# 입력창
user_input = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
if user_input:
    # 유저 메시지 출력
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 프롬프트 선택
    if mode == "💬 이유문법 설명":
        system_prompt = EXPLAINER_PROMPT
    elif mode == "🔎 구문 분석":
        system_prompt = ANALYST_PROMPT
    else:
        system_prompt = READER_PROMPT

    # 답변 생성
    with st.spinner("AI 선생님이 답변을 생각하고 있어요..."):
        answer = get_text_answer(st.session_state.query_engine, user_input, system_prompt)

    # 어시스턴트 메시지 출력
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.rerun()
