# app.py — 최소 동작 확인용(모듈 구조/스타일 로딩/헤더만)
import streamlit as st

st.set_page_config(
    page_title="나의 AI 영어 교사",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from src.ui import load_css, render_header

# 전역 스타일 로드 + 헤더 렌더링
load_css()
render_header()

st.info("✅ 베이스라인 확인용 화면입니다. 이 화면이 보이면 모듈 구조가 정상입니다.")
st.write("이제 여기서부터 RAG/Drive/관리자 기능을 단계적으로 붙여갑니다.")

# --- 여기서부터 '두뇌 준비(시뮬레이션)' 블록 추가 ---
import time

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
# --- '두뇌 준비(시뮬레이션)' 블록 끝 ---
