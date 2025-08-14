# app.py — 최소 동작 확인용(모듈 구조/스타일 로딩/헤더만)
# app.py — 최소 동작 확인용(모듈 구조/스타일 로딩/헤더/Drive 테스트)

import streamlit as st
import json
from collections.abc import Mapping

from src.ui import load_css, render_header
from src.rag_engine import smoke_test_drive

st.set_page_config(
    page_title="나의 AI 영어 교사",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 전역 스타일 로드 + 헤더 렌더링
load_css()
render_header()

st.info("✅ 베이스라인 확인용 화면입니다. 이 화면이 보이면 모듈 구조가 정상입니다.")
st.write("이제 여기서부터 RAG/Drive/관리자 기능을 단계적으로 붙여갑니다.")

# === 🔗 Google Drive 연결 테스트 (베이스라인 알림 '바로 아래') ===
st.markdown("## 🔗 Google Drive 연결 테스트")

col1, col2 = st.columns([0.35, 0.65])
with col1:
    if st.button("폴더 파일 미리보기 (최신 10개)"):
        # secrets에서 서비스 계정 JSON을 dict로 안전 파싱
        raw_sa = st.secrets.get("GDRIVE_SERVICE_ACCOUNT_JSON")

        if isinstance(raw_sa, Mapping):
            sa_info = dict(raw_sa)  # tables/dict 형태
        elif isinstance(raw_sa, str):
            try:
                sa_info = json.loads(raw_sa) if raw_sa else None  # 문자열 JSON 형태
            except Exception:
                sa_info = None
        else:
            sa_info = None

        folder_id = st.secrets.get("GDRIVE_FOLDER_ID")

        if not sa_info:
            st.error("❌ GDRIVE_SERVICE_ACCOUNT_JSON이 비어 있거나 형식이 잘못되었습니다.")
        elif not folder_id:
            st.error("❌ GDRIVE_FOLDER_ID가 설정되지 않았습니다.")
        else:
            try:
                files = smoke_test_drive(sa_info, folder_id, limit=10)
                if not files:
                    st.warning("⚠️ 파일이 없거나 권한이 부족합니다. (폴더 공유/권한/폴더ID 확인)")
                else:
                    st.success(f"연결 OK! {len(files)}개 미리보기")
                    for f in files:
                        name = f.get("name")
                        mime = f.get("mimeType")
                        mtime = f.get("modifiedTime", "")[:19].replace("T", " ")
                        st.write(f"• **{name}** — _{mime}, {mtime}_")
            except Exception as e:
                st.error("Google Drive 호출 중 오류가 발생했습니다.")
                st.exception(e)

with col2:
    st.info(
        "버튼을 눌러 Drive 폴더 연결이 정상인지 확인하세요. 먼저 **Secrets** 설정과 "
        "**폴더 공유(서비스 계정 이메일 Viewer 이상)** 가 필요합니다."
    )
# === /Google Drive 연결 테스트 ===


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
