# src/features/admin_panels.py
# -----------------------------------------------------------------------------
# 관리자 패널(사이드바/퀵바) 및 학생 대기 화면
# - 기존 코드의 import 오류(PERSISI_DIR 오탈자, QUALITY_REPORT_PATH 미정의)를 제거
# - 현재 프로젝트의 config 구성(settings, PERSIST_DIR, MANIFEST_PATH)에 맞춰 최소 구현
# - 다른 모듈에 대한 강한 의존성 없이 안전하게 동작하도록 설계
# -----------------------------------------------------------------------------
from __future__ import annotations
import os
import shutil
import streamlit as st

from src.config import settings, PERSIST_DIR, MANIFEST_PATH

# 선택: 스타일 보조 (있으면 사용, 없어도 동작)
try:
    from src.features.stepper_ui import ensure_progress_css
except Exception:
    ensure_progress_css = None


def render_admin_panels() -> None:
    """사이드바에 간단한 관리자 패널을 렌더링한다.
    - 두뇌 초기화(인덱스 삭제)
    - RAG/LLM 파라미터 요약(현재 값 표시)
    - 추후 필요한 옵션은 이 파일에서 확장
    """
    st.sidebar.markdown("### 🛠️ 관리자 패널")

    # 현재 설정 요약
    with st.sidebar.expander("현재 RAG/LLM 설정", expanded=False):
        st.write(f"- response_mode: `{st.session_state.get('response_mode', settings.RESPONSE_MODE)}`")
        st.write(f"- similarity_top_k: `{int(st.session_state.get('similarity_top_k', settings.SIMILARITY_TOP_K))}`")
        st.write(f"- temperature: `{float(st.session_state.get('temperature', 0.0))}`")

    # 두뇌 초기화
    with st.sidebar.expander("위험 구역: 두뇌 초기화", expanded=False):
        st.caption("인덱스 저장 폴더를 삭제합니다. 다음 질의 전 다시 빌드가 필요합니다.")
        if st.button("📥 강의 자료 다시 불러오기 (두뇌 초기화)", key="btn_reset_brain_sidebar"):
            try:
                if os.path.exists(PERSIST_DIR):
                    shutil.rmtree(PERSIST_DIR)
                if "query_engine" in st.session_state:
                    del st.session_state["query_engine"]
                st.success("두뇌 파일이 초기화되었습니다. 메인 화면에서 다시 준비하세요.")
            except Exception as e:
                st.error("초기화 중 오류가 발생했습니다.")
                with st.expander("자세한 오류 보기", expanded=True):
                    st.exception(e)


def render_admin_quickbar() -> None:
    """하단 고정 퀵바(간단 버전). 필요 시 확장."""
    st.markdown(
        """
        <style>
        .admin-quickbar {
            position: fixed; left: 0; right: 0; bottom: 10px;
            display: flex; justify-content: center; z-index: 9999;
        }
        .admin-quickbar > div {
            background: rgba(20,20,35,0.85);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 999px;
            padding: 6px 14px;
            font-size: 0.9rem; color: #E2E8F0;
            backdrop-filter: blur(6px);
        }
        </style>
        <div class="admin-quickbar"><div>관리자 퀵바 활성화</div></div>
        """,
        unsafe_allow_html=True
    )


def render_student_waiting_view() -> None:
    """학생 대기 화면(두뇌 미준비 상태)."""
    if ensure_progress_css:
        ensure_progress_css()
    st.info(
        "AI 교사의 두뇌가 아직 준비되지 않았습니다. "
        "관리자가 강의 자료를 연결하고 인덱스를 빌드하면 수업을 시작할 수 있어요."
    )
    st.caption("관리자에게 ‘🧠 AI 두뇌 준비’를 눌러달라고 요청하세요.")
