# src/features/admin_panels.py
from __future__ import annotations
import streamlit as st
from src.config import settings

def render_admin_panels() -> None:
    """사이드바 관리자 패널: 응답 선택(자동/수동) + 현재 설정 요약만 유지."""
    st.sidebar.markdown("### 🛠️ 관리자 패널")

    # === 응답 선택(자동/수동) ===
    st.sidebar.markdown("#### 🧭 응답 선택(자동/수동)")
    st.session_state.setdefault("use_manual_override", False)
    use_manual = st.sidebar.checkbox(
        "수동 모드(관리자 오버라이드) 사용",
        value=st.session_state["use_manual_override"]
    )
    st.session_state["use_manual_override"] = use_manual

    st.session_state.setdefault("manual_prompt_mode", "explainer")
    manual_label_map = {
        "explainer": "explainer(이유문법 설명)",
        "analyst": "analyst(구문 분석)",
        "reader": "reader(독해/요약)"
    }
    selected_label = manual_label_map[st.session_state["manual_prompt_mode"]]
    manual_mode = st.sidebar.selectbox(
        "수동 모드 선택",
        list(manual_label_map.values()),
        index=list(manual_label_map.values()).index(selected_label),
        help="체크박스를 켜면 이 선택이 학생 라디오를 덮어씁니다."
    )
    for k, v in manual_label_map.items():
        if v == manual_mode:
            st.session_state["manual_prompt_mode"] = k
            break

    # === 현재 RAG/LLM 설정 요약 ===
    with st.sidebar.expander("현재 RAG/LLM 설정", expanded=False):
        st.write(f"- response_mode: `{st.session_state.get('response_mode', settings.RESPONSE_MODE)}`")
        st.write(f"- similarity_top_k: `{int(st.session_state.get('similarity_top_k', settings.SIMILARITY_TOP_K))}`")
        st.write(f"- temperature: `{float(st.session_state.get('temperature', 0.0))}`")


def render_admin_quickbar() -> None:
    """하단 고정 퀵바(간단 배지)."""
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
    st.info(
        "AI 교사의 두뇌가 아직 준비되지 않았습니다. "
        "관리자가 강의 자료를 연결하고 인덱스를 빌드하면 수업을 시작할 수 있어요."
    )
    st.caption("관리자에게 ‘AI 두뇌 준비’를 눌러달라고 요청하세요.")
