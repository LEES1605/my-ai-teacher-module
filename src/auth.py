# src/auth.py
# ─────────────────────────────────────────────────────────────────────────────
# 관리자 인증 플로우:
#  - app.py 상단 우측의 🛠️ 버튼이 st.session_state.admin_mode=True 로 켠다.
#  - 여기서는 비밀번호가 있으면 안전하게 검사하고, 없으면 바로 허용.
#  - compare_digest 사용(타이밍 공격 완화)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import streamlit as st
from secrets import compare_digest

def _logout_button_area() -> None:
    c1, c2 = st.columns([0.7, 0.3])
    with c2:
        if st.button("🔒 관리자 모드 끄기", key="admin_logout"):
            st.session_state.pop("admin_verified", None)
            st.session_state["admin_mode"] = False
            st.experimental_rerun()

def admin_login_flow(admin_password: str) -> bool:
    """
    반환값:
      - True  → 관리자 패널 노출
      - False → 숨김
    """
    # 관리자 모드가 아니면 아무것도 안 보여줌
    if not st.session_state.get("admin_mode", False):
        return False

    # 비밀번호 미설정 시 바로 통과(개발/테스트)
    if not admin_password:
        st.info("관리자 비밀번호가 설정되지 않았습니다. (개발 모드)")
        _logout_button_area()
        return True

    # 이미 인증된 경우
    if st.session_state.get("admin_verified", False):
        st.success("관리자 인증됨")
        _logout_button_area()
        return True

    # 최초 인증 UI (폼)
    with st.expander("🔐 관리자 인증", expanded=True):
        with st.form("admin_login_form", clear_on_submit=False):
            pw = st.text_input("관리자 비밀번호", type="password", key="__admin_pw_input")
            ok = st.form_submit_button("확인")
        if ok:
            if compare_digest(str(pw), str(admin_password)):
                st.session_state["admin_verified"] = True
                st.success("인증 성공! 아래 관리자 패널을 사용할 수 있어요.")
                st.experimental_rerun()
            else:
                st.error("비밀번호가 올바르지 않습니다.")
    return False

# ── END: admin auth
