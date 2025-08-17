# src/auth.py
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
    if not st.session_state.get("admin_mode", False):
        return False

    if not admin_password:
        st.info("관리자 비밀번호가 설정되지 않았습니다. (개발 모드)")
        _logout_button_area()
        return True

    if st.session_state.get("admin_verified", False):
        st.success("관리자 인증됨")
        _logout_button_area()
        return True

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
