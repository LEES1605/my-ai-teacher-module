# src/ui.py
import streamlit as st
import base64
from src.config import BRAND_COLOR, TITLE_TEXT, TITLE_SIZE_REM, LOGO_HEIGHT_PX

@st.cache_data(show_spinner=False)
def get_img_as_base64(file: str) -> str:
    try:
        with open(file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

def load_css():
    """assets/style.css를 읽어 전역 CSS 주입."""
    try:
        with open("assets/style.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ assets/style.css 파일을 찾을 수 없어요.")

def render_header():
    """로고 + 타이틀 헤더 렌더링."""
    logo_b64 = get_img_as_base64("assets/academy_logo.png")
    st.markdown(
        f"""
        <style>
        .brand-title{{font-size:{TITLE_SIZE_REM}rem!important;color:{BRAND_COLOR};}}
        .brand-logo{{height:{LOGO_HEIGHT_PX}px!important;width:auto;object-fit:contain;display:block;}}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="header-bar">
          <div class="header-left">
            <img src="data:image/png;base64,{logo_b64}" alt="logo" class="brand-logo" />
            <h1 class="brand-title">{TITLE_TEXT}</h1>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
