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
