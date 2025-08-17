# src/ui.py
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit 공용 UI 유틸:
#  - load_css: assets/style.css + 브랜드 변수 주입 + (선택) 배경 이미지 적용
#  - safe_render_header: 로고/타이틀 헤더 출력(로고 없으면 타이틀만)
#  - ensure_progress_css: 진행바 전용 CSS 주입
#  - render_progress_bar: 커스텀 진행바 표시
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import os, base64
from typing import Optional
import streamlit as st
from src.config import settings

# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────
def _read_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None

def _img_to_data_uri(path: str) -> Optional[str]:
    b = _read_bytes(path)
    if not b:
        return None
    enc = base64.b64encode(b).decode("ascii")
    # 확장자에 따라 MIME 추정
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext in (".png",) else "image/jpeg"
    return f"data:{mime};base64,{enc}"

# ── 공개 API ────────────────────────────────────────────────────────────────
def load_css(css_path: str, use_bg: bool = False, bg_path: Optional[str] = None) -> None:
    """
    - assets/style.css를 로드하고 :root CSS 변수(브랜드 색/사이즈)를 주입
    - use_bg=True이고 배경 이미지가 있으면 앱 배경에 적용
    """
    # 기본 CSS 파일
    css_bytes = _read_bytes(css_path)
    if css_bytes:
        st.markdown(f"<style>{css_bytes.decode('utf-8')}</style>", unsafe_allow_html=True)
    else:
        # 최소 폴백
        st.markdown(
            """
            <style>
            :root { --brand:#9067C6; --title-size:3rem; --logo-h:110px; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # 브랜드 변수 주입(런타임 값 반영)
    st.markdown(
        f"""
        <style>
        :root {{
          --brand: {settings.BRAND_COLOR};
          --title-size: {settings.TITLE_SIZE_REM}rem;
          --logo-h: {int(settings.LOGO_HEIGHT_PX)}px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # (선택) 배경 이미지
    if use_bg and bg_path:
        uri = _img_to_data_uri(bg_path)
        if uri:
            st.markdown(
                f"""
                <style>
                [data-testid="stAppViewContainer"] > .main {{
                    background-image: url("{uri}");
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

def safe_render_header(logo_path: str = "assets/logo.png") -> None:
    """
    상단 헤더: 로고(있으면) + 타이틀
    - 화면 폭이 줄면 로고가 왼쪽, 타이틀이 오른쪽/아래로 반응형 정렬
    """
    logo_uri = _img_to_data_uri(logo_path)
    if logo_uri:
        html = f"""
        <div class="brand-header">
          <div class="brand-logo">
            <img src="{logo_uri}" alt="logo" />
          </div>
          <div class="brand-title-wrap">
            <h1 class="brand-title">{settings.TITLE_TEXT}</h1>
          </div>
        </div>
        """
    else:
        html = f"""
        <div class="brand-header no-logo">
          <div class="brand-title-wrap">
            <h1 class="brand-title">{settings.TITLE_TEXT}</h1>
          </div>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)

def ensure_progress_css() -> None:
    """진행바 전용 CSS를 주입(한 번만)."""
    if st.session_state.get("_gp_css_injected"):
        return
    st.markdown(
        """
        <style>
        .gp-wrap { width:100%; margin: 8px 0 4px 0; }
        .gp-bar {
            width: 100%;
            height: 12px;
            background: rgba(0,0,0,0.08);
            border-radius: 999px;
            overflow: hidden;
        }
        .gp-fill {
            height: 100%;
            width: 0%;
            background: var(--brand);
            transition: width .35s ease;
        }
        .gp-msg {
            font-size: .92rem;
            opacity: .9;
            margin-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_gp_css_injected"] = True

def render_progress_bar(slot, percent: int) -> None:
    """
    slot: st.empty() 등 컨테이너
    percent: 0~100
    """
    p = max(0, min(100, int(percent)))
    slot.markdown(
        f"""
        <div class="gp-wrap">
          <div class="gp-bar">
            <div class="gp-fill" style="width:{p}%"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── END: ui helpers
