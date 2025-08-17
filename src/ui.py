# src/ui.py
from __future__ import annotations
import base64
from pathlib import Path
import streamlit as st
from src.config import settings

# ---------- 내부 유틸 ----------
def _read_text(p: str) -> str:
    try:
        return Path(p).read_text(encoding="utf-8")
    except Exception:
        return ""

def _b64_image(path: str) -> str | None:
    try:
        data = Path(path).read_bytes()
        return base64.b64encode(data).decode("ascii")
    except Exception:
        return None

# ---------- CSS 로더 ----------
def load_css(css_path: str = "assets/style.css", use_bg: bool = True, bg_path: str | None = None) -> None:
    """
    앱 공통 CSS를 주입하고(파일 없으면 건너뜀),
    use_bg=True면 고정 배경 이미지를 반투명으로 깔아요.
    """
    css = _read_text(css_path)
    extra = ""

    if use_bg:
        bg_path = bg_path or settings.BG_IMAGE_PATH
        b64 = _b64_image(bg_path)
        if b64:
            extra = f"""
            .stApp::before {{
                content: "";
                position: fixed; inset: 0;
                background-image: url("data:image/png;base64,{b64}");
                background-position: center;
                background-size: cover;
                opacity: .06;
                pointer-events: none;
                z-index: -1;
            }}
            """

    st.markdown(
        f"""
        <style>
        /* 기본 스타일 */
        {css}

        /* 헤더 */
        .app-header {{
            display:flex; align-items:center; gap:12px;
            margin: 6px 0 14px 0;
        }}
        .app-header__logo {{
            width: 36px; height: 36px; border-radius: 6px; object-fit: contain;
        }}
        .app-header__titles {{
            display:flex; flex-direction:column;
        }}
        .app-header__title {{
            font-weight: 700; font-size: 1.4rem; line-height: 1.2; margin:0;
        }}
        .app-header__subtitle {{
            color:#6b7280; font-size:.95rem; margin:0;
        }}

        /* 진행바 영역 sticky */
        .progress-wrap {{ position: sticky; top: 0; z-index: 5; background: transparent; }}
        {extra}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------- 헤더(로고 + 제목/부제) ----------
def render_header(title: str, subtitle: str | None = None, logo_path: str = "assets/academy_logo.png") -> None:
    logo_b64 = _b64_image(logo_path)
    logo_html = (
        f'<img class="app-header__logo" src="data:image/png;base64,{logo_b64}" alt="logo" />'
        if logo_b64 else ""
    )
    st.markdown(
        f"""
        <div class="app-header">
           {logo_html}
           <div class="app-header__titles">
             <div class="app-header__title">{title}</div>
             {f'<div class="app-header__subtitle">{subtitle}</div>' if subtitle else ''}
           </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- 진행바 ----------
def render_progress_bar(pct: int, text: str | None = None):
    """pct: 0~100"""
    if text:
        st.write(text)
    return st.progress(max(0, min(100, int(pct))))
