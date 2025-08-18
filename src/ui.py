# ===== [01] TOP OF FILE ======================================================
# src/ui.py — 공용 UI 유틸
# - load_css: 전역 CSS + (선택) 배경 이미지 인라인 적용
# - safe_render_header: 상단 로고/타이틀 헤더
# - ensure_progress_css: 진행바 스타일 주입
# - render_progress_bar: 커스텀 진행바 렌더

# ===== [02] IMPORTS ==========================================================
from __future__ import annotations
import base64
from pathlib import Path
import streamlit as st
from src.config import settings

# ===== [03] HELPERS ==========================================================
@st.cache_data(show_spinner=False)
def _read_file_text(path_str: str) -> str:
    try:
        return Path(path_str).read_text(encoding="utf-8")
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def _file_as_base64(path_str: str) -> str:
    try:
        data = Path(path_str).read_bytes()
        return base64.b64encode(data).decode()
    except Exception:
        return ""

# ===== [04] PUBLIC: load_css =================================================
def load_css(file_path: str, use_bg: bool = False, bg_path: str | None = None) -> None:
    """
    전역 스타일 로딩 + (선택) 배경 이미지 적용.
    - style.css가 실패해도 앱이 최소 가독성을 유지하도록 폴백은 app.py에서 보강.
    """
    css = _read_file_text(file_path) or ""
    bg_css = ""
    if use_bg and bg_path:
        img_b64 = _file_as_base64(bg_path)
        if img_b64:
            bg_css = f"""
            .stApp{{
              background-image: url("data:image/png;base64,{img_b64}");
              background-size: cover;
              background-position: center;
              background-repeat: no-repeat;
              background-attachment: fixed;
            }}
            """
    st.markdown(f"<style>{bg_css}\n{css}</style>", unsafe_allow_html=True)

# ===== [05] PUBLIC: safe_render_header ======================================
def safe_render_header(
    title: str | None = None,
    subtitle: str | None = None,
    logo_path: str | None = "assets/academy_logo.png",
    logo_height_px: int | None = None,
) -> None:
    """
    상단에 로고 + 타이틀/서브타이틀을 안전하게 표시.
    - settings에서 기본값을 가져오되, 파라미터가 있으면 우선.
    """
    _title = title or getattr(settings, "TITLE_TEXT", "나의 AI 영어 교사")
    _subtitle = subtitle or getattr(settings, "SUBTITLE_TEXT", "")
    _logo_h = logo_height_px or int(getattr(settings, "LOGO_HEIGHT_PX", 56))
    logo_b64 = _file_as_base64(logo_path) if logo_path else ""

    st.markdown(
        f"""
        <style>
        .aihdr-wrap{{display:flex;align-items:center;gap:14px;margin:6px 0 10px;}}
        .aihdr-logo{{height:{_logo_h}px;width:auto;object-fit:contain;display:block}}
        .aihdr-title{{font-size:{getattr(settings,'TITLE_SIZE_REM',2.0)}rem;color:{getattr(settings,'BRAND_COLOR','#F7FAFC')};margin:0}}
        .aihdr-sub{{color:#E2E8F0;margin:2px 0 0 0;}}
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.85, 0.15])
    with left:
        st.markdown(
            f"""
            <div class="aihdr-wrap">
              {'<img src="data:image/png;base64,'+logo_b64+'" class="aihdr-logo"/>' if logo_b64 else ''}
              <div>
                <h1 class="aihdr-title">{_title}</h1>
                {f'<div class="aihdr-sub">{_subtitle}</div>' if _subtitle else ''}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ===== [06] PUBLIC: ensure_progress_css =====================================
def ensure_progress_css() -> None:
    """커스텀 진행바 CSS 주입(여러 번 호출되어도 안전)."""
    st.markdown(
        """
        <style>
        .gp-wrap{ width:100%; height:28px; border-radius:12px;
          background: rgba(255,255,255,.12);
          border:1px solid rgba(255,255,255,.25);
          position:relative; overflow:hidden;
          box-shadow: 0 4px 14px rgba(0,0,0,.15);
        }
        .gp-fill{ height:100%;
          background: linear-gradient(90deg,#7c5ad9,#9067C6);
          transition: width .25s ease;
        }
        .gp-label{ position:absolute; inset:0;
          display:flex; align-items:center; justify-content:center;
          font-weight:800; color:#F7FAFC; text-shadow: 0 1px 2px rgba(0,0,0,.4);
          font-size:20px; pointer-events:none;
        }
        .gp-msg{ margin-top:.5rem; color:#F7FAFC; opacity:.9; font-size:0.95rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ===== [07] PUBLIC: render_progress_bar =====================================
def render_progress_bar(slot, pct: int) -> None:
    """slot(=st.empty())에 진행바 렌더. pct는 0~100 정수."""
    pct = max(0, min(100, int(pct)))
    slot.markdown(
        f"""
        <div class="gp-wrap">
          <div class="gp-fill" style="width:{pct}%"></div>
          <div class="gp-label">{pct}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===== [08] END OF FILE ======================================================
