# src/ui.py
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit 공용 UI 유틸:
#  - load_css: assets/style.css + 브랜드 변수 주입 + (선택) 배경 이미지 적용
#  - safe_render_header: 로고/타이틀 헤더 출력(로고 없으면 타이틀만)
#  - ensure_progress_css: 진행바 전용 CSS 주입
#  - render_progress_bar: 커스텀 진행바 표시
#  - render_stepper: 단계별 상태(대기/진행/완료/오류) 시각화 (내장 CSS 폴백 포함)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import os, base64, html
from typing import Optional, Iterable, Mapping
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
    import base64 as _b64
    enc = _b64.b64encode(b).decode("ascii")
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext in (".png",) else "image/jpeg"
    return f"data:{mime};base64,{enc}"

# ── 공개 API ────────────────────────────────────────────────────────────────
def load_css(css_path: str, use_bg: bool = False, bg_path: Optional[str] = None) -> None:
    css_bytes = _read_bytes(css_path)
    if css_bytes:
        st.markdown(f"<style>{css_bytes.decode('utf-8')}</style>", unsafe_allow_html=True)

    # 런타임 브랜드 변수 주입
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
    logo_uri = _img_to_data_uri(logo_path)
    if logo_uri:
        block = f"""
        <div class="brand-header">
          <div class="brand-logo"><img src="{logo_uri}" alt="logo" /></div>
          <div class="brand-title-wrap"><h1 class="brand-title">{html.escape(settings.TITLE_TEXT)}</h1></div>
        </div>
        """
    else:
        block = f"""
        <div class="brand-header no-logo">
          <div class="brand-title-wrap"><h1 class="brand-title">{html.escape(settings.TITLE_TEXT)}</h1></div>
        </div>
        """
    st.markdown(block, unsafe_allow_html=True)

def ensure_progress_css() -> None:
    if st.session_state.get("_gp_css_injected"):
        return
    st.markdown(
        """
        <style>
        .gp-wrap { width:100%; margin: 8px 0 4px 0; }
        .gp-bar { width: 100%; height: 12px; background: rgba(0,0,0,0.08); border-radius: 999px; overflow: hidden; }
        .gp-fill { height:100%; width:0%; background: var(--brand); transition: width .35s ease; }
        .gp-msg { font-size:.92rem; opacity:.9; margin-top:4px; }
        /* 스텝퍼가 없을 때를 대비한 최소 스타일(폴백) */
        .stepper{ display:flex; align-items:center; gap:8px; margin:8px 0 6px 0; }
        .stepper-sticky{ position:sticky; top:8px; z-index:20; background:rgba(255,255,255,0.75); backdrop-filter: blur(6px);
                          padding:6px 8px; border-radius:12px; box-shadow: 0 4px 18px rgba(0,0,0,0.05); }
        .step{ display:flex; align-items:center; gap:8px; }
        .step-label{ font-size:.92rem; white-space:nowrap; opacity:.7; }
        .step-dot{ width:16px; height:16px; border-radius:999px; border:2px solid var(--brand); background:transparent; position:relative; }
        .step-line{ width:38px; height:2px; background: linear-gradient(90deg, var(--brand) 30%, rgba(0,0,0,0.1) 30%); opacity:.35; }
        .step--active .step-dot{ background:var(--brand); }
        .step--active .step-label{ opacity:1; font-weight:600; }
        .step--done .step-dot{ background:var(--brand); }
        .step--done .step-dot::after{ content:'✓'; position:absolute; top:-3px; left:3px; font-size:12px; color:white; }
        .step--done .step-label{ opacity:.95; }
        .step--error .step-dot{ background:#d7263d; border-color:#d7263d; }
        .step--error .step-label{ opacity:1; color:#d7263d; font-weight:700; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_gp_css_injected"] = True

def render_progress_bar(slot, percent: int) -> None:
    p = max(0, min(100, int(percent)))
    slot.markdown(
        f"""
        <div class="gp-wrap">
          <div class="gp-bar"><div class="gp-fill" style="width:{p}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_stepper(slot, steps: Iterable[tuple[str, str]], status: Mapping[str, str], sticky: bool = True) -> None:
    """
    단계 진행 상황을 가시적으로 표시하는 스텝퍼
    - steps: [("check","드라이브 변경 확인"), ("init","Drive 리더 초기화"), ...]
    - status: {"check":"active|done|pending|error", ...}
    - sticky=True면 상단 고정
    """
    # 폴백 CSS가 주입되도록 보장
    ensure_progress_css()

    items = []
    last_idx = len(list(steps)) - 1
    for idx, (k, label) in enumerate(steps):
        stt = status.get(k, "pending")
        esc = html.escape(label)
        line = "<div class='step-line'></div>" if idx != last_idx else ""
        items.append(
            f"<div class='step step--{stt}'><div class='step-dot'></div><div class='step-label'>{esc}</div>{line}</div>"
        )
    wrap_cls = "stepper-sticky" if sticky else ""
    html_block = f"<div class='stepper {wrap_cls}'>" + "".join(items) + "</div>"

    # 반드시 unsafe_allow_html=True 로 렌더 (st.write/코드블록 방지)
    slot.markdown(html_block, unsafe_allow_html=True)

# ── END: ui helpers
