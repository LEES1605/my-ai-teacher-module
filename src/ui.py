# src/ui.py — 공통 UI

from __future__ import annotations
import streamlit as st
from src.config import BRAND_COLOR, TITLE_TEXT, TITLE_SIZE_REM, LOGO_HEIGHT_PX

CSS = f"""
<style>
:root {{
  --brand: {BRAND_COLOR};
}}
.gp-wrap {{
  position: relative; height: 24px; background: #f2f2f2; border-radius: 6px; overflow: hidden;
}}
.gp-fill {{ height: 100%; background: var(--brand); transition: width .2s ease; }}
.gp-label {{
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  display:flex; align-items:center; justify-content:center; font-weight:600;
}}
.gp-msg {{ margin: 6px 0 0 0; color:#555; font-size: 0.9rem; }}
.header-title {{ font-size: {TITLE_SIZE_REM}rem; font-weight: 800; color: var(--brand); margin: 0 0 6px 0; }}
</style>
"""

def load_css():
    st.markdown(CSS, unsafe_allow_html=True)

def render_header():
    st.markdown(f"<div class='header-title'>{TITLE_TEXT}</div>", unsafe_allow_html=True)
