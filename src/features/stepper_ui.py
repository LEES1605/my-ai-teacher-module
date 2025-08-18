# ===== [F02] STEPPER UI ======================================================
import streamlit as st

def ensure_progress_css():
    st.markdown(
        """
        <style>
        .gp-msg { margin: 4px 0 10px 0; font-size: 0.96rem; }
        .stepper { display:flex; gap: 12px; align-items:center; margin: 6px 0 10px 0; }
        .step { display:flex; align-items:center; gap:8px; opacity:.55 }
        .step--active { opacity: 1 }
        .step--done { opacity: .9 }
        .step-dot{width:10px;height:10px;border-radius:999px;background:#cbd5e1}
        .step--active .step-dot{background:#6366f1}
        .step--done .step-dot{background:#10b981}
        .step-label{font-size:.9rem}
        .step-line{width:22px;height:2px;background:#e2e8f0;border-radius:999px}
        .progress-wrap { position: sticky; top: 0; z-index:5; background: transparent; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_stepper(slot, steps, status_dict, sticky=False):
    html = ['<div class="stepper{}">'.format(" progress-wrap" if sticky else "")]
    for i, (k, label) in enumerate(steps):
        cls = status_dict.get(k, "pending")
        cls_txt = "step--active" if cls == "active" else ("step--done" if cls == "done" else "")
        html.append(f"""
        <div class="step {cls_txt}">
          <div class="step-dot"></div>
          <div class="step-label">{label}</div>
          {'' if i == len(steps)-1 else "<div class='step-line'></div>"}
        </div>
        """)
    html.append("</div>")
    slot.markdown("".join(html), unsafe_allow_html=True)

def render_progress(slot, pct:int):
    pct = max(0, min(100, int(pct)))
    with slot:
        st.progress(pct)
