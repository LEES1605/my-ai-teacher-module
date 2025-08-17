# src/ui/progress.py
import streamlit as st, time
_injected_key = "_sticky_beep_injected"
def init():
    if st.session_state.get(_injected_key): return st.session_state[_injected_key]
    st.markdown(\"\"\"<style>/* (위 CSS 동일) */</style>
    <script>/* (aiBeep 동일) */</script>\"\"\", unsafe_allow_html=True)
    holder = st.empty()
    st.session_state[_injected_key] = holder
    return holder
def update(holder, pct, done, total, filename, note=\"\", started_at=None):
    mmss = \"00:00\"
    if started_at: 
        el=int(time.time()-started_at); mm,ss_=divmod(el,60); mmss=f\"{mm:02d}:{ss_:02d}\"
    holder.markdown(f\"\"\"<div id='sticky-progress'>
      <div class='stk-row'>
        <div class='stk-bar'><div class='stk-fill' style='width:{pct}%'></div></div>
        <span class='stk-pill'>{pct}%</span>
        <span class='stk-pill'>{done}/{total}</span>
        <span class='stk-pill'>{mmss}</span>
        <span class='stk-name'>{filename}</span>
        <span>{note}</span>
      </div></div>\"\"\", unsafe_allow_html=True)
def beep(freq=880):
    st.markdown(f\"<script>aiBeep({freq});</script>\", unsafe_allow_html=True)
