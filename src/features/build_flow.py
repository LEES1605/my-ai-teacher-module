# ===== [F04] BUILD FLOW ======================================================
import time, os, streamlit as st
from src.config import settings, PERSIST_DIR, MANIFEST_PATH, CHECKPOINT_PATH
from src.rag_engine import (
    get_or_build_index, init_llama_settings, _load_index_from_disk,
    try_restore_index_from_drive, export_brain_to_drive, prune_old_backups,
    _normalize_sa, _validate_sa, INDEX_BACKUP_PREFIX
)
from src.features.stepper_ui import render_stepper, render_progress
from src.features.drive_card import get_effective_gdrive_folder_id
from src.patches.overrides import (
    FEATURE_FLAGS, STEPPER_LABELS, STEPPER_KEYWORDS, STEPPER_DONE,
    STATE_KEYS, on_after_backup, on_after_finish
)

def _secret_or_str(v):
    try: return v.get_secret_value()
    except Exception: return str(v)

def build_or_resume_workflow() -> bool:
    # --- UI 준비
    stepper_slot = st.empty(); bar_slot = st.empty(); msg_slot = st.empty(); ctrl_slot = st.empty()
    steps = [(k, STEPPER_LABELS[k]) for k in ["check","init","list","index","save"]]
    st.session_state["_step_status"] = {k:"pending" for k,_ in steps}
    st.session_state["_step_curr"] = None

    def _advance_to(key:str):
        order = [k for k,_ in steps]
        cur  = st.session_state.get("_step_curr")
        if cur is None or order.index(key) >= order.index(cur):
            st.session_state["_step_status"][key] = "active"
            if cur and st.session_state["_step_status"].get(cur) == "active":
                st.session_state["_step_status"][cur] = "done"
            st.session_state["_step_curr"] = key
            render_stepper(stepper_slot, steps, st.session_state["_step_status"], sticky=True)

    def _set_done_all():
        for k,_ in steps: st.session_state["_step_status"][k] = "done"
        st.session_state["_step_curr"] = steps[-1][0]
        render_stepper(stepper_slot, steps, st.session_state["_step_status"], sticky=True)

    _advance_to("check"); render_progress(bar_slot, 0)
    msg_slot.markdown("<div class='gp-msg'>두뇌 준비를 시작합니다…</div>", unsafe_allow_html=True)

    # --- 중지 버튼
    st.session_state[STATE_KEYS.STOP_REQUESTED] = False
    with ctrl_slot.container():
        c1, c2 = st.columns([1,1])
        with c1: st.caption("진행 제어")
        with c2:
            if st.button("🛑 학습 중지", type="secondary"):
                st.session_state[STATE_KEYS.STOP_REQUESTED] = True
                st.info("중지 요청됨 — 현재 파일까지 마무리하고 곧 멈춥니다.")

    # --- 진행 업데이트 콜백
    st.session_state["_gp_pct"] = 0
    def update_pct(pct:int, msg:str|None=None):
        p = max(0, min(100, int(pct)))
        st.session_state["_gp_pct"] = p
        if   p < 10: _advance_to("check")
        elif p < 25: _advance_to("init")
        elif p < 50: _advance_to("list")
        elif p < 95: _advance_to("index")
        else:        _advance_to("save")
        render_progress(bar_slot, p)
        if msg is not None: update_msg(msg)

    def update_msg(text:str):
        t = (text or "").lower()
        for key, kws in STEPPER_KEYWORDS.items():
            if any(k in t for k in kws): _advance_to(key); break
        if any(k in t for k in STEPPER_DONE): _set_done_all()
        msg_slot.markdown(f"<div class='gp-msg'>{text}</div>", unsafe_allow_html=True)

    def should_stop() -> bool:
        return bool(st.session_state.get(STATE_KEYS.STOP_REQUESTED, False))

    # --- 1) 모델 준비
    init_llama_settings(
        api_key=_secret_or_str(settings.GEMINI_API_KEY),
        llm_model=settings.LLM_MODEL,
        embed_model=settings.EMBED_MODEL,
        temperature=float(st.session_state.get("temperature", 0.0)),
    )

    # --- 2) 인덱스 빌드 (체크포인트 + 중지 신호)
    index = get_or_build_index(
        update_pct=update_pct, update_msg=update_msg,
        gdrive_folder_id=get_effective_gdrive_folder_id(),
        raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
        persist_dir=PERSIST_DIR, manifest_path=MANIFEST_PATH,
        should_stop=should_stop
    )

    # --- 중지 시 ‘재개’ 모드로 전환
    if st.session_state.get(STATE_KEYS.STOP_REQUESTED):
        st.session_state[STATE_KEYS.BUILD_PAUSED] = True
        st.warning("학습을 중지했습니다. **‘▶ 재개’ 버튼으로** 이어서 학습할 수 있어요.")
        return False

    # --- 3) 쿼리 엔진 연결
    st.session_state.query_engine = index.as_query_engine(
        response_mode=st.session_state.get("response_mode", settings.RESPONSE_MODE),
        similarity_top_k=int(st.session_state.get("similarity_top_k", getattr(settings, "SIMILARITY_TOP_K", 5))),
    )

    # --- 4) 완료 + 자동 백업
    update_pct(100, "완료!"); time.sleep(0.4)
    if FEATURE_FLAGS.get("AUTO_BACKUP_ON_SUCCESS", True):
        try:
            creds = _validate_sa(_normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON))
            dest = getattr(settings, "BACKUP_FOLDER_ID", None) or get_effective_gdrive_folder_id()
            with st.spinner("⬆️ 인덱스 저장본을 드라이브로 자동 백업 중..."):
                _, file_name = export_brain_to_drive(creds, PERSIST_DIR, dest, filename=None)
            st.success(f"자동 백업 완료! 파일명: {file_name}")
            on_after_backup(file_name)
            prune_old_backups(creds, dest, keep=int(getattr(settings, "BACKUP_KEEP_N", 5)), prefix=INDEX_BACKUP_PREFIX)
        except Exception as e:
            st.warning("자동 백업 실패(로컬 저장본은 OK).")
            with st.expander("백업 오류 보기"): st.exception(e)

    on_after_finish()
    return True
