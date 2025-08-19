# ===== [01] IMPORTS ==========================================================
from __future__ import annotations
import time
import streamlit as st
from typing import Any

from src.config import settings, PERSIST_DIR, MANIFEST_PATH, CHECKPOINT_PATH
from src.rag_engine import (
    get_or_build_index, init_llama_settings, _load_index_from_disk,
    try_restore_index_from_drive, export_brain_to_drive, prune_old_backups,
    _normalize_sa, _validate_sa, INDEX_BACKUP_PREFIX,
)
from src.features.stepper_ui import render_stepper, render_progress
from src.features.drive_card import get_effective_gdrive_folder_id
from src.patches.overrides import (
    FEATURE_FLAGS, STEPPER_LABELS, STEPPER_KEYWORDS, STEPPER_DONE,
    STATE_KEYS, on_after_backup, on_after_finish,
)

# ===== [02] UTILS ============================================================
def _secret_or_str(v: Any) -> str:
    """pydantic SecretStr 또는 어떤 타입이든 문자열로 안전 변환."""
    try:
        return v.get_secret_value()  # pydantic SecretStr
    except Exception:
        return "" if v is None else str(v)

def _advance_to(steps: list[tuple[str, str]], key: str) -> None:
    order = [k for k, _ in steps]
    cur = st.session_state.get("_step_curr")
    if cur is None or order.index(key) >= order.index(cur):
        st.session_state["_step_status"][key] = "active"
        if cur and st.session_state["_step_status"].get(cur) == "active":
            st.session_state["_step_status"][cur] = "done"
        st.session_state["_step_curr"] = key

def _set_done_all(steps: list[tuple[str, str]]) -> None:
    for k, _ in steps:
        st.session_state["_step_status"][k] = "done"
    st.session_state["_step_curr"] = steps[-1][0]

def export_brain_to_drive_safe(
    creds: Any, persist_dir: str, folder_id: str, filename: str | None = None
) -> tuple[str | None, str | None]:
    """
    export_brain_to_drive의 안전 래퍼: 실패해도 전체 플로우를 중단시키지 않는다.
    """
    try:
        return export_brain_to_drive(creds, persist_dir, folder_id, filename=filename)
    except Exception as e:
        st.warning("자동 백업 실패(로컬 저장본은 OK).")
        with st.expander("백업 오류 보기"):
            st.exception(e)
        return None, None

# ===== [03] MAIN WORKFLOW ====================================================
def build_or_resume_workflow() -> bool:
    # --- UI 준비
    stepper_slot = st.empty()
    bar_slot = st.empty()
    msg_slot = st.empty()
    ctrl_slot = st.empty()

    steps = [(k, STEPPER_LABELS[k]) for k in ["check", "init", "list", "index", "save"]]
    st.session_state["_step_status"] = {k: "pending" for k, _ in steps}
    st.session_state["_step_curr"] = None

    # 초기 렌더
    _advance_to(steps, "check")
    render_stepper(stepper_slot, steps, st.session_state["_step_status"], sticky=True)
    render_progress(bar_slot, 0)
    msg_slot.markdown("<div class='gp-msg'>두뇌 준비를 시작합니다…</div>", unsafe_allow_html=True)

    # --- 중지 버튼
    st.session_state[STATE_KEYS.STOP_REQUESTED] = False
    with ctrl_slot.container():
        c1, c2 = st.columns([1, 1])
        with c1:
            st.caption("진행 제어")
        with c2:
            if st.button("🛑 학습 중지", type="secondary"):
                st.session_state[STATE_KEYS.STOP_REQUESTED] = True
                st.info("중지 요청됨 — 현재 파일까지 마무리하고 곧 멈춥니다.")

    # --- 진행 업데이트 콜백
    st.session_state["_gp_pct"] = 0

    def _refresh_stepper_for(p: int) -> None:
        if p < 10:
            _advance_to(steps, "check")
        elif p < 25:
            _advance_to(steps, "init")
        elif p < 50:
            _advance_to(steps, "list")
        elif p < 95:
            _advance_to(steps, "index")
        else:
            _advance_to(steps, "save")
        render_stepper(stepper_slot, steps, st.session_state["_step_status"], sticky=True)

    def update_msg(text: str) -> None:
        t = (text or "").lower()
        for key, kws in STEPPER_KEYWORDS.items():
            if any(k in t for k in kws):
                _advance_to(steps, key)
                break
        if any(k in t for k in STEPPER_DONE):
            _set_done_all(steps)
        msg_slot.markdown(f"<div class='gp-msg'>{text}</div>", unsafe_allow_html=True)
        render_stepper(stepper_slot, steps, st.session_state["_step_status"], sticky=True)

    def update_pct(pct: int, msg: str | None = None) -> None:
        p = max(0, min(100, int(pct)))
        st.session_state["_gp_pct"] = p
        _refresh_stepper_for(p)
        render_progress(bar_slot, p)
        if msg is not None:
            update_msg(msg)

    def should_stop() -> bool:
        return bool(st.session_state.get(STATE_KEYS.STOP_REQUESTED, False))

    # --- 1) 모델 준비
    init_llama_settings(
        api_key=_secret_or_str(settings.GEMINI_API_KEY),
        llm_model=settings.LLM_MODEL,
        embed_model=settings.EMBED_MODEL,
        temperature=float(st.session_state.get("temperature", 0.0)),
    )

    # --- 2) 인덱스 빌드/복구 (체크포인트 + 중지 신호 전달)
    index = get_or_build_index(
        update_pct=update_pct,
        update_msg=update_msg,
        gdrive_folder_id=get_effective_gdrive_folder_id(),
        raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
        persist_dir=PERSIST_DIR,
        manifest_path=MANIFEST_PATH,
        should_stop=should_stop,
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

            # ⬇️ with 블록의 본문이 반드시 들여쓰기되어 있어야 함!
            with st.spinner("⬆️ 인덱스 저장본을 드라이브로 자동 백업 중..."):
                _file_id, file_name = export_brain_to_drive_safe(creds, str(PERSIST_DIR), str(dest), filename=None)

            if file_name:
                st.success(f"자동 백업 완료! 파일명: {file_name}")
                on_after_backup(file_name)
                prune_old_backups(creds, str(dest), keep=int(getattr(settings, "BACKUP_KEEP_N", 5)), prefix=INDEX_BACKUP_PREFIX)
            else:
                st.info("백업 파일이 생성되지 않았습니다(자세한 내용은 위의 '백업 오류 보기' 참조).")
        except Exception as e:
            st.warning("자동 백업 중 알 수 없는 오류가 발생했습니다.")
            with st.expander("백업 오류 보기"):
                st.exception(e)

    on_after_finish()
    return True

# ===== [04] END ==============================================================
