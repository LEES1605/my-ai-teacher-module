# ===== [F05] ADMIN PANELS & QUICKBAR ========================================
import os, streamlit as st
from src.config import settings, PERSIST_DIR, MANIFEST_PATH, QUALITY_REPORT_PATH, CHECKPOINT_PATH
from src.features.quality_report import render_quality_report_view
from src.features.build_flow import build_or_resume_workflow
from src.features.drive_card import render_drive_check_card
from src.patches.overrides import FEATURE_FLAGS, RESPONSE_MODE_HELP, PRESETS, STATE_KEYS

def _clamp(v, lo, hi):
    try: v = int(v)
    except Exception: v = lo
    return max(lo, min(hi, v))

def render_admin_panels():
    with st.expander("⚙️ 고급 RAG/LLM 설정", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.setdefault("similarity_top_k", getattr(settings, "SIMILARITY_TOP_K", 5))
            k = st.slider("similarity_top_k", 1, 12, int(st.session_state["similarity_top_k"]))
        with col2:
            st.session_state.setdefault("temperature", 0.0)
            temp = st.slider("LLM temperature", 0.0, 1.0, float(st.session_state["temperature"]), 0.05)
        with col3:
            st.session_state.setdefault("response_mode", settings.RESPONSE_MODE)
            mode_sel = st.selectbox(
                "response_mode", ["compact","refine","tree_summarize"],
                index=["compact","refine","tree_summarize"].index(st.session_state["response_mode"]),
                help=RESPONSE_MODE_HELP
            )
        if st.button("적용"):
            st.session_state["similarity_top_k"] = int(k)
            st.session_state["temperature"] = float(temp)
            st.session_state["response_mode"] = str(mode_sel)
            st.success("RAG/LLM 설정이 저장되었습니다. (다음 쿼리부터 반영)")

    if FEATURE_FLAGS.get("SHOW_DRIVE_CARD", True):
        render_drive_check_card()

    with st.expander("🧩 최적화 설정(전처리/청킹/중복제거)", expanded=True):
        # 프리셋 버튼
        c1, c2, c3 = st.columns(3)
        keys = list(PRESETS.keys())
        if c1.button(keys[0]): _apply_profile(PRESETS[keys[0]]); st.toast("프리셋 적용!", icon="⚡"); st.rerun()
        if c2.button(keys[1]): _apply_profile(PRESETS[keys[1]]); st.toast("프리셋 적용!", icon="🔁"); st.rerun()
        if c3.button(keys[2]): _apply_profile(PRESETS[keys[2]]); st.toast("프리셋 적용!", icon="🔎"); st.rerun()

        st.caption(_current_opt_summary())
        st.divider()

        # 수동 조정
        c1, c2, c3 = st.columns(3)
        with c1:
            cs_min, cs_max = 200, 2000
            co_min, co_max = 0, 400
            mc_min, mc_max = 50, 3000
            st.session_state.setdefault("opt_chunk_size", getattr(settings, "CHUNK_SIZE", 1024))
            st.session_state.setdefault("opt_chunk_overlap", getattr(settings, "CHUNK_OVERLAP", 80))
            st.session_state.setdefault("opt_min_chars", getattr(settings, "MIN_CHARS_PER_DOC", 120))
            cs = st.number_input("청크 크기(문자)", min_value=cs_min, max_value=cs_max,
                                 value=_clamp(st.session_state["opt_chunk_size"], cs_min, cs_max), step=50)
            co = st.number_input("청크 오버랩(문자)", min_value=co_min, max_value=co_max,
                                 value=_clamp(st.session_state["opt_chunk_overlap"], co_min, co_max), step=10)
            mc = st.number_input("문서 최소 길이(문자)", min_value=mc_min, max_value=mc_max,
                                 value=_clamp(st.session_state["opt_min_chars"], mc_min, mc_max), step=50)
        with c2:
            st.session_state.setdefault("opt_dedup", getattr(settings, "DEDUP_BY_TEXT_HASH", True))
            dd = st.toggle("텍스트 해시로 중복 제거", value=bool(st.session_state["opt_dedup"]))
        with c3:
            st.session_state.setdefault("opt_skip_low_text", getattr(settings, "SKIP_LOW_TEXT_DOCS", True))
            st.session_state.setdefault("opt_pre_summarize", getattr(settings, "PRE_SUMMARIZE_DOCS", False))
            slt = st.toggle("저품질(짧은/빈약) 문서 스킵", value=bool(st.session_state["opt_skip_low_text"]))
            psu = st.toggle("문서 요약 메타데이터 생성(느려짐)", value=bool(st.session_state["opt_pre_summarize"]))
        if st.button("최적화 설정 적용"):
            st.session_state["opt_chunk_size"] = int(cs)
            st.session_state["opt_chunk_overlap"] = int(co)
            st.session_state["opt_min_chars"] = int(mc)
            st.session_state["opt_dedup"] = bool(dd)
            st.session_state["opt_skip_low_text"] = bool(slt)
            st.session_state["opt_pre_summarize"] = bool(psu)
            st.success("최적화 설정이 저장되었습니다. 다음 인덱싱부터 적용됩니다.")

    with st.expander("🛠️ 관리자 도구", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📥 강의 자료 다시 불러오기 (두뇌 초기화)"):
                import shutil
                if os.path.exists(PERSIST_DIR): shutil.rmtree(PERSIST_DIR)
                for p in (CHECKPOINT_PATH, MANIFEST_PATH, QUALITY_REPORT_PATH):
                    try:
                        if os.path.exists(p): os.remove(p)
                    except Exception: pass
                if "query_engine" in st.session_state: del st.session_state["query_engine"]
                st.session_state.pop(STATE_KEYS.BUILD_PAUSED, None)
                st.success("두뇌 파일이 초기화되었습니다.")
        with c2:
            from src.rag_engine import _normalize_sa, _validate_sa, export_brain_to_drive, prune_old_backups, INDEX_BACKUP_PREFIX
            from src.features.drive_card import get_effective_gdrive_folder_id
            try:
                if st.button("⬆️ 두뇌 저장본 드라이브로 내보내기(날짜 포함)"):
                    creds = _validate_sa(_normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON))
                    dest = getattr(settings, "BACKUP_FOLDER_ID", None) or get_effective_gdrive_folder_id()
                    with st.spinner("ZIP 업로드 중..."):
                        _, file_name = export_brain_to_drive(creds, PERSIST_DIR, dest, filename=None)
                    st.success(f"업로드 완료! 파일명: {file_name}")
                    prune_old_backups(creds, dest, keep=int(getattr(settings, "BACKUP_KEEP_N", 5)), prefix=INDEX_BACKUP_PREFIX)
            except Exception as e:
                st.error("내보내기 실패"); 
                with st.expander("자세한 오류 보기"): st.exception(e)
        with c3:
            from src.rag_engine import _normalize_sa, _validate_sa, try_restore_index_from_drive
            from src.features.drive_card import get_effective_gdrive_folder_id
            try:
                if st.button("⬇️ 드라이브에서 최신 백업 가져오기"):
                    creds = _validate_sa(_normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON))
                    dest = getattr(settings, "BACKUP_FOLDER_ID", None) or get_effective_gdrive_folder_id()
                    with st.spinner("복원 중..."):
                        ok = try_restore_index_from_drive(creds, PERSIST_DIR, dest)
                    st.success("복원 완료!" if ok else "백업 ZIP을 찾지 못했습니다.")
            except Exception as e:
                st.error("가져오기 실패"); 
                with st.expander("자세한 오류 보기"): st.exception(e)

    with st.expander("🔎 인덱스 상태 진단", expanded=False):
        st.write(f"• 로컬 저장 경로: `{PERSIST_DIR}` → {'존재' if os.path.isdir(PERSIST_DIR) else '없음'}")
        st.write(f"• 체크포인트: `{CHECKPOINT_PATH}` → {'존재' if os.path.exists(CHECKPOINT_PATH) else '없음'}")
        render_quality_report_view()

def _apply_profile(p):
    st.session_state["opt_chunk_size"]    = int(p["cs"])
    st.session_state["opt_chunk_overlap"] = int(p["co"])
    st.session_state["opt_min_chars"]     = int(p["mc"])
    st.session_state["opt_dedup"]         = bool(p["dd"])
    st.session_state["opt_skip_low_text"] = bool(p["slt"])
    st.session_state["opt_pre_summarize"] = bool(p["psu"])

def _current_opt_summary() -> str:
    return (
        f"현재: chunk **{st.session_state.get('opt_chunk_size')}**, "
        f"overlap **{st.session_state.get('opt_chunk_overlap')}**, "
        f"min_chars **{st.session_state.get('opt_min_chars')}**, "
        f"dedup **{st.session_state.get('opt_dedup')}**, "
        f"skip_low **{st.session_state.get('opt_skip_low_text')}**, "
        f"pre_summarize **{st.session_state.get('opt_pre_summarize')}**"
    )

def render_admin_quickbar():
    st.subheader("🧰 관리자 빠른 제어", divider="gray")
    col = st.columns([1])[0]
    with col:
        c1, _ = st.columns(2)
        if st.session_state.get(STATE_KEYS.BUILD_PAUSED):
            if c1.button("▶ 재개", use_container_width=True):
                st.session_state.pop(STATE_KEYS.BUILD_PAUSED, None)
                finished = build_or_resume_workflow()
                if finished: st.rerun()
        else:
            if c1.button("🧠 준비 시작", use_container_width=True):
                finished = build_or_resume_workflow()
                if finished: st.rerun()

def render_student_waiting_view():
    st.info("두뇌 준비 중입니다. 관리자가 준비를 완료하면 채팅이 활성화됩니다.")
    with st.chat_message("assistant"):
        st.markdown("안녕하세요! 곧 수업을 시작할게요. 조금만 기다려 주세요 😊")
    st.text_input("채팅은 준비 완료 후 사용 가능합니다.", disabled=True, placeholder="(잠시만 기다려 주세요)")
