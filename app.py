# app.py
# ===== TOP OF FILE ============================================================
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_RUN_ON_SAVE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

import time
import streamlit as st
import json

from src.config import settings, APP_DATA_DIR, PERSIST_DIR, MANIFEST_PATH, QUALITY_REPORT_PATH
from src.ui import (
    load_css, safe_render_header, ensure_progress_css,
    render_progress_bar, render_stepper
)
from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
from src.rag_engine import (
    get_or_build_index, init_llama_settings, get_text_answer,
    _load_index_from_disk, try_restore_index_from_drive,
    export_brain_to_drive, prune_old_backups, _normalize_sa, _validate_sa,
    INDEX_BACKUP_PREFIX, CHECKPOINT_PATH,
)
from src.auth import admin_login_flow

# ── 페이지 & 전역 스타일 ─────────────────────────────────────────────────────
st.set_page_config(page_title="나의 AI 영어 교사", layout="wide", initial_sidebar_state="collapsed")
load_css("assets/style.css", use_bg=settings.USE_BG_IMAGE, bg_path="assets/background_book.png")
ensure_progress_css()
safe_render_header()

# ── 우측 상단 관리자 아이콘 ───────────────────────────────────────────────────
_, _, c3 = st.columns([0.8, 0.1, 0.1])
with c3:
    if st.button("🛠️", key="admin_icon_top_bar"):
        st.session_state.admin_mode = True

# ── 관리자 인증 ───────────────────────────────────────────────────────────────
is_admin = admin_login_flow(settings.ADMIN_PASSWORD or "")

# ── 저장본 자동 연결/복원(무소음) ─────────────────────────────────────────────
def _auto_attach_or_restore_silently() -> bool:
    try:
        if os.path.exists(PERSIST_DIR):
            init_llama_settings(
                api_key=settings.GEMINI_API_KEY.get_secret_value(),
                llm_model=settings.LLM_MODEL,
                embed_model=settings.EMBED_MODEL,
                temperature=float(st.session_state.get("temperature", 0.0)),
            )
            index = _load_index_from_disk(PERSIST_DIR)
            st.session_state.query_engine = index.as_query_engine(
                response_mode=st.session_state.get("response_mode", settings.RESPONSE_MODE),
                similarity_top_k=int(st.session_state.get("similarity_top_k", settings.SIMILARITY_TOP_K)),
            )
            st.session_state["_auto_attach_note"] = "local_ok"
            return True
        creds = _validate_sa(_normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON))
        ok = try_restore_index_from_drive(creds, PERSIST_DIR, settings.BACKUP_FOLDER_ID or settings.GDRIVE_FOLDER_ID)
        if ok:
            index = _load_index_from_disk(PERSIST_DIR)
            st.session_state.query_engine = index.as_query_engine(
                response_mode=st.session_state.get("response_mode", settings.RESPONSE_MODE),
                similarity_top_k=int(st.session_state.get("similarity_top_k", settings.SIMILARITY_TOP_K)),
            )
            st.session_state["_auto_attach_note"] = "restored_from_drive"
            return True
        st.session_state["_auto_attach_note"] = "no_cache_no_backup"
    except Exception as e:
        st.session_state["_attach_error"] = repr(e)
    return False

if "query_engine" not in st.session_state:
    _auto_attach_or_restore_silently()

# ── 품질 리포트 뷰어 ───────────────────────────────────────────────────────────
def render_quality_report_view():
    st.subheader("📊 최적화 품질 리포트", divider="gray")
    if not os.path.exists(QUALITY_REPORT_PATH):
        st.info("아직 리포트가 없습니다. 인덱싱을 한 뒤에 자동 생성됩니다.")
        return
    try:
        with open(QUALITY_REPORT_PATH, "r", encoding="utf-8") as f:
            rep = json.load(f)
    except Exception as e:
        st.error("리포트 파일을 읽는 중 오류.")
        with st.expander("오류 보기"):
            st.exception(e)
        return

    s = rep.get("summary", {})
    st.write(
        f"- 전체 문서: **{s.get('total_docs', 0)}**개  "
        f"- 처리 파일: **{s.get('processed_docs', 0)}**개  "
        f"- 채택(kept): **{s.get('kept_docs', 0)}**개  "
        f"- 스킵(저품질): **{s.get('skipped_low_text', 0)}**개  "
        f"- 스킵(중복): **{s.get('skipped_dup', 0)}**개  "
        f"- 총 본문 문자수: **{s.get('total_chars', 0):,}**"
    )
    with st.expander("파일별 상세"):
        files = rep.get("files", {})
        rows = []
        for fid, info in files.items():
            rows.append([
                info.get("name", fid),
                info.get("kept", 0),
                info.get("skipped_low_text", 0),
                info.get("skipped_dup", 0),
                info.get("total_chars", 0),
                info.get("modifiedTime", ""),
            ])
        if rows:
            st.dataframe(
                rows,
                column_config={
                    0: "파일명", 1: "채택", 2: "저품질 스킵", 3: "중복 스킵", 4: "문자수", 5: "수정시각",
                },
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("아직 수집된 파일 통계가 없습니다.")

# ── 관리자 전용 패널 ──────────────────────────────────────────────────────────
if is_admin:
    with st.expander("⚙️ 고급 RAG/LLM 설정", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.setdefault("similarity_top_k", settings.SIMILARITY_TOP_K)
            k = st.slider("similarity_top_k", 1, 12, st.session_state["similarity_top_k"])
        with col2:
            st.session_state.setdefault("temperature", 0.0)
            temp = st.slider("LLM temperature", 0.0, 1.0, float(st.session_state["temperature"]), 0.05)
        with col3:
            st.session_state.setdefault("response_mode", settings.RESPONSE_MODE)
            mode_sel = st.selectbox(
                "response_mode", ["compact", "refine", "tree_summarize"],
                index=["compact","refine","tree_summarize"].index(st.session_state["response_mode"]),
            )
        if st.button("적용"):
            st.session_state["similarity_top_k"] = k
            st.session_state["temperature"] = temp
            st.session_state["response_mode"] = mode_sel
            st.success("RAG/LLM 설정이 저장되었습니다. (다음 쿼리부터 반영)")

    # 최적화 옵션 UI (이전과 동일)
    with st.expander("🧩 최적화 설정(전처리/청킹/중복제거)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.session_state.setdefault("opt_chunk_size", settings.CHUNK_SIZE)
            st.session_state.setdefault("opt_chunk_overlap", settings.CHUNK_OVERLAP)
            cs = st.number_input("청크 크기(문자)", 200, 2000, int(st.session_state["opt_chunk_size"]), 50)
            co = st.number_input("청크 오버랩(문자)", 0, 400, int(st.session_state["opt_chunk_overlap"]), 10)
        with c2:
            st.session_state.setdefault("opt_min_chars", settings.MIN_CHARS_PER_DOC)
            st.session_state.setdefault("opt_dedup", settings.DEDUP_BY_TEXT_HASH)
            mc = st.number_input("문서 최소 길이(문자)", 100, 3000, int(st.session_state["opt_min_chars"]), 50)
            dd = st.toggle("텍스트 해시로 중복 제거", value=bool(st.session_state["opt_dedup"]))
        with c3:
            st.session_state.setdefault("opt_skip_low_text", settings.SKIP_LOW_TEXT_DOCS)
            st.session_state.setdefault("opt_pre_summarize", settings.PRE_SUMMARIZE_DOCS)
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
                if os.path.exists(PERSIST_DIR):
                    shutil.rmtree(PERSIST_DIR)
                # 체크포인트/매니페스트/리포트도 함께 삭제
                for p in (CHECKPOINT_PATH, MANIFEST_PATH, QUALITY_REPORT_PATH):
                    try:
                        if os.path.exists(p): os.remove(p)
                    except Exception:
                        pass
                if "query_engine" in st.session_state:
                    del st.session_state["query_engine"]
                st.success("두뇌 파일이 초기화되었습니다. 아래에서 다시 준비하세요.")
        with c2:
            if st.button("⬆️ 두뇌 저장본 드라이브로 내보내기(날짜 포함)"):
                try:
                    creds = _validate_sa(_normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON))
                    dest = settings.BACKUP_FOLDER_ID or settings.GDRIVE_FOLDER_ID
                    with st.spinner("두뇌를 ZIP(날짜 포함)으로 묶고 드라이브에 업로드 중..."):
                        _, file_name = export_brain_to_drive(creds, PERSIST_DIR, dest, filename=None)
                    st.success(f"업로드 완료! 파일명: {file_name}")
                    deleted = prune_old_backups(creds, dest, keep=int(settings.BACKUP_KEEP_N), prefix=INDEX_BACKUP_PREFIX)
                    if deleted:
                        st.info(f"오래된 백업 {len(deleted)}개 정리 완료.")
                except Exception as e:
                    st.error("내보내기 실패. 두뇌가 준비되었는지와 폴더 권한(편집자)을 확인하세요.")
                    with st.expander("자세한 오류 보기"):
                        st.exception(e)
        with c3:
            if st.button("⬇️ 드라이브에서 최신 백업 가져오기"):
                try:
                    creds = _validate_sa(_normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON))
                    dest = settings.BACKUP_FOLDER_ID or settings.GDRIVE_FOLDER_ID
                    with st.spinner("드라이브에서 최신 백업 ZIP을 내려받아 복원 중..."):
                        ok = try_restore_index_from_drive(creds, PERSIST_DIR, dest)
                    if ok:
                        st.success("복원 완료! 아래에서 두뇌를 연결하거나 대화를 시작하세요.")
                    else:
                        st.warning("백업 ZIP을 찾지 못했습니다. 먼저 내보내기를 해주세요.")
                except Exception as e:
                    st.error("가져오기 실패. 폴더 권한(편집자)과 파일 존재 여부를 확인하세요.")
                    with st.expander("자세한 오류 보기"):
                        st.exception(e)

    with st.expander("🔎 인덱스 상태 진단", expanded=False):
        st.write(f"• 로컬 저장 경로: `{PERSIST_DIR}` → {'존재' if os.path.isdir(PERSIST_DIR) else '없음'}")
        st.write(f"• 체크포인트: `{CHECKPOINT_PATH}` → {'존재' if os.path.exists(CHECKPOINT_PATH) else '없음'}")
        render_quality_report_view()

# ── 메인 워크플로우 ───────────────────────────────────────────────────────────
def main():
    # 준비되어 있으면 채팅으로 바로 진입
    if "query_engine" in st.session_state:
        render_chat_ui()
        return

    # 두뇌가 없고, 관리자만 준비 UI를 봄
    if is_admin:
        st.info("AI 교사를 준비하려면 아래 버튼을 누르세요. (체크포인트/중지 버튼 지원)")

        if st.button("🧠 AI 두뇌 준비 시작하기"):
            # 진행 UI 슬롯
            stepper_slot = st.empty(); bar_slot = st.empty(); msg_slot = st.empty(); ctrl_slot = st.empty()

            steps = [("check","드라이브 변경 확인"),("init","Drive 리더 초기화"),
                     ("list","문서 목록 불러오는 중"),("index","인덱스 생성"),("save","두뇌 저장")]
            st.session_state["_step_status"] = {k:"pending" for k,_ in steps}
            st.session_state["_step_curr"] = None

            def _set_active(key:str):
                prev = st.session_state.get("_step_curr")
                if prev and st.session_state["_step_status"].get(prev)=="active":
                    st.session_state["_step_status"][prev] = "done"
                st.session_state["_step_status"][key] = "active"
                st.session_state["_step_curr"] = key
                render_stepper(stepper_slot, steps, st.session_state["_step_status"], sticky=True)

            def _set_done_all():
                for k,_ in steps: st.session_state["_step_status"][k] = "done"
                render_stepper(stepper_slot, steps, st.session_state["_step_status"], sticky=True)

            _set_active("check"); render_progress_bar(bar_slot, 0)
            msg_slot.markdown("<div class='gp-msg'>두뇌 준비를 시작합니다…</div>", unsafe_allow_html=True)

            # (신규) 중지 플래그 초기화 및 버튼
            st.session_state["stop_requested"] = False
            with ctrl_slot.container():
                st.caption("진행 제어")
                if st.button("🛑 학습 중지", type="secondary"):
                    st.session_state["stop_requested"] = True
                    st.info("중지 요청됨 — 현재 파일까지 마무리하고 곧 멈춥니다.")

            st.session_state["_gp_pct"] = 0
            def update_pct(pct:int, msg:str|None=None):
                st.session_state["_gp_pct"] = max(0, min(100, int(pct)))
                render_progress_bar(bar_slot, st.session_state["_gp_pct"])
                if msg is not None: update_msg(msg)

            def update_msg(text:str):
                if "변경 확인" in text: _set_active("check")
                elif "리더 초기화" in text: _set_active("init")
                elif "목록 불러오는 중" in text: _set_active("list")
                elif "인덱스 생성" in text: _set_active("index")
                elif "저장 중" in text: _set_active("save")
                elif "완료" in text: _set_done_all()
                msg_slot.markdown(f"<div class='gp-msg'>{text}</div>", unsafe_allow_html=True)

            def should_stop() -> bool:
                return bool(st.session_state.get("stop_requested", False))

            # 1) LLM/Embedding 준비
            init_llama_settings(
                api_key=settings.GEMINI_API_KEY.get_secret_value(),
                llm_model=settings.LLM_MODEL,
                embed_model=settings.EMBED_MODEL,
                temperature=float(st.session_state.get("temperature", 0.0)),
            )
            # 2) 인덱스 준비/빌드(중지 신호 전달)
            index = get_or_build_index(
                update_pct=update_pct, update_msg=update_msg,
                gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
                raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
                persist_dir=PERSIST_DIR, manifest_path=MANIFEST_PATH,
                should_stop=should_stop
            )
            # 3) 엔진 연결
            st.session_state.query_engine = index.as_query_engine(
                response_mode=st.session_state.get("response_mode", settings.RESPONSE_MODE),
                similarity_top_k=int(st.session_state.get("similarity_top_k", settings.SIMILARITY_TOP_K)),
            )

            # 완료/중지 상태 메시지
            if st.session_state.get("stop_requested"):
                st.warning("학습을 중지했습니다. 다음 실행 시 **중단 지점 다음 파일부터** 이어서 학습합니다.")
            else:
                update_pct(100, "완료!")
                time.sleep(0.4)

            # 4) 자동 백업(+오래된 백업 정리) — '완료한 경우'에만 수행
            if settings.AUTO_BACKUP_TO_DRIVE and not st.session_state.get("stop_requested"):
                try:
                    creds = _validate_sa(_normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON))
                    dest = settings.BACKUP_FOLDER_ID or settings.GDRIVE_FOLDER_ID
                    with st.spinner("⬆️ 인덱스 저장본을 드라이브로 자동 백업 중..."):
                        _, file_name = export_brain_to_drive(creds, PERSIST_DIR, dest, filename=None)
                    st.success(f"자동 백업 완료! 파일명: {file_name}")
                    deleted = prune_old_backups(creds, dest, keep=int(settings.BACKUP_KEEP_N), prefix=INDEX_BACKUP_PREFIX)
                    if deleted:
                        st.info(f"오래된 백업 {len(deleted)}개 정리 완료.")
                except Exception as e:
                    st.warning("자동 백업에 실패했지만, 로컬 저장본은 정상적으로 준비되었습니다.")
                    with st.expander("백업 오류 보기"):
                        st.exception(e)

            # 진행 UI 정리 및 재실행
            stepper_slot.empty(); bar_slot.empty(); msg_slot.empty(); ctrl_slot.empty()
            st.rerun()
        return

    # 학생 화면
    with st.container():
        st.info("수업 준비 중입니다. 잠시 후 선생님이 두뇌를 연결하면 자동으로 채팅이 열립니다.")
        st.caption("이 화면은 학생 전용으로, 관리자 기능과 준비 과정은 표시하지 않습니다.")

def render_chat_ui():
    if "messages" not in st.session_state: st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    st.markdown("---")

    mode = st.radio("**어떤 도움이 필요한가요?**",
                    ["💬 이유문법 설명","🔎 구문 분석","📚 독해 및 요약"],
                    horizontal=True, key="mode_select")

    prompt = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.spinner("AI 선생님이 답변을 생각하고 있어요..."):
            selected_prompt = EXPLAINER_PROMPT if mode=="💬 이유문법 설명" else \
                              ANALYST_PROMPT if mode=="🔎 구문 분석" else READER_PROMPT
            answer = get_text_answer(st.session_state.query_engine, prompt, selected_prompt)
        st.session_state.messages.append({"role":"assistant","content":answer})
        st.rerun()

if __name__ == "__main__":
    main()
# ===== END OF FILE ============================================================
