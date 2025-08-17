# app.py
# ===== TOP OF FILE ============================================================
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_RUN_ON_SAVE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

import time
import streamlit as st

from src.config import settings, APP_DATA_DIR, PERSIST_DIR, MANIFEST_PATH
from src.ui import (
    load_css, safe_render_header, ensure_progress_css,
    render_progress_bar, render_stepper
)
from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
from src.rag_engine import (
    get_or_build_index, init_llama_settings, get_text_answer,
    _load_index_from_disk, try_restore_index_from_drive,
    export_brain_to_drive, prune_old_backups, _normalize_sa, _validate_sa,
    INDEX_BACKUP_PREFIX,
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
            return True
        creds = _validate_sa(_normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON))
        ok = try_restore_index_from_drive(creds, PERSIST_DIR, settings.BACKUP_FOLDER_ID or settings.GDRIVE_FOLDER_ID)
        if ok:
            index = _load_index_from_disk(PERSIST_DIR)
            st.session_state.query_engine = index.as_query_engine(
                response_mode=st.session_state.get("response_mode", settings.RESPONSE_MODE),
                similarity_top_k=int(st.session_state.get("similarity_top_k", settings.SIMILARITY_TOP_K)),
            )
            return True
    except Exception:
        pass
    return False

if "query_engine" not in st.session_state:
    _auto_attach_or_restore_silently()

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

    with st.expander("🛠️ 관리자 도구", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📥 강의 자료 다시 불러오기 (두뇌 초기화)"):
                import shutil
                if os.path.exists(PERSIST_DIR):
                    shutil.rmtree(PERSIST_DIR)
                if "query_engine" in st.session_state:
                    del st.session_state["query_engine"]
                st.success("두뇌 파일이 초기화되었습니다. 아래에서 다시 준비하세요.")
        with c2:
            if st.button("⬆️ 두뇌 저장본 드라이브로 내보내기(날짜 포함)"):
                try:
                    creds = _validate_sa(_normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON))
                    dest = settings.BACKUP_FOLDER_ID or settings.GDRIVE_FOLDER_ID
                    with st.spinner("두뇌를 ZIP(날짜 포함)으로 묶고 드라이브에 업로드 중..."):
                        file_id, file_name = export_brain_to_drive(creds, PERSIST_DIR, dest, filename=None)  # ← 자동 날짜
                    st.success(f"업로드 완료! 파일명: {file_name}")

                    # 보관 개수 정책 적용(오래된 백업 정리)
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

# ── 메인 워크플로우 ───────────────────────────────────────────────────────────
def main():
    # 준비되어 있으면 채팅으로 바로 진입
    if "query_engine" in st.session_state:
        render_chat_ui()
        return

    # 두뇌가 없고, 관리자만 준비 UI를 봄
    if is_admin:
        st.info("AI 교사를 준비하려면 아래 버튼을 누르세요. 자료량에 따라 시간이 소요될 수 있습니다.")

        if st.button("🧠 AI 두뇌 준비 시작하기"):
            # 진행 UI 슬롯
            stepper_slot = st.empty(); bar_slot = st.empty(); msg_slot = st.empty()

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

            # 1) LLM/Embedding 준비
            init_llama_settings(
                api_key=settings.GEMINI_API_KEY.get_secret_value(),
                llm_model=settings.LLM_MODEL,
                embed_model=settings.EMBED_MODEL,
                temperature=float(st.session_state.get("temperature", 0.0)),
            )
            # 2) 인덱스 준비/빌드
            index = get_or_build_index(
                update_pct=update_pct, update_msg=update_msg,
                gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
                raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
                persist_dir=PERSIST_DIR, manifest_path=MANIFEST_PATH,
            )
            # 3) 엔진 연결
            st.session_state.query_engine = index.as_query_engine(
                response_mode=st.session_state.get("response_mode", settings.RESPONSE_MODE),
                similarity_top_k=int(st.session_state.get("similarity_top_k", settings.SIMILARITY_TOP_K)),
            )
            update_pct(100, "완료!")
            time.sleep(0.4)

            # 4) 자동 백업(+오래된 백업 정리)
            if settings.AUTO_BACKUP_TO_DRIVE:
                try:
                    creds = _validate_sa(_normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON))
                    dest = settings.BACKUP_FOLDER_ID or settings.GDRIVE_FOLDER_ID
                    with st.spinner("⬆️ 인덱스 저장본을 드라이브로 자동 백업 중..."):
                        _, file_name = export_brain_to_drive(creds, PERSIST_DIR, dest, filename=None)  # ← 날짜 포함
                    st.success(f"자동 백업 완료! 파일명: {file_name}")

                    deleted = prune_old_backups(creds, dest, keep=int(settings.BACKUP_KEEP_N), prefix=INDEX_BACKUP_PREFIX)
                    if deleted:
                        st.info(f"오래된 백업 {len(deleted)}개 정리 완료.")
                except Exception as e:
                    st.warning("자동 백업에 실패했지만, 로컬 저장본은 정상적으로 준비되었습니다.")
                    with st.expander("백업 오류 보기"):
                        st.exception(e)

            stepper_slot.empty(); bar_slot.empty(); msg_slot.empty()
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
