# ===== [01] TOP OF FILE ======================================================
# Streamlit AI-Teacher App (stepper fix + pause/resume + robust message mapping)

# ===== [02] ENV VARS =========================================================
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_RUN_ON_SAVE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ===== [03] IMPORTS ==========================================================
import time
import json
import streamlit as st

from src.config import (
    settings,
    APP_DATA_DIR,
    PERSIST_DIR,
    MANIFEST_PATH,
    QUALITY_REPORT_PATH,
)
from src.ui import load_css, render_header
from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
from src.rag_engine import (
    get_or_build_index,
    init_llama_settings,
    get_text_answer,
    _load_index_from_disk,
    try_restore_index_from_drive,
    export_brain_to_drive,
    prune_old_backups,
    _normalize_sa,
    _validate_sa,
    INDEX_BACKUP_PREFIX,
    CHECKPOINT_PATH,
)
from src.auth import admin_login_flow

# ===== [04] SETTINGS & CONSTANTS =============================================
def _secret_or_str(v):
    try:
        return v.get_secret_value()  # SecretStr
    except Exception:
        return str(v)

def _default_top_k() -> int:
    return int(getattr(settings, "SIMILARITY_TOP_K", 5))

def _auto_backup_flag() -> bool:
    return bool(
        getattr(settings, "AUTO_BACKUP_TO_DRIVE", None)
        if getattr(settings, "AUTO_BACKUP_TO_DRIVE", None) is not None
        else getattr(settings, "AUTO_BACKUP_ON_SUCCESS", True)
    )

def clamp(v, lo, hi) -> int:
    try:
        v = int(v)
    except Exception:
        v = lo
    return max(lo, min(hi, v))

# ===== [05] CSS / STEPPER UI =================================================
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
    with slot: st.progress(pct)

def safe_render_header():
    try:
        render_header(
            "세상에서 가장 쉬운 이유문법",
            "AI 교사와 함께하는 똑똑한 학습",
            logo_path="assets/academy_logo.png",
        )
    except Exception:
        pass

# ===== [06] PAGE SETUP =======================================================
st.set_page_config(page_title="나의 AI 영어 교사", layout="wide", initial_sidebar_state="collapsed")
load_css(
    "assets/style.css",
    use_bg=getattr(settings, "USE_BG_IMAGE", True),
    bg_path=getattr(settings, "BG_IMAGE_PATH", "assets/background_book.png"),
)
ensure_progress_css()
safe_render_header()

_, _, c3 = st.columns([0.8, 0.1, 0.1])
with c3:
    if st.button("🛠️", key="admin_icon_top_bar"):
        st.session_state.admin_mode = True

# ===== [07] ADMIN AUTH / DIAG ===============================================
is_admin = admin_login_flow(getattr(settings, "ADMIN_PASSWORD", ""))

def _has_sa_any() -> bool:
    def _g(k: str):
        try: return st.secrets.get(k, None)
        except Exception: return None
    j = str(getattr(settings, "GDRIVE_SERVICE_ACCOUNT_JSON", "") or "").strip()
    email = os.environ.get("APP_SA_CLIENT_EMAIL") or _g("APP_SA_CLIENT_EMAIL")
    pkey  = os.environ.get("APP_SA_PRIVATE_KEY")  or _g("APP_SA_PRIVATE_KEY")
    return bool(j) or (bool(email) and bool(pkey))

if is_admin and not _has_sa_any():
    st.error("GDRIVE 서비스계정 자격증명이 비었습니다. Secrets에 JSON 또는 이메일/프라이빗키를 입력해 주세요.")
    st.caption("APP_GDRIVE_SERVICE_ACCOUNT_JSON 또는 APP_SA_CLIENT_EMAIL / APP_SA_PRIVATE_KEY 사용 가능.")

# ===== [08] AUTO ATTACH / RESTORE ===========================================
def _auto_attach_or_restore_silently() -> bool:
    try:
        if os.path.exists(PERSIST_DIR):
            init_llama_settings(
                api_key=_secret_or_str(settings.GEMINI_API_KEY),
                llm_model=settings.LLM_MODEL,
                embed_model=settings.EMBED_MODEL,
                temperature=float(st.session_state.get("temperature", 0.0)),
            )
            index = _load_index_from_disk(PERSIST_DIR)
            st.session_state.query_engine = index.as_query_engine(
                response_mode=st.session_state.get("response_mode", settings.RESPONSE_MODE),
                similarity_top_k=int(st.session_state.get("similarity_top_k", _default_top_k())),
            )
            st.session_state["_auto_attach_note"] = "local_ok"
            return True

        creds = _validate_sa(_normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON))
        dest = getattr(settings, "BACKUP_FOLDER_ID", None) or settings.GDRIVE_FOLDER_ID
        ok = try_restore_index_from_drive(creds, PERSIST_DIR, dest)
        if ok:
            index = _load_index_from_disk(PERSIST_DIR)
            st.session_state.query_engine = index.as_query_engine(
                response_mode=st.session_state.get("response_mode", settings.RESPONSE_MODE),
                similarity_top_k=int(st.session_state.get("similarity_top_k", _default_top_k())),
            )
            st.session_state["_auto_attach_note"] = "restored_from_drive"
            return True

        st.session_state["_auto_attach_note"] = "no_cache_no_backup"
    except Exception as e:
        st.session_state["_attach_error"] = repr(e)
    return False

if "query_engine" not in st.session_state:
    _auto_attach_or_restore_silently()

# ===== [09] QUALITY REPORT VIEW =============================================
def render_quality_report_view():
    st.subheader("📊 최적화 품질 리포트", divider="gray")
    if not os.path.exists(QUALITY_REPORT_PATH):
        st.info("아직 리포트가 없습니다. 인덱싱 후 자동 생성됩니다.")
        return
    try:
        with open(QUALITY_REPORT_PATH, "r", encoding="utf-8") as f:
            rep = json.load(f)
    except Exception as e:
        st.error("리포트 파일을 읽는 중 오류.")
        with st.expander("오류 보기"): st.exception(e)
        return

    s = rep.get("summary", {}) or {}
    st.write(
        f"- 전체 문서: **{s.get('total_docs', 0)}**개  "
        f"- 처리 파일: **{s.get('processed_docs', 0)}**개  "
        f"- 채택(kept): **{s.get('kept_docs', 0)}**개  "
        f"- 스킵(저품질): **{s.get('skipped_low_text', 0)}**개  "
        f"- 스킵(중복): **{s.get('skipped_dup', 0)}**개  "
        f"- 총 본문 문자수: **{s.get('total_chars', 0):,}**"
    )
    with st.expander("파일별 상세"):
        rows = []
        for fid, info in (rep.get("files", {}) or {}).items():
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
                column_config={0:"파일명",1:"채택",2:"저품질 스킵",3:"중복 스킵",4:"문자수",5:"수정시각"},
                hide_index=True, use_container_width=True
            )
        else:
            st.caption("아직 수집된 파일 통계가 없습니다.")

# ===== [10] ADMIN PANELS =====================================================
if is_admin:
    with st.expander("⚙️ 고급 RAG/LLM 설정", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.setdefault("similarity_top_k", _default_top_k())
            k = st.slider("similarity_top_k", 1, 12, int(st.session_state["similarity_top_k"]))
        with col2:
            st.session_state.setdefault("temperature", 0.0)
            temp = st.slider("LLM temperature", 0.0, 1.0, float(st.session_state["temperature"]), 0.05)
        with col3:
            st.session_state.setdefault("response_mode", settings.RESPONSE_MODE)
            mode_sel = st.selectbox(
                "response_mode", ["compact","refine","tree_summarize"],
                index=["compact","refine","tree_summarize"].index(st.session_state["response_mode"]),
            )
        if st.button("적용"):
            st.session_state["similarity_top_k"] = int(k)
            st.session_state["temperature"] = float(temp)
            st.session_state["response_mode"] = str(mode_sel)
            st.success("RAG/LLM 설정이 저장되었습니다. (다음 쿼리부터 반영)")

# ===== [10B] OPTIMIZATION PANEL (with presets) ===============================
with st.expander("🧩 최적화 설정(전처리/청킹/중복제거)", expanded=True):
    # 1) 프로필 정의
    #    - cs: chunk size, co: chunk overlap, mc: min chars
    #    - dd: dedup by hash, slt: skip low text, psu: pre-summarize
    PROFILES = {
        "⚡ 속도 우선": dict(cs=1600, co=40,  mc=80,  dd=True, slt=True, psu=False),
        "🔁 균형":     dict(cs=1024, co=80,  mc=120, dd=True, slt=True, psu=False),
        "🔎 품질 우선": dict(cs=800,  co=120, mc=200, dd=True, slt=True, psu=True),
    }

    # 상태 기본값 준비
    st.session_state.setdefault("opt_chunk_size",     settings.CHUNK_SIZE)
    st.session_state.setdefault("opt_chunk_overlap",  settings.CHUNK_OVERLAP)
    st.session_state.setdefault("opt_min_chars",      settings.MIN_CHARS_PER_DOC)
    st.session_state.setdefault("opt_dedup",          settings.DEDUP_BY_TEXT_HASH)
    st.session_state.setdefault("opt_skip_low_text",  settings.SKIP_LOW_TEXT_DOCS)
    st.session_state.setdefault("opt_pre_summarize",  settings.PRE_SUMMARIZE_DOCS)

    # 2) 프로필 적용 함수
    def _apply_profile(p: dict):
        st.session_state["opt_chunk_size"]    = int(p["cs"])
        st.session_state["opt_chunk_overlap"] = int(p["co"])
        st.session_state["opt_min_chars"]     = int(p["mc"])
        st.session_state["opt_dedup"]         = bool(p["dd"])
        st.session_state["opt_skip_low_text"] = bool(p["slt"])
        st.session_state["opt_pre_summarize"] = bool(p["psu"])

    # 3) 프로필 버튼 (원클릭 설정)
    st.write("원클릭 프로필:")
    c1, c2, c3 = st.columns(3)
    if c1.button("⚡ 속도 우선"):
        _apply_profile(PROFILES["⚡ 속도 우선"])
        st.toast("⚡ 속도 우선 프로필을 적용했어요!", icon="⚡")
        st.rerun()
    if c2.button("🔁 균형"):
        _apply_profile(PROFILES["🔁 균형"])
        st.toast("🔁 균형 프로필을 적용했어요!", icon="🔁")
        st.rerun()
    if c3.button("🔎 품질 우선"):
        _apply_profile(PROFILES["🔎 품질 우선"])
        st.toast("🔎 품질 우선 프로필을 적용했어요!", icon="🔎")
        st.rerun()

    # 현재 설정 요약 뱃지
    st.caption(
        f"현재 설정 요약 → chunk: **{st.session_state['opt_chunk_size']}** / "
        f"overlap: **{st.session_state['opt_chunk_overlap']}** / "
        f"min_chars: **{st.session_state['opt_min_chars']}** / "
        f"dedup: **{st.session_state['opt_dedup']}** / "
        f"skip_low_text: **{st.session_state['opt_skip_low_text']}** / "
        f"pre_summarize: **{st.session_state['opt_pre_summarize']}**"
    )

    st.divider()

    # 4) 세부 수동 조정 (원하면 덮어쓰기 가능)
    c1, c2, c3 = st.columns(3)
    with c1:
        # 범위 정의
        cs_min, cs_max = 200, 2000
        co_min, co_max = 0, 400
        mc_min, mc_max = 50, 3000  # 80도 허용되도록 50부터

        # 현재값을 범위로 보정
        cs_def = clamp(st.session_state["opt_chunk_size"], cs_min, cs_max)
        co_def = clamp(st.session_state["opt_chunk_overlap"], co_min, co_max)
        mc_def = clamp(st.session_state["opt_min_chars"], mc_min, mc_max)

        cs = st.number_input("청크 크기(문자)", min_value=cs_min, max_value=cs_max, value=int(cs_def), step=50)
        co = st.number_input("청크 오버랩(문자)", min_value=co_min, max_value=co_max, value=int(co_def), step=10)
        mc = st.number_input("문서 최소 길이(문자)", min_value=mc_min, max_value=mc_max, value=int(mc_def), step=50)

    with c2:
        dd = st.toggle("텍스트 해시로 중복 제거", value=bool(st.session_state["opt_dedup"]))
    with c3:
        slt = st.toggle("저품질(짧은/빈약) 문서 스킵", value=bool(st.session_state["opt_skip_low_text"]))
        psu = st.toggle("문서 요약 메타데이터 생성(느려짐)", value=bool(st.session_state["opt_pre_summarize"]))

    if st.button("최적화 설정 적용"):
        st.session_state["opt_chunk_size"]    = int(cs)
        st.session_state["opt_chunk_overlap"] = int(co)
        st.session_state["opt_min_chars"]     = int(mc)
        st.session_state["opt_dedup"]         = bool(dd)
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
                st.session_state.pop("build_paused", None)
                st.success("두뇌 파일이 초기화되었습니다. 아래에서 다시 준비하세요.")
        with c2:
            if st.button("⬆️ 두뇌 저장본 드라이브로 내보내기(날짜 포함)"):
                try:
                    creds = _validate_sa(_normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON))
                    dest = getattr(settings, "BACKUP_FOLDER_ID", None) or settings.GDRIVE_FOLDER_ID
                    with st.spinner("두뇌를 ZIP(날짜 포함)으로 묶고 드라이브에 업로드 중..."):
                        _, file_name = export_brain_to_drive(creds, PERSIST_DIR, dest, filename=None)
                    st.success(f"업로드 완료! 파일명: {file_name}")
                    deleted = prune_old_backups(creds, dest, keep=int(getattr(settings, "BACKUP_KEEP_N", 5)), prefix=INDEX_BACKUP_PREFIX)
                    if deleted: st.info(f"오래된 백업 {len(deleted)}개 정리 완료.")
                except Exception as e:
                    st.error("내보내기 실패. 두뇌가 준비되었는지와 폴더 권한(편집자)을 확인하세요.")
                    with st.expander("자세한 오류 보기"): st.exception(e)
        with c3:
            if st.button("⬇️ 드라이브에서 최신 백업 가져오기"):
                try:
                    creds = _validate_sa(_normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON))
                    dest = getattr(settings, "BACKUP_FOLDER_ID", None) or settings.GDRIVE_FOLDER_ID
                    with st.spinner("드라이브에서 최신 백업 ZIP을 내려받아 복원 중..."):
                        ok = try_restore_index_from_drive(creds, PERSIST_DIR, dest)
                    if ok: st.success("복원 완료! 아래에서 두뇌를 연결하거나 대화를 시작하세요.")
                    else:  st.warning("백업 ZIP을 찾지 못했습니다. 먼저 내보내기를 해주세요.")
                except Exception as e:
                    st.error("가져오기 실패. 폴더 권한(편집자)과 파일 존재 여부를 확인하세요.")
                    with st.expander("자세한 오류 보기"): st.exception(e)

    with st.expander("🔎 인덱스 상태 진단", expanded=False):
        st.write(f"• 로컬 저장 경로: `{PERSIST_DIR}` → {'존재' if os.path.isdir(PERSIST_DIR) else '없음'}")
        st.write(f"• 체크포인트: `{CHECKPOINT_PATH}` → {'존재' if os.path.exists(CHECKPOINT_PATH) else '없음'}")
        render_quality_report_view()

# ===== [11] BUILD WORKFLOW (RESUME/STOP) ====================================
def _build_or_resume_workflow():
    stepper_slot = st.empty(); bar_slot = st.empty(); msg_slot = st.empty(); ctrl_slot = st.empty()
    steps = [("check","드라이브 변경 확인"),("init","Drive 리더 초기화"),
             ("list","문서 목록 불러오는 중"),("index","인덱스 생성"),("save","두뇌 저장")]
    st.session_state["_step_status"] = {k:"pending" for k,_ in steps}
    st.session_state["_step_curr"] = None

    def _advance_to(key:str):
        """뒤로 물러나지 않고 앞으로만 전진."""
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

    _advance_to("check")
    render_progress(bar_slot, 0)
    msg_slot.markdown("<div class='gp-msg'>두뇌 준비를 시작합니다…</div>", unsafe_allow_html=True)

    # 중지 버튼
    st.session_state["stop_requested"] = False
    with ctrl_slot.container():
        c1, c2 = st.columns([1,1])
        with c1: st.caption("진행 제어")
        with c2:
            if st.button("🛑 학습 중지", type="secondary"):
                st.session_state["stop_requested"] = True
                st.info("중지 요청됨 — 현재 파일까지 마무리하고 곧 멈춥니다.")

    # 진행 업데이트
    st.session_state["_gp_pct"] = 0

    def update_pct(pct:int, msg:str|None=None):
        # 퍼센트만 들어와도 단계 추정(메시지 없을 때 보조)
        pct_i = max(0, min(100, int(pct)))
        st.session_state["_gp_pct"] = pct_i
        # 대략적인 구간별 단계 추정
        if pct_i < 10:      _advance_to("check")
        elif pct_i < 25:    _advance_to("init")
        elif pct_i < 50:    _advance_to("list")
        elif pct_i < 95:    _advance_to("index")
        else:               _advance_to("save")
        render_progress(bar_slot, pct_i)
        if msg is not None: update_msg(msg)

    def update_msg(text:str):
        """한/영 다양한 로그 문구를 폭넓게 매칭해서 단계 갱신."""
        t = (text or "").lower()

        # check
        if any(k in t for k in ["변경 확인","change check","drive change","check"]):
            _advance_to("check")

        # init(초기화/세팅 계열)
        elif any(k in t for k in ["리더 초기화","reader init","initialize","init","인증","credential","service"]):
            _advance_to("init")

        # list(목록/로딩/매니페스트 계열)
        elif any(k in t for k in ["목록","list","files","file list","manifest","매니페스트","로드","불러오"]):
            _advance_to("list")

        # index(인덱싱/청킹/임베딩 계열)
        elif any(k in t for k in ["인덱스","index","chunk","청크","embed","임베","build","vector","persisting"]):
            _advance_to("index")

        # save(저장/백업/쓰기 계열)
        elif any(k in t for k in ["저장","save","persist","write","백업","backup","upload"]):
            _advance_to("save")

        # 완료
        if any(k in t for k in ["완료","done","finish","finished","success"]):
            _set_done_all()

        msg_slot.markdown(f"<div class='gp-msg'>{text}</div>", unsafe_allow_html=True)

    def should_stop() -> bool:
        return bool(st.session_state.get("stop_requested", False))

    # 1) 모델 준비
    init_llama_settings(
        api_key=_secret_or_str(settings.GEMINI_API_KEY),
        llm_model=settings.LLM_MODEL,
        embed_model=settings.EMBED_MODEL,
        temperature=float(st.session_state.get("temperature", 0.0)),
    )

    # 2) 인덱스 준비/빌드(체크포인트 + 중지 신호)
    index = get_or_build_index(
        update_pct=update_pct,
        update_msg=update_msg,
        gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
        raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
        persist_dir=PERSIST_DIR,
        manifest_path=MANIFEST_PATH,
        should_stop=should_stop,
    )

    # 중지한 경우는 쿼리엔진 연결/백업을 생략하고 '재개' 대기 상태로 돌려놓음
    if st.session_state.get("stop_requested"):
        st.session_state["build_paused"] = True
        st.warning("학습을 중지했습니다. **아래의 ‘▶ 재개’ 버튼으로** 이어서 학습할 수 있어요.")
        return False  # not finished

    # 3) 연결
    st.session_state.query_engine = index.as_query_engine(
        response_mode=st.session_state.get("response_mode", settings.RESPONSE_MODE),
        similarity_top_k=int(st.session_state.get("similarity_top_k", _default_top_k())),
    )

    # 4) 완료 표시
    update_pct(100, "완료!")
    time.sleep(0.4)

    # 5) 자동 백업
    if _auto_backup_flag():
        try:
            creds = _validate_sa(_normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON))
            dest = getattr(settings, "BACKUP_FOLDER_ID", None) or settings.GDRIVE_FOLDER_ID
            with st.spinner("⬆️ 인덱스 저장본을 드라이브로 자동 백업 중..."):
                _, file_name = export_brain_to_drive(creds, PERSIST_DIR, dest, filename=None)
            st.success(f"자동 백업 완료! 파일명: {file_name}")
            deleted = prune_old_backups(
                creds, dest,
                keep=int(getattr(settings, "BACKUP_KEEP_N", 5)),
                prefix=INDEX_BACKUP_PREFIX,
            )
            if deleted: st.info(f"오래된 백업 {len(deleted)}개 정리 완료.")
        except Exception as e:
            st.warning("자동 백업에 실패했지만, 로컬 저장본은 정상적으로 준비되었습니다.")
            with st.expander("백업 오류 보기"): st.exception(e)

    return True  # finished

# ===== [12] MAIN =============================================================
def main():
    # 이미 두뇌가 붙어 있으면 바로 채팅
    if "query_engine" in st.session_state and not st.session_state.get("build_paused"):
        render_chat_ui()
        return

    if is_admin:
        st.info("AI 교사를 시작/재개하려면 아래 버튼을 사용하세요. (체크포인트/중지 지원)")

        # 중지 후 재개 모드
        if st.session_state.get("build_paused"):
            c1, c2 = st.columns([1,1])
            with c1:
                if st.button("▶ 재개"):
                    st.session_state.pop("build_paused", None)
                    finished = _build_or_resume_workflow()
                    if finished:
                        st.rerun()
                        return
            with c2:
                if st.button("↺ 처음부터 다시"):
                    st.session_state.pop("build_paused", None)
                    # 초기화 버튼과 같지만 빠르게 처리
                    import shutil
                    if os.path.exists(PERSIST_DIR): shutil.rmtree(PERSIST_DIR)
                    for p in (CHECKPOINT_PATH, MANIFEST_PATH, QUALITY_REPORT_PATH):
                        try:
                            if os.path.exists(p): os.remove(p)
                        except Exception: pass
                    st.success("초기화 완료. 다시 ‘AI 두뇌 준비 시작하기’를 눌러주세요.")
            st.stop()

        # 일반 시작
        if st.button("🧠 AI 두뇌 준비 시작하기"):
            finished = _build_or_resume_workflow()
            if finished:
                st.rerun()
            return

        # (아직 준비 안 된 학생 화면 숨김)
        with st.container():
            st.caption("아직 두뇌가 준비되지 않았습니다. 위의 버튼으로 준비를 시작하세요.")
        return

    # 학생 화면(관리자 외)
    with st.container():
        st.info("수업 준비 중입니다. 잠시 후 선생님이 두뇌를 연결하면 자동으로 채팅이 열립니다.")
        st.caption("이 화면은 학생 전용으로, 관리자 기능과 준비 과정은 표시하지 않습니다.")

# ===== [13] CHAT UI ==========================================================
def render_chat_ui():
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    st.markdown("---")

    mode = st.radio("**어떤 도움이 필요한가요?**",
                    ["💬 이유문법 설명","🔎 구문 분석","📚 독해 및 요약"],
                    horizontal=True, key="mode_select")

    prompt = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.spinner("AI 선생님이 답변을 생각하고 있어요..."):
            selected_prompt = (
                EXPLAINER_PROMPT if mode=="💬 이유문법 설명" else
                ANALYST_PROMPT  if mode=="🔎 구문 분석"  else
                READER_PROMPT
            )
            answer = get_text_answer(st.session_state.query_engine, prompt, selected_prompt)
        st.session_state.messages.append({"role":"assistant","content":answer})
        st.rerun()

# ===== [99] END OF FILE ======================================================
