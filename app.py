# ===== [01] TOP OF FILE ======================================================
# Streamlit AI-Teacher — 관리자 가드 강화 + 단일 '관리자 모드 끄기' + 친절한 에러
import os, time, re, datetime as dt, traceback, base64, sys
from pathlib import Path
import streamlit as st

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_RUN_ON_SAVE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ===== [02] IMPORTS (fallback) ==============================================
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from src.config import settings, PERSIST_DIR
    from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
    from src.rag_engine import (
        get_or_build_index, init_llama_settings, get_text_answer,
        _normalize_sa, _validate_sa
    )
    from src.auth import admin_login_flow
    _IMPORT_MODE = "src"
except Exception:
    import config as _config
    settings = _config.settings
    PERSIST_DIR = _config.PERSIST_DIR

    import prompts as _prompts
    EXPLAINER_PROMPT = _prompts.EXPLAINER_PROMPT
    ANALYST_PROMPT = _prompts.ANALYST_PROMPT
    READER_PROMPT  = _prompts.READER_PROMPT

    import rag_engine as _rag
    get_or_build_index = _rag.get_or_build_index
    init_llama_settings = _rag.init_llama_settings
    get_text_answer     = _rag.get_text_answer
    _normalize_sa       = _rag._normalize_sa
    _validate_sa        = _rag._validate_sa

    import auth as _auth
    admin_login_flow = _auth.admin_login_flow
    _IMPORT_MODE = "root"

# ===== [03] SECRET/STRING HELPER ============================================
def _sec(value) -> str:
    try:
        from pydantic.types import SecretStr
        if isinstance(value, SecretStr):
            return value.get_secret_value()
    except Exception:
        pass
    if value is None: return ""
    if isinstance(value, dict):
        import json; return json.dumps(value, ensure_ascii=False)
    return str(value)

# ===== [04] UI HELPERS =======================================================
@st.cache_data(show_spinner=False)
def _read_text(path_str: str) -> str:
    try: return Path(path_str).read_text(encoding="utf-8")
    except Exception: return ""

@st.cache_data(show_spinner=False)
def _file_b64(path_str: str) -> str:
    try: return base64.b64encode(Path(path_str).read_bytes()).decode()
    except Exception: return ""

def load_css(file_path: str, use_bg: bool = False, bg_path: str | None = None) -> None:
    css = _read_text(file_path) or ""
    bg_css = ""
    if use_bg and bg_path:
        b64 = _file_b64(bg_path)
        if b64:
            bg_css = f"""
            .stApp{{background-image:url("data:image/png;base64,{b64}");
                    background-size:cover;background-position:center;
                    background-repeat:no-repeat;background-attachment:fixed;}}
            """
    st.markdown(f"<style>{bg_css}\n{css}</style>", unsafe_allow_html=True)

def safe_render_header(title: str | None = None, subtitle: str | None = None,
                       logo_path: str | None = "assets/academy_logo.png",
                       logo_height_px: int | None = None) -> None:
    _title = title or getattr(settings,"TITLE_TEXT","나의 AI 영어 교사")
    _subtitle = subtitle or getattr(settings,"SUBTITLE_TEXT","")
    _logo_h = int(logo_height_px or getattr(settings,"LOGO_HEIGHT_PX",56))
    logo_b64 = _file_b64(logo_path) if logo_path else ""
    st.markdown(f"""
    <style>
      .aihdr-wrap{{display:flex;align-items:center;gap:14px;margin:6px 0 10px;}}
      .aihdr-logo{{height:{_logo_h}px;width:auto;object-fit:contain;display:block}}
      .aihdr-title{{font-size:{getattr(settings,'TITLE_SIZE_REM',2.2)}rem;color:{getattr(settings,'BRAND_COLOR','#F8FAFC')};margin:0}}
      .aihdr-sub{{color:#C7D2FE;margin:2px 0 0 0;}}
    </style>
    """, unsafe_allow_html=True)
    left, _ = st.columns([0.85,0.15])
    with left:
        st.markdown(f"""
        <div class="aihdr-wrap">
          {'<img src="data:image/png;base64,'+logo_b64+'" class="aihdr-logo"/>' if logo_b64 else ''}
          <div>
            <h1 class="aihdr-title">{_title}</h1>
            {f'<div class="aihdr-sub">{_subtitle}</div>' if _subtitle else ''}
          </div>
        </div>
        """, unsafe_allow_html=True)

def ensure_progress_css() -> None:
    st.markdown("""
    <style>
      .gp-wrap{ width:100%; height:28px; border-radius:12px;
        background:#1f2937; border:1px solid #334155;
        position:relative; overflow:hidden; box-shadow:0 4px 14px rgba(0,0,0,.25);}
      .gp-fill{ height:100%; background:linear-gradient(90deg,#7c5ad9,#9067C6); transition:width .25s ease;}
      .gp-label{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
        font-weight:800; color:#E8EDFF; text-shadow:0 1px 2px rgba(0,0,0,.5); font-size:18px; pointer-events:none;}
      .gp-msg{ margin-top:.5rem; color:#E8EDFF; opacity:.9; font-size:0.95rem;}
    </style>
    """, unsafe_allow_html=True)

def render_progress_bar(slot, pct: int) -> None:
    pct = max(0, min(100, int(pct)))
    slot.markdown(
        f"""<div class="gp-wrap"><div class="gp-fill" style="width:{pct}%"></div>
        <div class="gp-label">{pct}%</div></div>""",
        unsafe_allow_html=True,
    )

# ===== [05] PAGE SETUP =======================================================
st.set_page_config(page_title="나의 AI 영어 교사", layout="wide", initial_sidebar_state="expanded")
st.session_state.setdefault("admin_mode", False)
load_css("assets/style.css", use_bg=True, bg_path="assets/background_book.png")
ensure_progress_css()
safe_render_header(subtitle=f"임포트 경로: {_IMPORT_MODE}")

# ===== [06] LOG PANEL ========================================================
def _log(msg: str):
    st.session_state.setdefault("_ui_logs", [])
    ts = dt.datetime.now().strftime("%H:%M:%S")
    st.session_state["_ui_logs"].append(f"[{ts}] {msg}")

def _log_exception(prefix: str, exc: Exception):
    _log(f"{prefix}: {exc}")
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    st.session_state["_ui_traceback"] = tb

def _log_kv(k, v): _log(f"{k}: {v}")

# ===== [07] ADMIN ENTRY / GUARD =============================================
# 오른쪽 상단 공구 아이콘 → admin_mode=True (항상 표시)
_, _, _c3 = st.columns([0.82, 0.09, 0.09])
with _c3:
    if st.button("🛠️", key="admin_icon_top_bar"):
        st.session_state.admin_mode = True
        _log("관리자 버튼 클릭")

RAW_ADMIN_PW = _sec(getattr(settings, "ADMIN_PASSWORD", ""))
HAS_ADMIN_PW = bool(RAW_ADMIN_PW.strip())

# 비밀번호가 설정되지 않았다면, 관리자 기능 자체를 잠금
is_admin = admin_login_flow(RAW_ADMIN_PW) if HAS_ADMIN_PW else False

# ===== [08] Drive 복구(실제 다운로드 구현) ===================================
def try_restore_index_from_drive(
    creds: dict,
    persist_dir: str | Path,
    folder_id: str,
    update_msg: Callable[[str], None] | None = None,
) -> tuple[bool, str | None]:
    """
    Google Drive v3 API를 사용해 folder_id 하위의 파일/폴더를 재귀적으로 내려받아
    persist_dir에 동일한 구조로 복구합니다.
    - creds: 서비스 계정 JSON(dict)
    - persist_dir: 로컬 저장 경로
    - folder_id: 백업 폴더 ID
    반환: (성공여부, 참고메모)
    """
    from pathlib import Path
    import io
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    def _emit_msg(m: str):
        try:
            if update_msg: update_msg(m)
        except Exception:
            pass

    try:
        if not folder_id or not str(folder_id).strip():
            raise FolderIdMissing("폴더 ID가 비어 있습니다.")
        if "client_email" not in creds:
            raise ServiceAccountInvalid("서비스 계정 키에 client_email이 없습니다.")

        # [08-1] 자격증명/드라이브 클라이언트
        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        credentials = Credentials.from_service_account_info(creds, scopes=scopes)
        svc = build("drive", "v3", credentials=credentials, cache_discovery=False)

        # [08-2] 헬퍼: 폴더/파일 구분 및 리스트
        def _list_children(fid: str) -> list[dict]:
            files = []
            page_token = None
            query = f"'{fid}' in parents and trashed=false"
            while True:
                res = svc.files().list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, size)",
                    pageToken=page_token,
                ).execute()
                files.extend(res.get("files", []))
                page_token = res.get("nextPageToken")
                if not page_token:
                    break
            return files

        def _is_folder(item: dict) -> bool:
            return item.get("mimeType") == "application/vnd.google-apps.folder"

        # [08-3] 헬퍼: Google Docs류는 export, 일반 파일은 download
        # 인덱스는 보통 일반 파일(json, pkl, bin, txt 등)이므로 우선 일반 다운로드에 초점
        GOOGLE_DOC_EXPORT = {
            "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
            "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
            "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
        }

        def _download_file(file_id: str, name: str, mime_type: str, out_path: Path):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if mime_type in GOOGLE_DOC_EXPORT:
                export_mime, ext = GOOGLE_DOC_EXPORT[mime_type]
                request = svc.files().export_media(fileId=file_id, mimeType=export_mime)
                fh = io.FileIO(str(out_path.with_suffix(ext)), "wb")
            else:
                request = svc.files().get_media(fileId=file_id)
                fh = io.FileIO(str(out_path), "wb")
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _status, done = downloader.next_chunk()
            fh.close()

        # [08-4] 재귀 내려받기
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        downloaded_count = 0

        def _walk_and_download(cur_folder_id: str, dst_dir: Path):
            nonlocal downloaded_count
            items = _list_children(cur_folder_id)
            for it in items:
                fname = it.get("name", "unnamed")
                fid   = it.get("id")
                mime  = it.get("mimeType", "")
                if _is_folder(it):
                    _emit_msg(f"폴더: {fname} 내려받는 중…")
                    _walk_and_download(fid, dst_dir / fname)
                else:
                    _emit_msg(f"파일: {fname} 내려받는 중…")
                    _download_file(fid, fname, mime, dst_dir / fname)
                    downloaded_count += 1

        _emit_msg("Drive에서 백업 파일을 내려받는 중…")
        _walk_and_download(folder_id, persist_dir)

        if downloaded_count == 0:
            # 폴더는 있었지만 내부가 비었거나 권한 부족
            return (False, "폴더에 다운로드할 파일이 없거나 접근 권한이 부족합니다.")
        _emit_msg(f"다운로드 완료: {downloaded_count}개 파일")
        return (True, f"{downloaded_count} files downloaded")

    except RAGEngineError:
        raise
    except Exception as e:
        raise DriveRestoreFailed("Drive 복구 중 예기치 못한 오류가 발생했습니다.", debug=repr(e))

# ===== [09] SIDEBAR (관리자일 때만) ==========================================
with st.sidebar:
    if HAS_ADMIN_PW and is_admin and st.session_state.get("admin_mode"):
        # 🔒 관리자 모드 끄기 — 버튼은 사이드바 딱 한 곳만
        if st.button("🔒 관리자 모드 끄기"):
            st.session_state.admin_mode = False
            _log("관리자 모드 끔")
            st.rerun()

        st.markdown("## ⚙️ 관리자 패널")

        st.markdown("### 🧭 응답 모드")
        st.session_state.setdefault("use_manual_override", False)
        st.session_state["use_manual_override"] = st.checkbox(
            "수동 모드(관리자 오버라이드) 사용", value=st.session_state["use_manual_override"])
        st.session_state.setdefault("manual_prompt_mode", "explainer")
        st.session_state["manual_prompt_mode"] = st.selectbox(
            "수동 모드 선택", ["explainer","analyst","reader"],
            index=["explainer","analyst","reader"].index(st.session_state["manual_prompt_mode"])
        )

        with st.expander("🤖 RAG/LLM 설정", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state.setdefault("similarity_top_k", getattr(settings,"SIMILARITY_TOP_K",5))
                st.session_state["similarity_top_k"] = st.slider("similarity_top_k", 1, 12, int(st.session_state["similarity_top_k"]))
            with c2:
                st.session_state.setdefault("temperature", 0.0)
                st.session_state["temperature"] = st.slider("LLM temperature", 0.0, 1.0, float(st.session_state["temperature"]), 0.05)
            with c3:
                st.session_state.setdefault("response_mode", getattr(settings,"RESPONSE_MODE","compact"))
                st.session_state["response_mode"] = st.selectbox(
                    "response_mode", ["compact","refine","tree_summarize"],
                    index=["compact","refine","tree_summarize"].index(st.session_state["response_mode"])
                )

        with st.expander("🛠️ 관리자 도구", expanded=False):
            if st.button("↺ 두뇌 초기화(인덱스 삭제)"):
                import shutil
                try:
                    if os.path.exists(PERSIST_DIR): shutil.rmtree(PERSIST_DIR)
                    st.session_state.pop("query_engine", None)
                    _log("두뇌 초기화 완료"); st.success("두뇌 파일 삭제됨. 메인에서 다시 준비하세요.")
                except Exception as e:
                    _log_exception("두뇌 초기화 실패", e); st.error("초기화 중 오류. 우측 로그/Traceback 확인.")

# ===== [10] MAIN: 강의 준비 & 진단 & 채팅 ===================================
with left:
    # --- [10-1] 두뇌 준비 ----------------------------------------------------
    if "query_engine" not in st.session_state:
        st.markdown("## 📚 강의 준비")
        st.info("‘AI 두뇌 준비’는 로컬 저장본이 있으면 연결하고, 없으면 Drive에서 복구합니다.\n서비스 계정 권한과 폴더 ID가 올바른지 확인하세요.")

        btn_col, diag_col = st.columns([0.55, 0.45])
        with btn_col:
            if st.button("🧠 AI 두뇌 준비(복구/연결)"):
                bar_slot = st.empty()
                msg_slot = st.empty()
                key = "_gp_pct"; st.session_state[key] = 0

                def update_pct(p, m=None):
                    st.session_state[key] = max(0, min(100, int(p)))
                    render_progress_bar(bar_slot, st.session_state[key])
                    if m:
                        msg_slot.markdown(f"<div class='gp-msg'>{m}</div>", unsafe_allow_html=True)
                        _log(m)

                try:
                    update_pct(0, "두뇌 준비를 시작합니다…")

                    # 1) LLM 초기화
                    try:
                        init_llama_settings(
                            api_key=_sec(getattr(settings, "GEMINI_API_KEY", "")),
                            llm_model=settings.LLM_MODEL,
                            embed_model=settings.EMBED_MODEL,
                            temperature=float(st.session_state.get("temperature", 0.0))
                        )
                        _log("LLM 설정 완료"); update_pct(2, "설정 확인 중…")
                    except Exception as ee:
                        public = getattr(ee, "public_msg", str(ee))
                        _log_exception("LLM 초기화 실패", ee)
                        st.error(f"LLM 초기화 실패: {public}"); st.stop()

                    # 2) 인덱스 로드/복구
                    try:
                        folder_id = getattr(settings, "GDRIVE_FOLDER_ID", None) or getattr(settings, "BACKUP_FOLDER_ID", None)
                        raw_sa = getattr(settings, "GDRIVE_SERVICE_ACCOUNT_JSON", None)
                        persist_dir = PERSIST_DIR
                        _log_kv("PERSIST_DIR", persist_dir)
                        _log_kv("local_cache", "exists ✅" if os.path.exists(persist_dir) else "missing ❌")
                        _log_kv("folder_id", str(folder_id or "(empty)"))
                        _log_kv("has_service_account", "yes" if raw_sa else "no")

                        index = get_or_build_index(
                            update_pct=update_pct,
                            update_msg=lambda m: update_pct(st.session_state[key], m),
                            gdrive_folder_id=folder_id,
                            raw_sa=raw_sa,
                            persist_dir=persist_dir,
                            manifest_path=getattr(settings, "MANIFEST_PATH", None)
                        )
                    except Exception as ee:
                        public = getattr(ee, "public_msg", str(ee))
                        _log_exception("인덱스 준비 실패", ee)
                        st.error(f"두뇌 준비 실패: {public}")
                        st.stop()

                    # 3) QueryEngine 생성
                    try:
                        st.session_state.query_engine = index.as_query_engine(
                            response_mode=st.session_state.get("response_mode", getattr(settings,"RESPONSE_MODE","compact")),
                            similarity_top_k=int(st.session_state.get("similarity_top_k", getattr(settings,"SIMILARITY_TOP_K",5)))
                        )
                        update_pct(100, "두뇌 준비 완료!"); _log("query_engine 생성 완료 ✅")
                        time.sleep(0.3); st.rerun()
                    except Exception as ee:
                        public = getattr(ee, "public_msg", str(ee))
                        _log_exception("QueryEngine 생성 실패", ee)
                        st.error(f"두뇌 준비는 되었으나 QueryEngine 생성에서 실패: {public}")
                        st.stop()

                except Exception as e:
                    _log_exception("예상치 못한 오류", e)
                    st.error("두뇌 준비 중 알 수 없는 오류. 우측 로그/Traceback을 확인하세요.")
                    st.stop()

            if st.button("📥 강의 자료 다시 불러오기(두뇌 초기화)"):
                import shutil
                try:
                    if os.path.exists(PERSIST_DIR): shutil.rmtree(PERSIST_DIR)
                    st.session_state.pop("query_engine", None)
                    _log("본문에서 두뇌 초기화 실행"); st.success("두뇌 파일을 삭제했습니다. 다시 ‘AI 두뇌 준비’를 눌러주세요.")
                except Exception as e:
                    _log_exception("본문 초기화 실패", e); st.error("초기화 중 오류. 우측 로그/Traceback 확인.")

        with diag_col:
            st.markdown("#### 🧪 연결 진단(빠름)")
            st.caption("로컬 캐시/SA/폴더 ID/Drive 복구를 검사하고 로그에 기록합니다.")
            if st.button("연결 진단 실행"):
                try:
                    _log_kv("PERSIST_DIR", PERSIST_DIR)
                    if os.path.exists(PERSIST_DIR):
                        _log_kv("local_cache", f"exists ✅, files={len(os.listdir(PERSIST_DIR))}")
                    else:
                        _log_kv("local_cache", "missing ❌")
                    try:
                        sa_norm = _normalize_sa(getattr(settings,"GDRIVE_SERVICE_ACCOUNT_JSON", None))
                        creds = _validate_sa(sa_norm)
                        _log("service_account: valid ✅")
                        _log_kv("sa_client_email", creds.get("client_email","(unknown)"))
                    except Exception as se:
                        _log_exception("service_account invalid ❌", se)
                    folder_id = getattr(settings, "BACKUP_FOLDER_ID", None) or getattr(settings, "GDRIVE_FOLDER_ID", None)
                    _log_kv("folder_id", str(folder_id or "(empty)"))
                    st.success("진단 완료. 우측 로그/Traceback 확인하세요.")
                except Exception as e:
                    _log_exception("연결 진단 자체 실패", e)
                    st.error("연결 진단 중 오류. 우측 로그/Traceback 확인.")
        st.stop()

    # --- [10-2] 채팅 UI ------------------------------------------------------
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    st.markdown("---")

    mode_label = st.radio("**어떤 도움이 필요한가요?**",
                          ["💬 이유문법 설명","🔎 구문 분석","📚 독해 및 요약"],
                          horizontal=True, key="mode_select")
    prompt = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
    if not prompt: st.stop()

    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if HAS_ADMIN_PW and is_admin and st.session_state.get("admin_mode") and st.session_state.get("use_manual_override"):
        final_mode = st.session_state.get("manual_prompt_mode","explainer"); origin="관리자 수동"
    else:
        final_mode = "explainer" if mode_label.startswith("💬") else "analyst" if mode_label.startswith("🔎") else "reader"
        origin="학생 선택"

    selected_prompt = EXPLAINER_PROMPT if final_mode=="explainer" else ANALYST_PROMPT if final_mode=="analyst" else READER_PROMPT
    _log(f"모드 결정: {origin} → {final_mode}")

    try:
        with st.spinner("AI 선생님이 답변을 생각하고 있어요..."):
            answer = get_text_answer(st.session_state.query_engine, prompt, selected_prompt)
        st.session_state.messages.append({"role":"assistant","content":answer}); st.rerun()
    except Exception as e:
        _log_exception("답변 생성 실패", e); st.error("답변 생성 중 오류. 우측 로그/Traceback 확인.")

# ===== [11] END OF FILE ======================================================
