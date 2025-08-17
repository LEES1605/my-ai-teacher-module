# app.py — 스텝 인덱싱(중간취소/재개) + 두뇌준비 안정화
#        + 인덱싱 보고서(스킵 표시)
#        + Drive 대화로그 저장(❶ OAuth: Markdown / ❷ 서비스계정: JSONL, chat_log/)
#        + 페르소나: 🤖Gemini(친절/꼼꼼), 🤖ChatGPT(유머러스/보완)

from __future__ import annotations
import os, time, uuid, re, json
import pandas as pd
import streamlit as st

# ============= 0) 페이지 설정 (반드시 최상단) ====================================
st.set_page_config(page_title="나의 AI 영어 교사", layout="wide", initial_sidebar_state="collapsed")

# ============= 1) 부트 가드 & 런타임 안정화 ===================================
ss = st.session_state
ss.setdefault("_boot_log", [])
ss.setdefault("_oauth_checked", False)

def _boot(msg: str): ss["_boot_log"].append(msg)

with st.sidebar:
    st.caption("🛠 Boot log (임시)")
    _boot_box = st.empty()

def _flush_boot():
    try:
        _boot_box.write("\n".join(ss["_boot_log"]) or "(empty)")
    except Exception:
        pass

_boot("A: page_config set"); _flush_boot()

# 런타임 튜닝(불필요한 리소스/경합 방지)
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ============= 2) 세션 키 초기화 ==============================================
ss.setdefault("session_id", uuid.uuid4().hex[:12])
ss.setdefault("messages", [])
ss.setdefault("auto_save_chatlog", True)    # OAuth Markdown 저장(내 드라이브)
ss.setdefault("save_logs", False)           # SA JSONL 저장(공유드라이브 writer 필요)
ss.setdefault("prep_both_running", False)
ss.setdefault("prep_both_done", ("qe_google" in ss) or ("qe_openai" in ss))
ss.setdefault("prep_cancel_requested", False)
ss.setdefault("session_terminated", False)
ss.setdefault("index_job", None)

# (진단) 하트비트
st.caption(f"heartbeat ✅ keys={list(ss.keys())[:8]}")

# ============= 3) 기본 UI 헤더/스타일 =========================================
from src.ui import load_css, render_header
load_css()
render_header()
st.info("✅ 변경이 있을 때만 인덱싱합니다. 저장된 두뇌가 있으면 즉시 로드합니다. (중간 취소/재개 지원)")

# ============= 4) OAuth 리다이렉트 처리(최종화 1회만) ==========================
#  - 이전 코드에 있었던 이중 호출을 제거하여 깔끔하게 유지
try:
    from src.google_oauth import finish_oauth_if_redirected
    if not st.secrets.get("OAUTH_DISABLE_FINISH"):
        if not ss.get("_oauth_finalized", False):
            finalized = finish_oauth_if_redirected()
            if finalized:
                ss["_oauth_finalized"] = True
                try:
                    st.query_params.clear()
                except Exception:
                    st.experimental_set_query_params()
                st.rerun()
except Exception as e:
    st.warning(f"OAuth finalize skipped: {e}")

# ============= 5) 사이드바: OAuth 로그인/로그아웃, 저장 옵션 ==================
from src.google_oauth import start_oauth, is_signed_in, build_drive_service, get_user_email, sign_out
with st.sidebar:
    ss.auto_save_chatlog = st.toggle(
        "대화 자동 저장 (OAuth/내 드라이브, Markdown)",
        value=ss.auto_save_chatlog
    )
    ss.save_logs = st.toggle(
        "대화 JSONL 저장 (서비스계정/chat_log/)",
        value=ss.save_logs,
        help="공유드라이브 Writer 권한 필요. 쿼터 문제 시 끄기 권장."
    )
    st.markdown("---")
    st.markdown("### Google 로그인 (내 드라이브 저장)")
    if not is_signed_in():
        if st.button("🔐 Google로 로그인"):
            url = start_oauth()
            st.markdown(f"[여기를 눌러 로그인하세요]({url})")
    else:
        st.success(f"로그인됨: {get_user_email() or '알 수 없음'}")
        if st.button("로그아웃"):
            sign_out()
            st.rerun()

# ============= 6) Google Drive 연결 테스트(안정화 버전) ========================
st.markdown("## 🔗 Google Drive 연결 테스트")
st.caption("서비스계정 저장은 공유드라이브 Writer 권한이 필요. 인덱싱은 Readonly면 충분합니다.")

from src.config import settings
from src.rag_engine import smoke_test_drive, preview_drive_files, drive_diagnostics

try:
    ok_sa, head_sa, details_sa = drive_diagnostics(settings.GDRIVE_FOLDER_ID)  # (ok, 헤드라인, 상세 리스트[str])
    if ok_sa:
        st.success(head_sa)
    else:
        st.warning(head_sa)
    with st.expander("서비스계정 JSON 진단 상세", expanded=not ok_sa):
        st.code("\n".join(details_sa), language="text")
except Exception as e:
    st.warning("진단 함수 예외:")
    st.code(
        f"{type(e).__name__}: {e}\n"
        f"타입={type(settings.GDRIVE_SERVICE_ACCOUNT_JSON).__name__}\n"
        f"프리뷰={str(settings.GDRIVE_SERVICE_ACCOUNT_JSON)[:200]}...",
        language="text"
    )

colL, colR = st.columns([0.65, 0.35], vertical_alignment="top")

with colL:
    if st.button("폴더 파일 미리보기 (최신 10개)", use_container_width=True):
        ok, msg, rows = preview_drive_files(max_items=10)
        if ok and rows:
            df = pd.DataFrame(rows)
            # 보기 친화적으로 가공
            df["type"] = df["mime"].str.replace("application/vnd.google-apps.", "", regex=False)
            df = df.rename(columns={"modified": "modified_at"})[["name","link","type","modified_at"]]
            st.dataframe(
                df,
                use_container_width=True,
                height=360,
                column_config={
                    "name": st.column_config.TextColumn("파일명"),
                    "link": st.column_config.LinkColumn("open", display_text="열기"),
                    "type": st.column_config.TextColumn("유형"),
                    "modified_at": st.column_config.TextColumn("수정시각"),
                },
                hide_index=True
            )
            st.success(f"총 {len(rows)}개 항목 표시 (최신 10개 기준).")
        elif ok:
            st.info("폴더에 파일이 없거나 접근할 수 없습니다.")
        else:
            st.error(msg)

with colR:
    ok, msg = smoke_test_drive()
    if ok:
        st.success(msg)     # ⚠️ 반환값을 st.write 등으로 다시 찍지 않음
    else:
        st.warning(msg)

# ============= 6.5) 📤 관리자: 자료 업로드 (원본 → prepared 저장) ===============
with st.expander("📤 관리자: 자료 업로드 (원본→prepared 저장)", expanded=False):
    st.caption(
        "원본 파일을 prepared 폴더에 저장합니다. 텍스트 추출물은 인덱스 캐시에만 저장됩니다.\n"
        "로컬 파일 업로드 + Google Docs/Slides/Sheets URL 가져오기 모두 지원합니다.\n"
        "옵션을 켜면 AI가 제목을 정해 파일명을 변경합니다."
    )

    # ── 옵션: AI가 제목 자동 생성 ────────────────────────────────────────────────
    auto_title = st.toggle("AI 제목 자동 생성(업로드/가져오기 후 이름 바꾸기)", value=True,
                           help="LLM이 짧고 명확한 제목을 뽑아 파일명을 바꿉니다. 키가 없으면 휴리스틱으로 제목 생성.")
    title_hint = st.text_input("제목 힌트(선택)", placeholder="예: 고1 영어 문법 / 학원 교재 / 중간고사 대비 등")

    # ── (A) 로컬 파일 업로드: 여러 형식 지원 ─────────────────────────────────────
    SUPPORTED_TYPES = [
        "pdf", "docx", "doc", "pptx", "ppt", "md", "txt", "rtf", "odt", "html", "epub",
        "xlsx", "xls", "csv"
    ]
    files = st.file_uploader(
        "로컬 파일 선택 (여러 개 가능)",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True
    )

    # ── (B) Google Docs/Slides/Sheets URL로 가져오기 (줄바꿈으로 여러 개) ───────
    gdocs_urls = st.text_area(
        "Google Docs/Slides/Sheets URL 붙여넣기 (여러 개면 줄바꿈으로 구분)",
        placeholder="예) https://docs.google.com/document/d/............/edit\nhttps://docs.google.com/presentation/d/............/edit",
        height=96
    )

    # 진행/상태 영역
    prog = st.progress(0, text="대기 중…")
    status_area = st.empty()
    result_area = st.empty()

    # ── 유틸: 타임스탬프/파일명 정리/확장자→MIME ─────────────────────────────────
    def _ts():
        import time
        return time.strftime("%Y%m%d_%H%M%S")

    def _safe_name(name: str) -> str:
        import re
        # Windows/NIX 금지문자 제거
        name = re.sub(r'[\\/:*?"<>|]+', " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name or "untitled"

    def _guess_mime_by_ext(fname: str) -> str:
        ext = (fname.rsplit(".", 1)[-1] if "." in fname else "").lower()
        MIMES = {
            "pdf":  "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "doc":  "application/msword",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "ppt":  "application/vnd.ms-powerpoint",
            "md":   "text/markdown",
            "txt":  "text/plain",
            "rtf":  "application/rtf",
            "odt":  "application/vnd.oasis.opendocument.text",
            "html": "text/html",
            "epub": "application/epub+zip",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "xls":  "application/vnd.ms-excel",
            "csv":  "text/csv",
        }
        return MIMES.get(ext, "application/octet-stream")

    def _parse_gdoc_id(s: str) -> str | None:
        import re
        s = s.strip()
        if not s:
            return None
        for pat in [r"/d/([-\w]{15,})", r"[?&]id=([-\w]{15,})$", r"^([-\w]{15,})$"]:
            m = re.search(pat, s)
            if m:
                return m.group(1)
        return None

# ── AI 제목 생성기(LLM + 휴리스틱) ────────────────────────────────────────────
from src.rag_engine import make_llm, llm_complete

def _get_title_model():
    """OpenAI 있으면 OpenAI, 없으면 Gemini, 둘 다 없으면 None"""
    global _TITLE_MODEL                      # ← nonlocal 대신 global 사용
    if _TITLE_MODEL is not None:
        return _TITLE_MODEL
    try:
        from src.config import settings
        if getattr(settings, "OPENAI_API_KEY", None) and settings.OPENAI_API_KEY.get_secret_value():
            _TITLE_MODEL = make_llm(
                "openai",
                settings.OPENAI_API_KEY.get_secret_value(),
                getattr(settings, "OPENAI_LLM_MODEL", "gpt-4o-mini"),
                0.2,
            )
            return _TITLE_MODEL
        # OpenAI 키 없으면 Gemini로
        _TITLE_MODEL = make_llm(
            "google",
            settings.GEMINI_API_KEY.get_secret_value(),
            getattr(settings, "LLM_MODEL", "gemini-1.5-pro"),
            0.2,
        )
        return _TITLE_MODEL
    except Exception:
        return None

def _heuristic_title(orig_base: str, hint: str = "") -> str:
    """확장자 제거된 원래 이름 + 힌트를 깔끔히 정리해 40자 내로"""
    import re
    base = orig_base
    base = re.sub(r"\.[^.]+$", "", base)            # .ext 제거
    base = re.sub(r"^\d{8}_\d{6}__", "", base)      # 타임스탬프 패턴 제거
    base = base.replace("_", " ").replace("-", " ")
    base = re.sub(r"\s+", " ", base).strip()
    if hint:
        base = f"{hint.strip()} — {base}" if base else hint.strip()
    return (base[:40]).strip() or "untitled"

def _ai_title(orig_base: str, sample_text: str = "", hint: str = "") -> str:
    """LLM으로 짧은 한국어 제목 생성(최대 40자). 실패 시 휴리스틱."""
    model = _get_title_model()
    if model is None:
        return _heuristic_title(orig_base, hint)

    prompt = (
        "다음 파일의 제목을 한국어로 간결하게 만들어 주세요. 규칙:\n"
        "1) 최대 40자, 2) 불필요한 숫자/확장자 제거, 3) 핵심 키워드 위주, 4) 따옴표/괄호 남발 금지,\n"
        "5) 문장형 말투보다 명사구 선호, 6) 출력은 제목만(부가 설명/따옴표 X).\n\n"
        f"[파일명 힌트]\n{orig_base}\n\n"
    )
    if hint:
        prompt += f"[추가 힌트]\n{hint}\n\n"
    if sample_text:
        prompt += f"[본문 일부]\n{sample_text[:1200]}\n\n"

    try:
        title = llm_complete(model, prompt).strip()
        # 안전화 (아래 _safe_name은 같은 섹션에 이미 정의되어 있어야 합니다)
        title = _safe_name(title)
        return (title[:40]).strip() or _heuristic_title(orig_base, hint)
    except Exception:
        return _heuristic_title(orig_base, hint)

    # ── 업로드/가져오기 실행 ─────────────────────────────────────────────────────
    if st.button("업로드/가져오기 → prepared", type="primary"):
        import io, time
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseUpload
        from googleapiclient.errors import HttpError
        from src.rag_engine import _normalize_sa
        from src.config import settings
        from src.google_oauth import is_signed_in, build_drive_service

        # 서비스계정 Drive(쓰기), OAuth Drive(있으면 읽기/복사)
        creds_sa = _normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON)
        drive_sa = build("drive", "v3", credentials=creds_sa)
        drive_oauth = build_drive_service() if is_signed_in() else None

        rows, done, total_steps = [], 0, 1
        if files: total_steps += len(files)
        url_list = [u.strip() for u in (gdocs_urls.splitlines() if gdocs_urls else []) if u.strip()]
        if url_list: total_steps += len(url_list)

        def _tick(msg):
            nonlocal done
            done += 1
            pct = int(done / max(total_steps, 1) * 100)
            prog.progress(pct, text=msg)
            status_area.info(msg)

        # 업로드/가져오기 결과(Drive id, name, link) 저장
        created = []  # [{id, name, link, ext, orig_base, sample_text?}]

        try:
            # 1) 로컬 파일 업로드
            if files:
                for f in files:
                    data = f.read()
                    buf = io.BytesIO(data)
                    base = _safe_name(f.name)
                    ext = (base.rsplit(".", 1)[-1].lower() if "." in base else "")
                    name = f"{_ts()}__{base}"
                    mime = _guess_mime_by_ext(base)

                    media = MediaIoBaseUpload(buf, mimetype=mime, resumable=False)
                    meta = {"name": name, "parents": [settings.GDRIVE_FOLDER_ID]}
                    _tick(f"업로드 중: {name}")
                    res = drive_sa.files().create(body=meta, media_body=media, fields="id,webViewLink").execute()
                    created.append({"id": res["id"], "name": name, "link": res.get("webViewLink",""),
                                    "ext": ext, "orig_base": base, "sample_text": ""})

                    time.sleep(0.05)

            # 2) Google 문서 링크 → export/copy 후 저장
            for raw in url_list:
                file_id = _parse_gdoc_id(raw)
                if not file_id:
                    rows.append({"name": f"(잘못된 링크) {raw[:40]}…", "open": ""})
                    _tick("잘못된 링크 건너뜀")
                    continue

                drive_ro = drive_oauth or drive_sa  # 로그인되어 있으면 OAuth 우선
                try:
                    meta = drive_ro.files().get(fileId=file_id, fields="id,name,mimeType").execute()
                    name0 = meta.get("name", "untitled")
                    mtype = meta.get("mimeType", "")
                except HttpError as he:
                    if drive_ro is drive_oauth:
                        try:
                            meta = drive_sa.files().get(fileId=file_id, fields="id,name,mimeType").execute()
                            name0 = meta.get("name", "untitled")
                            mtype = meta.get("mimeType", "")
                            drive_ro = drive_sa
                        except Exception as e2:
                            rows.append({"name": f"(접근 실패) {raw[:40]}…", "open": f"{type(e2).__name__}: 권한 필요"})
                            _tick("접근 실패(공유 필요)")
                            continue
                    else:
                        rows.append({"name": f"(접근 실패) {raw[:40]}…", "open": f"{type(he).__name__}: 권한 필요"})
                        _tick("접근 실패(공유 필요)")
                        continue

                GOOGLE_NATIVE = {
                    "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
                    "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
                    "application/vnd.google-apps.spreadsheet": ("application/pdf", ".pdf"),
                }

                if mtype in GOOGLE_NATIVE:
                    export_mime, ext = GOOGLE_NATIVE[mtype]
                    _tick(f"내보내는 중: {name0}{ext} (Google 문서)")
                    data = drive_ro.files().export(fileId=file_id, mimeType=export_mime).execute()
                    buf = io.BytesIO(data)
                    name = f"{_ts()}__{_safe_name(name0)}{ext}"
                    media = MediaIoBaseUpload(buf, mimetype=export_mime, resumable=False)
                    meta2 = {"name": name, "parents": [settings.GDRIVE_FOLDER_ID]}
                    res2 = drive_sa.files().create(body=meta2, media_body=media, fields="id,webViewLink").execute()
                    created.append({"id": res2["id"], "name": name, "link": res2.get("webViewLink",""),
                                    "ext": ext.strip("."), "orig_base": name0, "sample_text": ""})
                else:
                    # 네이티브가 아닌 경우: prepared로 복사
                    _tick(f"복사 중: {name0} (파일)")
                    body = {"name": f"{_ts()}__{_safe_name(name0)}", "parents": [settings.GDRIVE_FOLDER_ID]}
                    try:
                        res3 = drive_sa.files().copy(fileId=file_id, body=body, fields="id,webViewLink").execute()
                    except HttpError:
                        if drive_oauth:
                            res3 = drive_oauth.files().copy(fileId=file_id, body=body, fields="id,webViewLink").execute()
                        else:
                            rows.append({"name": f"(복사 실패) {name0}", "open": "권한 부족 — 서비스계정에 공유하거나 OAuth 로그인"})
                            continue
                    created.append({"id": res3["id"], "name": body["name"], "link": res3.get("webViewLink",""),
                                    "ext": "", "orig_base": name0, "sample_text": ""})
                    time.sleep(0.05)

            # 3) (옵션) AI 제목으로 파일명 변경
            renamed_rows = []
            if auto_title and created:
                used = set()
                for item in created:
                    fid = item["id"]
                    old_name = item["name"]
                    ext = f".{item['ext']}" if item.get("ext") else ""
                    # 샘플 텍스트 확보: 간단히 원래 이름만으로도 가능, 텍스트 파일/MD면 일부 본문까지
                    sample = ""
                    if item.get("ext") in {"txt", "md"}:
                        # 텍스트/MD는 원본 업로드 전에도 읽을 수 있지만 지금은 이미 업로드 상태.
                        # 간단히 파일명 기반으로도 충분. (추후 Drive download 후 본문 사용 가능)
                        sample = ""

                    ai_title = _ai_title(item.get("orig_base", old_name), sample_text=sample, hint=title_hint)
                    # 최종 파일명: 타임스탬프__AI제목 + 원래 확장자
                    new_name = f"{_ts()}__{ai_title}{ext}"
                    new_name = _safe_name(new_name)
                    # 중복 최소화(동일 배치 내)
                    k = new_name; n = 2
                    while k in used:
                        k = f"{new_name} ({n})"; n += 1
                    new_name = k; used.add(new_name)

                    try:
                        upd = drive_sa.files().update(fileId=fid, body={"name": new_name}).execute()
                        renamed_rows.append({"original": old_name, "renamed_to": new_name, "open": item["link"]})
                    except Exception as e:
                        renamed_rows.append({"original": old_name, "renamed_to": f"(이름 변경 실패) {e}", "open": item["link"]})
            else:
                for item in created:
                    renamed_rows.append({"original": item["name"], "renamed_to": "(AI 제목 생성 꺼짐)", "open": item["link"]})

            # 4) 결과 표시
            prog.progress(100, text="완료")
            status_area.success(f"총 {len(created)}개 항목 처리 완료 (prepared)")
            if renamed_rows:
                import pandas as pd
                df = pd.DataFrame(renamed_rows)
                result_area.dataframe(
                    df, use_container_width=True, hide_index=True,
                    column_config={
                        "original": st.column_config.TextColumn("원래 파일명"),
                        "renamed_to": st.column_config.TextColumn("변경 후 파일명"),
                        "open": st.column_config.LinkColumn("열기", display_text="열기")
                    }
                )
            st.toast("업로드/가져오기 완료 — 변경 사항은 인덱싱 시 반영됩니다.", icon="✅")

            # 인덱싱을 다시 돌릴 수 있도록 준비 버튼 재활성화
            ss.prep_both_done = False

        except Exception as e:
            prog.progress(0, text="오류")
            status_area.error(f"처리 실패: {e}")
# ==============================================================================

# ============= 6.7) 📚 문법서 토픽별 소책자 생성(Drive 저장) =====================
with st.expander("📚 문법서 토픽별 소책자 생성(Drive 저장)", expanded=False):
    st.caption(
        "원본은 prepared/에 그대로 두고, 문법 토픽별 최적화된 소책자(.md)를 "
        "prepared_volumes/ 하위 폴더에 저장합니다. overview.md와 manifest.json도 함께 생성합니다."
    )

    default_topics = [
        "Parts of Speech(품사)", "Articles(관사)", "Nouns & Pronouns(명사/대명사)",
        "Verbs & Tenses(시제: 현재/과거/완료/진행)", "Modals(조동사)", "Passive(수동태)",
        "Gerunds & Infinitives(동명사/부정사)", "Adjectives & Adverbs(형용사/부사/비교급)",
        "Prepositions(전치사)", "Phrasal Verbs(구동사)", "Conjunctions & Clauses(접속사/절)",
        "Conditionals(조건문)", "Relative Clauses(관계사절)", "Reported Speech(화법전환)",
        "Questions & Negation(의문문/부정문)", "Sentence Structure(문장구조·어순)"
    ]
    topics_text = st.text_area(
        "토픽 목록(줄바꿈으로 구분, 수정 가능)", 
        value="\n".join(default_topics), height=200
    )
    booklet_title = st.text_input("소책자 세트 제목(폴더명)", value="Grammar Booklets")
    make_citations = st.toggle("소책자 하단에 ‘참고 자료(출처)’ 포함", value=True)
    start_btn = st.button("토픽별 소책자 생성 → Drive 저장", type="primary", use_container_width=True)

    if start_btn:
        if "qe_google" not in ss:
            st.warning("먼저 상단의 [🚀 한 번에 준비하기]로 인덱스를 준비하세요.")
        else:
            import re, json, time, io
            import pandas as pd
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaIoBaseUpload
            from src.rag_engine import _normalize_sa, get_text_answer
            from src.config import settings

            def _ts(): return time.strftime("%Y%m%d_%H%M%S")
            def _safe(s: str) -> str:
                s = re.sub(r'[\\/:*?"<>|]+', " ", str(s))
                s = re.sub(r"\s+", " ", s).strip()
                return s[:120] or "untitled"

            # 1) Drive 준비: prepared_volumes/<세트명_타임스탬프>/ 생성
            creds = _normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON)
            drive = build("drive", "v3", credentials=creds)

            def _ensure_child(parent_id: str, name: str) -> str:
                q = (
                    f"'{parent_id}' in parents and name = '{name}' "
                    "and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
                )
                res = drive.files().list(
                    q=q, fields="files(id,name)", pageSize=1,
                    includeItemsFromAllDrives=True, supportsAllDrives=True
                ).execute()
                files = res.get("files", [])
                if files: 
                    return files[0]["id"]
                meta = {"name": name, "parents":[parent_id], "mimeType": "application/vnd.google-apps.folder"}
                f = drive.files().create(body=meta, fields="id").execute()
                return f["id"]

            parent_volumes_id = _ensure_child(settings.GDRIVE_FOLDER_ID, "prepared_volumes")
            set_name = f"{_safe(booklet_title)}_{_ts()}"
            set_folder = _ensure_child(parent_volumes_id, set_name)

            # 2) 토픽별 생성
            topics = [t.strip() for t in topics_text.splitlines() if t.strip()]
            prog = st.progress(0, text="생성 중…")
            table_rows, manifest = [], {"title": booklet_title, "created_at": _ts(), "items": []}

            for i, topic in enumerate(topics, start=1):
                # 프롬프트(원전 기반 요약)
                guide = (
                    "당신은 영어 문법 교사입니다. 아래 토픽을 학생용 소책자 형태로 정리하세요.\n"
                    f"• 토픽: {topic}\n"
                    "• 핵심 개념을 한국어로, 규칙/형태는 영문 혼용\n"
                    "• 예문 3~5개 (쉬운→중간 난이도), 한-영 병기\n"
                    "• 자주 하는 실수/오개념 3개 정리\n"
                    "• 미니 연습문제 5문항(+정답/해설)\n"
                    "• 분량 500~900자 내외\n"
                )
                if make_citations:
                    guide += "• 마지막에 ‘---\\n*참고 자료: 파일명 …’ 섹션 포함\n"

                # 생성
                md = get_text_answer(
                    ss["qe_google"],
                    f"[토픽]\n{topic}\n\n[과제]\n위 가이드를 따르되, 학생용 마크다운으로 작성",
                    guide,
                )
                name = f"{_safe(topic)}.md"

                # Drive에 업로드
                buf = io.BytesIO(md.encode("utf-8"))
                media = MediaIoBaseUpload(buf, mimetype="text/markdown", resumable=False)
                meta = {"name": name, "parents": [set_folder]}
                file = drive.files().create(body=meta, media_body=media, fields="id,webViewLink").execute()

                table_rows.append({"topic": topic, "open": file.get("webViewLink")})
                manifest["items"].append({"topic": topic, "file_id": file["id"], "name": name})

                prog.progress(int(i/len(topics)*100), text=f"[{i}/{len(topics)}] {topic}")

            # 3) overview.md & manifest.json 저장
            overview_lines = [f"# {booklet_title}", "", f"생성시각: {time.strftime('%Y-%m-%d %H:%M:%S')} (KST)", ""]
            for it in manifest["items"]:
                overview_lines.append(f"- {it['topic']} — {it['name']}")
            overview_md = "\n".join(overview_lines) + "\n"

            # overview
            ov_buf = io.BytesIO(overview_md.encode("utf-8"))
            ov_meta = {"name": "overview.md", "parents": [set_folder]}
            ov_media = MediaIoBaseUpload(ov_buf, mimetype="text/markdown", resumable=False)
            ov_file = drive.files().create(body=ov_meta, media_body=ov_media, fields="id,webViewLink").execute()

            # manifest
            mf_buf = io.BytesIO(json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
            mf_meta = {"name": "manifest.json", "parents": [set_folder]}
            mf_media = MediaIoBaseUpload(mf_buf, mimetype="application/json", resumable=False)
            mf_file = drive.files().create(body=mf_meta, media_body=mf_media, fields="id,webViewLink").execute()

            prog.progress(100, text="완료")
            st.success(f"총 {len(table_rows)}개 소책자 생성 → 폴더: prepared_volumes/{set_name}")
            if table_rows:
                st.dataframe(
                    pd.DataFrame(table_rows),
                    use_container_width=True, hide_index=True,
                    column_config={
                        "topic": st.column_config.TextColumn("토픽"),
                        "open": st.column_config.LinkColumn("열기", display_text="열기")
                    }
                )
            st.toast("Drive 저장 완료 — 원본은 prepared/ 유지, 소책자는 prepared_volumes/ 보관", icon="✅")
# ===============================================================================

# ============= 7) 인덱싱 보고서(스킵된 파일 포함) ===============================
rep = ss.get("indexing_report")
if rep:
    with st.expander("🧾 인덱싱 보고서 (스킵된 파일 보기)", expanded=False):
        st.write(
            f"총 파일(매니페스트): {rep.get('total_manifest')}, "
            f"로딩된 문서 수: {rep.get('loaded_docs')}, "
            f"스킵: {rep.get('skipped_count')}"
        )
        skipped = rep.get("skipped", [])
        if skipped:
            st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)
        else:
            st.caption("스킵된 파일이 없습니다 🎉")

# ============= 8) 두뇌 준비(증분 인덱싱 · 중간취소/재개) =======================
st.markdown("---")
st.subheader("🧠 두뇌 준비 — 저장본 로드 ↔ 변경 시 증분 인덱싱 (중간 취소/재개)")

c_g, c_o = st.columns(2)
with c_g: st.caption("Gemini 진행"); g_bar = st.empty(); g_msg = st.empty()
with c_o: st.caption("ChatGPT 진행"); o_bar = st.empty(); o_msg = st.empty()

def _render_progress(slot_bar, slot_msg, pct: int, msg: str | None = None):
    p = max(0, min(100, int(pct)))
    slot_bar.markdown(
        f"<div class='gp-wrap'><div class='gp-fill' style='width:{p}%'></div>"
        f"<div class='gp-label'>{p}%</div></div>",
        unsafe_allow_html=True
    )
    if msg is not None:
        slot_msg.markdown(f"<div class='gp-msg'>{msg}</div>", unsafe_allow_html=True)

def _is_cancelled() -> bool:
    return bool(ss.get("prep_cancel_requested", False))

from src.rag_engine import (
    set_embed_provider, make_llm, CancelledError,
    start_index_builder, resume_index_builder, cancel_index_builder,
    get_text_answer, llm_complete,   # 대화 단계에서 재사용
)

def run_prepare_both_step():
    # 1) 임베딩 설정 (OpenAI 우선, 없으면 Google)
    embed_provider = "openai"
    embed_api = (getattr(settings, "OPENAI_API_KEY", None).get_secret_value()
                 if getattr(settings, "OPENAI_API_KEY", None) else "")
    embed_model = getattr(settings, "OPENAI_EMBED_MODEL", "text-embedding-3-small")
    if not embed_api:
        embed_provider = "google"
        embed_api = settings.GEMINI_API_KEY.get_secret_value()
        embed_model = getattr(settings, "EMBED_MODEL", "text-embedding-004")
    try:
        _render_progress(g_bar, g_msg, 3, f"임베딩 설정({embed_provider})")
        _render_progress(o_bar, o_msg, 3, f"임베딩 설정({embed_provider})")
        set_embed_provider(embed_provider, embed_api, embed_model)
    except Exception as e:
        _render_progress(g_bar, g_msg, 100, f"임베딩 실패: {e}")
        _render_progress(o_bar, o_msg, 100, f"임베딩 실패: {e}")
        ss.prep_both_running = False
        return

    # 2) 인덱스 스텝 진행
    def upd(p, m=None):
        _render_progress(g_bar, g_msg, p, m)
        _render_progress(o_bar, o_msg, p, m)

    def umsg(m):
        _render_progress(g_bar, g_msg, ss.get("p_shared", 0), m)
        _render_progress(o_bar, o_msg, ss.get("p_shared", 0), m)

    persist_dir = f"{getattr(settings,'PERSIST_DIR','/tmp/my_ai_teacher/storage_gdrive')}_shared"
    job = ss.get("index_job")

    try:
        if job is None:
            res = start_index_builder(
                update_pct=upd,
                update_msg=umsg,
                gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
                raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
                persist_dir=persist_dir,
                manifest_path=getattr(settings, "MANIFEST_PATH", "/tmp/my_ai_teacher/drive_manifest.json"),
                max_docs=None,     # 전체 인덱싱
                is_cancelled=_is_cancelled,
            )
        else:
            res = resume_index_builder(
                job=job,
                update_pct=upd,
                update_msg=umsg,
                is_cancelled=_is_cancelled,
                batch_size=6
            )

        status = res.get("status")
        if status == "running":
            ss.index_job = res["job"]
            _render_progress(g_bar, g_msg, res.get("pct", 8), res.get("msg", "진행 중…"))
            _render_progress(o_bar, o_msg, res.get("pct", 8), res.get("msg", "진행 중…"))
            time.sleep(0.15)
            st.rerun()
            return

        if status == "cancelled":
            ss.prep_both_running = False
            ss.prep_cancel_requested = False
            ss.index_job = None
            _render_progress(g_bar, g_msg, res.get("pct", 0), "사용자 취소")
            _render_progress(o_bar, o_msg, res.get("pct", 0), "사용자 취소")
            return

        if status != "done":
            _render_progress(g_bar, g_msg, 100, "인덱싱 실패")
            _render_progress(o_bar, o_msg, 100, "인덱싱 실패")
            ss.prep_both_running = False
            return

        index = res["index"]
        ss.index_job = None

    except Exception as e:
        ss.prep_both_running = False
        ss.index_job = None
        _render_progress(g_bar, g_msg, 100, f"에러: {e}")
        _render_progress(o_bar, o_msg, 100, f"에러: {e}")
        return

    # 3) QueryEngine 생성
    try:
        g_llm = make_llm(
            "google",
            settings.GEMINI_API_KEY.get_secret_value(),
            getattr(settings, "LLM_MODEL", "gemini-1.5-pro"),
            float(ss.get("temperature", 0.0))
        )
        ss["llm_google"] = g_llm
        ss["qe_google"] = index.as_query_engine(
            llm=g_llm,
            response_mode=ss.get("response_mode", getattr(settings, "RESPONSE_MODE", "compact")),
            similarity_top_k=int(ss.get("similarity_top_k", getattr(settings, "SIMILARITY_TOP_K", 5))),
        )
        _render_progress(g_bar, g_msg, 100, "완료!")
    except Exception as e:
        _render_progress(g_bar, g_msg, 100, f"Gemini 준비 실패: {e}")

    try:
        if getattr(settings, "OPENAI_API_KEY", None) and settings.OPENAI_API_KEY.get_secret_value():
            o_llm = make_llm(
                "openai",
                settings.OPENAI_API_KEY.get_secret_value(),
                getattr(settings, "OPENAI_LLM_MODEL", "gpt-4o-mini"),
                float(ss.get("temperature", 0.0))
            )
            ss["llm_openai"] = o_llm
            ss["qe_openai"] = index.as_query_engine(
                llm=o_llm,
                response_mode=ss.get("response_mode", getattr(settings, "RESPONSE_MODE", "compact")),
                similarity_top_k=int(ss.get("similarity_top_k", getattr(settings, "SIMILARITY_TOP_K", 5))),
            )
            _render_progress(o_bar, o_msg, 100, "완료!")
        else:
            _render_progress(o_bar, o_msg, 100, "키 누락 — OPENAI_API_KEY 필요")
    except Exception as e:
        _render_progress(o_bar, o_msg, 100, f"ChatGPT 준비 실패: {e}")

    ss.prep_both_running = False
    ss.prep_both_done = True
    time.sleep(0.2)
    st.rerun()

# 실행/취소 버튼 줄
left, right = st.columns([0.7, 0.3])
with left:
    clicked = st.button(
        "🚀 한 번에 준비하기",
        key="prepare_both",
        use_container_width=True,
        disabled=ss.prep_both_running or ss.prep_both_done
    )
with right:
    cancel_clicked = st.button(
        "⛔ 준비 취소",
        key="cancel_prepare",
        use_container_width=True,
        type="secondary",
        disabled=not ss.prep_both_running
    )

from src.rag_engine import cancel_index_builder  # 재노출(명시)

if cancel_clicked and ss.prep_both_running:
    ss.prep_cancel_requested = True
    if ss.get("index_job"):
        cancel_index_builder(ss.index_job)
    st.rerun()

if clicked and not (ss.prep_both_running or ss.prep_both_done):
    ss.prep_cancel_requested = False
    ss.prep_both_running = True
    ss.index_job = None
    st.rerun()

if ss.prep_both_running:
    run_prepare_both_step()

st.caption("준비 버튼을 다시 활성화하려면 아래 재설정 버튼을 누르세요.")
if st.button("🔧 재설정(버튼 다시 활성화)", disabled=not ss.prep_both_done):
    ss.prep_both_done = False
    st.rerun()

# ============= 9) 대화 UI (그룹토론) ===========================================
st.markdown("---")
st.subheader("💬 그룹토론 — 학생 ↔ 🤖Gemini(친절/꼼꼼) ↔ 🤖ChatGPT(유머러스/보완)")

ready_google = "qe_google" in ss
ready_openai = "qe_openai" in ss

if ss.session_terminated:
    st.warning("세션이 종료된 상태입니다. 새로고침으로 다시 시작하세요.")
    st.stop()

if not ready_google:
    st.info("먼저 **[🚀 한 번에 준비하기]**로 두뇌를 준비하세요. (OpenAI 키 없으면 Gemini만 응답)")
    st.stop()

# 과거 메시지 렌더
for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 맥락/도우미
def _strip_sources(text: str) -> str:
    return re.sub(r"\n+---\n\*참고 자료:.*$", "", text, flags=re.DOTALL)

def _build_context(messages, limit_pairs=2, max_chars=2000) -> str:
    pairs, buf_user = [], None
    for m in reversed(messages):
        role, content = m.get("role"), str(m.get("content","")).strip()
        if role == "assistant":
            content = re.sub(r"^\*\*🤖 .*?\*\*\s*\n+", "", content).strip()
            if buf_user is not None:
                pairs.append((buf_user, content))
                buf_user = None
                if len(pairs) >= limit_pairs:
                    break
        elif role == "user" and buf_user is None:
            buf_user = content
    pairs = list(reversed(pairs))
    blocks = [f"[학생]\n{u}\n\n[교사]\n{a}" for u,a in pairs]
    ctx = "\n\n---\n\n".join(blocks).strip()
    return ctx[-max_chars:] if len(ctx) > max_chars else ctx

from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
def _persona():
    mode = ss.get("mode_select", "💬 이유문법 설명")
    base = EXPLAINER_PROMPT if mode=="💬 이유문법 설명" else (ANALYST_PROMPT if mode=="🔎 구문 분석" else READER_PROMPT)
    common = "역할: 학생의 영어 실력을 돕는 AI 교사.\n규칙: 근거가 불충분하면 그 사실을 명확히 밝힌다. 예시는 짧고 점진적으로."
    return base + "\n" + common

GEMINI_STYLE = "당신은 착하고 똑똑한 친구 같은 교사입니다. 칭찬과 격려, 정확한 설명."
CHATGPT_STYLE = (
    "당신은 유머러스하지만 정확한 동료 교사입니다. 동료(Gemini)의 답을 읽고 "
    "빠진 부분을 보완/교정하고 마지막에 <최종 정리>로 요약하세요. 과한 농담 금지."
)

mode = st.radio(
    "학습 모드",
    ["💬 이유문법 설명", "🔎 구문 분석", "📚 독해 및 요약"],
    horizontal=True, key="mode_select"
)

# 서비스계정 JSONL 저장 (chat_log/)
from src import chat_store
from src.drive_log import get_chatlog_folder_id, save_chatlog_markdown_oauth

def _jsonl_log(items):
    if not ss.save_logs:
        return
    try:
        parent_id = (getattr(settings, "CHATLOG_FOLDER_ID", None) or settings.GDRIVE_FOLDER_ID)
        sa = settings.GDRIVE_SERVICE_ACCOUNT_JSON
        if isinstance(sa, str):
            try:
                sa = json.loads(sa)
            except Exception:
                pass
        sub_id = get_chatlog_folder_id(parent_folder_id=parent_id, sa_json=sa)
        chat_store.append_jsonl(folder_id=sub_id, sa_json=sa, items=items)
        st.toast("대화 JSONL 저장 완료", icon="💾")
    except Exception as e:
        st.warning(f"대화 JSONL 저장 실패: {e}")

# 입력 & 응답
user_input = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
if user_input:
    ss.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    _jsonl_log([chat_store.make_entry(
        ss.session_id, "user", "user", user_input, mode, model="user"
    )])

    # Gemini 1차
    with st.spinner("🤖 Gemini 선생님이 먼저 답합니다…"):
        prev_ctx = _build_context(ss.messages[:-1], limit_pairs=2, max_chars=2000)
        gemini_query = (f"[이전 대화]\n{prev_ctx}\n\n" if prev_ctx else "") + f"[학생 질문]\n{user_input}"
        ans_g = get_text_answer(ss["qe_google"], gemini_query, _persona() + "\n" + GEMINI_STYLE)

    content_g = f"**🤖 Gemini**\n\n{ans_g}"
    ss.messages.append({"role":"assistant","content":content_g})
    with st.chat_message("assistant"):
        st.markdown(content_g)

    _jsonl_log([chat_store.make_entry(
        ss.session_id, "assistant", "Gemini", content_g, mode, model=getattr(settings,"LLM_MODEL","gemini")
    )])

    # ChatGPT 보완(키가 있을 때)
    if ready_openai:
        review_directive = (
            "역할: 동료 AI 영어교사\n"
            "목표: [이전 대화], [학생 질문], [동료의 1차 답변]을 읽고 사실오류/빠진점/모호함을 보완.\n"
            "지침: 1)핵심 간결 재정리 2)틀린 부분 근거와 교정 3)예문 2~3개 4)<최종 정리>로 요약. 외부검색 금지."
        )
        prev_all = _build_context(ss.messages, limit_pairs=2, max_chars=2000)
        augmented = (
            (f"[이전 대화]\n{prev_all}\n\n" if prev_all else "") +
            f"[학생 질문]\n{user_input}\n\n"
            f"[동료의 1차 답변(Gemini)]\n{_strip_sources(ans_g)}\n\n"
            "[당신의 작업]\n위 기준으로만 보완/검증."
        )
        with st.spinner("🤝 ChatGPT 선생님이 보완/검증 중…"):
            ans_o = llm_complete(
                ss.get("llm_openai"),
                _persona() + "\n" + CHATGPT_STYLE + "\n\n" + review_directive + "\n\n" + augmented
            )
        content_o = f"**🤖 ChatGPT**\n\n{ans_o}"
        ss.messages.append({"role":"assistant","content":content_o})
        with st.chat_message("assistant"):
            st.markdown(content_o)
    else:
        with st.chat_message("assistant"):
            st.info("ChatGPT 키가 없어 Gemini만 응답했습니다. OPENAI_API_KEY를 추가하면 보완/검증이 활성화됩니다.")

    # OAuth Markdown 저장(내 드라이브)
    if ss.auto_save_chatlog and ss.messages:
        try:
            if is_signed_in():
                svc = build_drive_service()
                parent_id = (st.secrets.get("OAUTH_CHAT_PARENT_ID") or "").strip() or None
                _fid = save_chatlog_markdown_oauth(ss.session_id, ss.messages, svc, parent_id)
                st.toast("내 드라이브에 대화 저장 완료 ✅", icon="💾")
            else:
                st.info("구글 계정으로 로그인하면 대화가 **내 드라이브**에 저장됩니다.")
        except Exception as e:
            st.warning(f"OAuth 저장 실패: {e}")

# EOF
