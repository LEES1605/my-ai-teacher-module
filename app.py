# app.py — 관리자 초기 "완전 스캔(heavy) + 숫자 진행률 + 멈춤 감지" 지원
#        + prepared/ 직접 투입 감지(중복 스킵) + 업로드/가져오기
#        + 증분 인덱싱(중간취소/재개) + 소책자 생성 자동화
#        + Drive 대화로그 저장(OAuth/서비스계정) + 그룹토론 UI(Gemini+ChatGPT)

from __future__ import annotations
import os, time, uuid, re, json, io, hashlib
import pandas as pd
import streamlit as st

# ================= 0) 페이지/런타임 기본 =================
st.set_page_config(page_title="나의 AI 영어 교사", layout="wide", initial_sidebar_state="collapsed")
ss = st.session_state
ss.setdefault("_boot_log", []); ss.setdefault("_oauth_checked", False)

def _boot(msg: str): ss["_boot_log"].append(msg)
with st.sidebar:
    st.caption("🛠 Boot log (임시)"); _boot_box = st.empty()
def _flush_boot():
    try: _boot_box.write("\n".join(ss["_boot_log"]) or "(empty)")
    except Exception: pass
_boot("A: page_config set"); _flush_boot()

# 런타임 튜닝
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ================= 1) 세션 키 초기화 ====================
ss.setdefault("session_id", uuid.uuid4().hex[:12])
ss.setdefault("messages", [])
ss.setdefault("auto_save_chatlog", True)
ss.setdefault("save_logs", False)
ss.setdefault("prep_both_running", False)
ss.setdefault("prep_both_done", ("qe_google" in ss) or ("qe_openai" in ss))
ss.setdefault("prep_cancel_requested", False)
ss.setdefault("session_terminated", False)
ss.setdefault("index_job", None)

# 업로드/감시 파이프라인
ss.setdefault("_auto_booklets_pending", False)
ss.setdefault("_auto_topics_text", "")
ss.setdefault("_auto_booklet_title", "Grammar Booklets")
ss.setdefault("_auto_make_citations", True)

# 폴더 감시 옵션
ss.setdefault("_watch_on_load", True)            # 페이지 열리면 자동 스캔
ss.setdefault("_watch_ran_once", False)          # 무한 rerun 방지
ss.setdefault("_watch_rename_title", False)      # 감시 시 AI 제목 정리 여부

# 관리자 전용 상태
ss.setdefault("_admin_initial_fullscan_done", False)
ss.setdefault("_admin_scan_cancel", False)

# 진행/워치독 설정
STALL_WARN_SEC = int(os.getenv("STALL_WARN_SEC", "20"))   # N초 이상 무진행 → 경고
YIELD_SLEEP    = float(os.getenv("YIELD_SLEEP", "0.05"))  # 루프 UI 갱신 텀

# Fast-scan 노브(일반 사용자 자동 감시용; 관리자는 heavy 우선)
FAST_LIST_LIMIT = int(os.getenv("PREPARED_FAST_LIST_LIMIT", "400"))
FAST_HASH_LIMIT = int(os.getenv("PREPARED_FAST_HASH_LIMIT", "40"))

st.caption(f"heartbeat ✅ keys={list(ss.keys())[:8]}")

# ================= 2) 헤더/스타일 =======================
from src.ui import load_css, render_header
load_css(); render_header()
st.info("✅ 첫 방문(관리자)은 정확도를 위해 **완전 스캔**을 수행합니다. 진행률/지연 경고로 상태를 확인할 수 있어요.")

# ================= 3) OAuth 리다이렉트 ===================
try:
    from src.google_oauth import finish_oauth_if_redirected
    if not st.secrets.get("OAUTH_DISABLE_FINISH"):
        if not ss.get("_oauth_finalized", False):
            finalized = finish_oauth_if_redirected()
            if finalized:
                ss["_oauth_finalized"] = True
                try: st.query_params.clear()
                except Exception: st.experimental_set_query_params()
                st.rerun()
except Exception as e:
    st.warning(f"OAuth finalize skipped: {e}")

# ================= 4) 사이드바: 로그인/옵션 ==============
from src.google_oauth import start_oauth, is_signed_in, build_drive_service, get_user_email, sign_out
with st.sidebar:
    ss.auto_save_chatlog = st.toggle("대화 자동 저장 (OAuth/내 드라이브, Markdown)", value=ss.auto_save_chatlog)
    ss.save_logs = st.toggle("대화 JSONL 저장 (서비스계정/chat_log/)", value=ss.save_logs,
                             help="공유드라이브 Writer 권한 필요. 쿼터 문제 시 끄기 권장.")
    st.markdown("---")
    st.markdown("### Google 로그인 (내 드라이브 저장)")
    if not is_signed_in():
        if st.button("🔐 Google로 로그인"):
            url = start_oauth(); st.markdown(f"[여기를 눌러 로그인하세요]({url})")
    else:
        st.success(f"로그인됨: {get_user_email() or '알 수 없음'}")
        if st.button("로그아웃"): sign_out(); st.rerun()

# ================= 5) 관리자 판별 ========================
def _admin_emails() -> set[str]:
    raw = str(st.secrets.get("ADMIN_EMAILS", "")).strip()
    return set([e.strip().lower() for e in raw.split(",") if e.strip()])
def _is_admin() -> bool:
    if str(st.secrets.get("ADMIN_MODE_ALWAYS_ON","0")).strip() == "1":
        return True
    try:
        return (get_user_email() or "").lower() in _admin_emails()
    except Exception:
        return False

IS_ADMIN = _is_admin()

# ================= 6) Drive 연결 테스트 ==================
st.markdown("## 🔗 Google Drive 연결 테스트")
st.caption("서비스계정 저장은 공유드라이브 Writer 권한이 필요. 인덱싱은 Readonly면 충분합니다.")

from src.config import settings
from src.rag_engine import smoke_test_drive, preview_drive_files, drive_diagnostics
try:
    ok_sa, head_sa, details_sa = drive_diagnostics(settings.GDRIVE_FOLDER_ID)
    if ok_sa: st.success(head_sa)
    else:     st.warning(head_sa)
    with st.expander("서비스계정 JSON 진단 상세", expanded=not ok_sa):
        st.code("\n".join(details_sa), language="text")
except Exception as e:
    st.warning("진단 함수 예외:")
    st.code(f"{type(e).__name__}: {e}\n타입={type(settings.GDRIVE_SERVICE_ACCOUNT_JSON).__name__}\n"
            f"프리뷰={str(settings.GDRIVE_SERVICE_ACCOUNT_JSON)[:200]}...", language="text")

colL, colR = st.columns([0.65, 0.35], vertical_alignment="top")
with colL:
    if st.button("폴더 파일 미리보기 (최신 10개)", use_container_width=True):
        ok, msg, rows = preview_drive_files(max_items=10)
        if ok and rows:
            df = pd.DataFrame(rows)
            df["type"] = df["mime"].str.replace("application/vnd.google-apps.", "", regex=False)
            df = df.rename(columns={"modified": "modified_at"})[["name","link","type","modified_at"]]
            st.dataframe(df, use_container_width=True, height=360,
                         column_config={"name": st.column_config.TextColumn("파일명"),
                                        "link": st.column_config.LinkColumn("open", display_text="열기"),
                                        "type": st.column_config.TextColumn("유형"),
                                        "modified_at": st.column_config.TextColumn("수정시각")},
                         hide_index=True)
            st.success(f"총 {len(rows)}개 항목 표시 (최신 10개 기준).")
        elif ok: st.info("폴더에 파일이 없거나 접근할 수 없습니다.")
        else:    st.error(msg)

with colR:
    ok, msg = smoke_test_drive()
    if ok: st.success(msg)
    else:  st.warning(msg)

# ================= 7) 유틸 ===============================
def _ts(): return time.strftime("%Y%m%d_%H%M%S")
def _safe_name(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', " ", str(name)); name = re.sub(r"\s+", " ", name).strip()
    return name or "untitled"
def _guess_mime_by_ext(fname: str) -> str:
    ext = (fname.rsplit(".", 1)[-1] if "." in fname else "").lower()
    MIMES = {
        "pdf":"application/pdf","docx":"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "doc":"application/msword","pptx":"application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "ppt":"application/vnd.ms-powerpoint","md":"text/markdown","txt":"text/plain","rtf":"application/rtf",
        "odt":"application/vnd.oasis.opendocument.text","html":"text/html","epub":"application/epub+zip",
        "xlsx":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet","xls":"application/vnd.ms-excel","csv":"text/csv",
    }; return MIMES.get(ext, "application/octet-stream")

# 제목 생성기
_TITLE_MODEL = None
from src.rag_engine import make_llm, llm_complete
def _get_title_model():
    global _TITLE_MODEL
    if _TITLE_MODEL is not None: return _TITLE_MODEL
    try:
        if getattr(settings, "OPENAI_API_KEY", None) and settings.OPENAI_API_KEY.get_secret_value():
            _TITLE_MODEL = make_llm("openai", settings.OPENAI_API_KEY.get_secret_value(),
                                    getattr(settings,"OPENAI_LLM_MODEL","gpt-4o-mini"), 0.2)
        else:
            _TITLE_MODEL = make_llm("google", settings.GEMINI_API_KEY.get_secret_value(),
                                    getattr(settings,"LLM_MODEL","gemini-1.5-pro"), 0.2)
    except Exception:
        _TITLE_MODEL = None
    return _TITLE_MODEL
def _heuristic_title(orig_base: str, hint: str = "") -> str:
    base = re.sub(r"\.[^.]+$", "", orig_base)
    base = re.sub(r"^\d{8}_\d{6}__", "", base)
    base = base.replace("_"," ").replace("-"," "); base = re.sub(r"\s+"," ",base).strip()
    if hint: base = f"{hint.strip()} — {base}" if base else hint.strip()
    return (base[:40]).strip() or "untitled"
def _ai_title(orig_base: str, sample_text: str = "", hint: str = "") -> str:
    model = _get_title_model()
    if model is None: return _heuristic_title(orig_base, hint)
    prompt = (
        "다음 파일의 제목을 한국어로 간결하게 만들어 주세요. 규칙:\n"
        "1) 최대 40자, 2) 불필요한 숫자/확장자 제거, 3) 핵심 키워드 위주, 4) 따옴표/괄호 남발 금지,\n"
        "5) 문장형 말투보다 명사구 선호, 6) 출력은 제목만(부가 설명/따옴표 X).\n\n"
        f"[파일명 힌트]\n{orig_base}\n\n"
    )
    if hint: prompt += f"[추가 힌트]\n{hint}\n\n"
    if sample_text: prompt += f"[본문 일부]\n{sample_text[:1200]}\n\n"
    try:
        title = llm_complete(model, prompt).strip()
        return (_safe_name(title)[:40]).strip() or _heuristic_title(orig_base, hint)
    except Exception:
        return _heuristic_title(orig_base, hint)

# ================= 8) 소책자 생성 =======================
def generate_booklets_drive(topics: list[str], booklet_title: str, make_citations: bool=True):
    """ss['qe_google'] 기반으로 topics별 .md 생성 → prepared_volumes/<세트명_타임스탬프>/ 저장"""
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    from src.rag_engine import _normalize_sa, get_text_answer

    def _safe(s: str) -> str:
        s = re.sub(r'[\\/:*?"<>|]+', " ", str(s)); s = re.sub(r"\s+", " ", s).strip(); return s[:120] or "untitled"

    creds = _normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON)
    drive = build("drive","v3",credentials=creds)

    def _ensure_child(parent_id: str, name: str) -> str:
        q = (f"'{parent_id}' in parents and name = '{name}' "
             "and mimeType = 'application/vnd.google-apps.folder' and trashed = false")
        res = drive.files().list(q=q, fields="files(id,name)", pageSize=1,
                                 includeItemsFromAllDrives=True, supportsAllDrives=True).execute()
        fs = res.get("files", []); 
        if fs: return fs[0]["id"]
        meta = {"name": name, "parents":[parent_id], "mimeType":"application/vnd.google-apps.folder"}
        f = drive.files().create(body=meta, fields="id").execute(); return f["id"]

    parent_volumes_id = _ensure_child(settings.GDRIVE_FOLDER_ID, "prepared_volumes")
    set_name = f"{_safe(booklet_title)}_{_ts()}"; set_folder = _ensure_child(parent_volumes_id, set_name)

    rows, manifest = [], {"title": booklet_title, "created_at": _ts(), "items": []}
    for topic in topics:
        guide = ("당신은 영어 문법 교사입니다. 아래 토픽을 학생용 소책자 형태로 정리하세요.\n"
                 f"• 토픽: {topic}\n"
                 "• 핵심 개념은 한국어, 규칙/형태는 영문 혼용\n"
                 "• 예문 3~5개(쉬움→중간), 한-영 병기\n"
                 "• 자주 하는 실수 3개\n"
                 "• 미니 연습문제 5문항(+정답/해설)\n"
                 "• 500~900자 내외\n")
        if make_citations: guide += "• 마지막에 ‘---\\n*참고 자료: 파일명 …’ 섹션 포함\n"
        md = get_text_answer(ss["qe_google"], f"[토픽]\n{topic}\n\n[과제]\n위 가이드로 학생용 마크다운 작성", guide)
        name = f"{_safe(topic)}.md"; buf = io.BytesIO(md.encode("utf-8"))
        media = MediaIoBaseUpload(buf, mimetype="text/markdown", resumable=False)
        meta = {"name": name, "parents": [set_folder]}
        file = drive.files().create(body=meta, media_body=media, fields="id,webViewLink").execute()
        rows.append({"topic": topic, "open": file.get("webViewLink")})
        manifest["items"].append({"topic": topic, "file_id": file["id"], "name": name})

    # overview & manifest
    ov = [f"# {booklet_title}", "", f"생성시각: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for it in manifest["items"]: ov.append(f"- {it['topic']} — {it['name']}")
    overview_md = "\n".join(ov) + "\n"

    from googleapiclient.http import MediaIoBaseUpload
    drive.files().create(
        body={"name": "overview.md", "parents": [set_folder]},
        media_body=MediaIoBaseUpload(io.BytesIO(overview_md.encode("utf-8")), mimetype="text/markdown", resumable=False),
        fields="id").execute()
    drive.files().create(
        body={"name": "manifest.json", "parents": [set_folder]},
        media_body=MediaIoBaseUpload(io.BytesIO(json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")),
                                     mimetype="application/json", resumable=False),
        fields="id").execute()
    return {"folder_name": set_name, "folder_id": set_folder, "rows": rows}

# ================= 9) prepared 감시/매니페스트 ==========
from googleapiclient.discovery import build as _build_gd
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from src.rag_engine import _normalize_sa
def _build_sa_drive():
    creds = _normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON)
    return _build_gd("drive","v3",credentials=creds)
def _find_file(drive, parent_id: str, name: str):
    q = f"'{parent_id}' in parents and name = '{name}' and trashed = false"
    res = drive.files().list(q=q, fields="files(id,name,mimeType)", pageSize=1,
                             includeItemsFromAllDrives=True, supportsAllDrives=True).execute()
    fs = res.get("files", []); return (fs[0]["id"] if fs else None)
def _load_manifest(drive):
    mid = _find_file(drive, settings.GDRIVE_FOLDER_ID, "prepared_manifest.json")
    data = {"items":{}, "updated_at": _ts()}
    if mid:
        req = drive.files().get_media(fileId=mid); buf = io.BytesIO(); MediaIoBaseDownload(buf, req).next_chunk()
        try: data = json.loads(buf.getvalue().decode("utf-8"))
        except Exception: pass
    return mid, data
def _save_manifest(drive, manifest, file_id=None):
    body = MediaIoBaseUpload(io.BytesIO(json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")),
                             mimetype="application/json", resumable=False)
    if file_id:
        drive.files().update(fileId=file_id, media_body=body).execute()
    else:
        drive.files().create(body={"name":"prepared_manifest.json","parents":[settings.GDRIVE_FOLDER_ID]},
                             media_body=body, fields="id").execute()
def _list_prepared(drive, max_items: int | None = None):
    q = f"'{settings.GDRIVE_FOLDER_ID}' in parents and trashed = false"
    fields = "files(id,name,mimeType,modifiedTime,version,md5Checksum,size), nextPageToken"
    token=None; items=[]; cap = max_items or FAST_LIST_LIMIT
    while True:
        page_size = min(1000, max(1, cap-len(items)))
        res = drive.files().list(q=q, fields=fields, pageSize=page_size, pageToken=token,
                                 includeItemsFromAllDrives=True, supportsAllDrives=True).execute()
        items.extend(res.get("files", [])); 
        if len(items) >= cap: break
        token=res.get("nextPageToken")
        if not token: break
    return items
def _list_prepared_all(drive):
    q = f"'{settings.GDRIVE_FOLDER_ID}' in parents and trashed = false"
    fields = "files(id,name,mimeType,modifiedTime,version,md5Checksum,size), nextPageToken"
    token=None; items=[]
    while True:
        res = drive.files().list(q=q, fields=fields, pageSize=1000, pageToken=token,
                                 includeItemsFromAllDrives=True, supportsAllDrives=True).execute()
        items.extend(res.get("files", [])); token=res.get("nextPageToken")
        if not token: break
    return items
def _file_sig(drive, meta: dict, fast: bool = False) -> str:
    """
    내용 서명:
      - 일반 파일: md5Checksum 우선. 없으면 (fast=True) 임시서명(size+mtime) 또는 (fast=False) 바이너리 SHA256
      - Google 문서류: (fast=True) version+mtime 임시서명, (fast=False) PDF export → SHA256
    """
    mt = meta.get("mimeType", ""); fid = meta["id"]
    if not mt.startswith("application/vnd.google-apps/"):
        if meta.get("md5Checksum"): return "md5:"+meta["md5Checksum"]
        if fast:
            return f"tmp:{meta.get('size','?')}@{meta.get('modifiedTime','')}"
        req = drive.files().get_media(fileId=fid); buf = io.BytesIO(); MediaIoBaseDownload(buf, req).next_chunk()
        return "sha256:"+hashlib.sha256(buf.getvalue()).hexdigest()
    if fast:
        return f"ver:{meta.get('version','?')}@{meta.get('modifiedTime','')}"
    data = drive.files().export(fileId=fid, mimeType="application/pdf").execute()
    return "sha256:"+hashlib.sha256(data).hexdigest()

# ---- 일반 사용자용(빠른 스캔) ----
def watch_scan_and_schedule(topics_text: str, title: str, cite: bool, rename_title: bool=False):
    drive = _build_sa_drive()
    mid, manifest = _load_manifest(drive)
    known: dict = manifest.get("items", {})
    pending_full: list[str] = manifest.get("pending_full", [])
    heavy_budget = FAST_HASH_LIMIT

    # 대기열 일부 확정
    if pending_full and heavy_budget > 0:
        still=[]
        for fid in pending_full:
            if heavy_budget<=0: still.append(fid); continue
            meta = drive.files().get(fileId=fid, fields="id,name,mimeType,modifiedTime,version,md5Checksum,size").execute()
            try:
                sig = _file_sig(drive, meta, fast=False); heavy_budget -= 1
                known[fid] = {"name": meta.get("name","untitled"), "sig": sig, "processed_at": known.get(fid,{}).get("processed_at")}
            except Exception:
                still.append(fid)
        pending_full = still

    metas = _list_prepared(drive, max_items=FAST_LIST_LIMIT)
    name_sig_set = {(v.get("name"), v.get("sig")) for v in known.values()}
    new_or_changed = []
    for m in metas:
        fid, name = m["id"], m.get("name","untitled")
        if name == "prepared_manifest.json": continue
        is_google = m.get("mimeType","").startswith("application/vnd.google-apps/")
        has_md5 = bool(m.get("md5Checksum"))
        needs_heavy = (is_google or not has_md5)
        try:
            sig = _file_sig(drive, m, fast=not needs_heavy or heavy_budget<=0)
            if needs_heavy and heavy_budget<=0 and fid not in pending_full: pending_full.append(fid)
            if needs_heavy and heavy_budget>0:
                heavy_budget -= 1
        except Exception as e:
            st.warning(f"시그니처 실패: {name} → {e}")
            continue
        prev = known.get(fid)
        if prev and prev.get("sig")==sig: continue
        if (name, sig) in name_sig_set:
            known[fid] = {"name": name, "sig": sig, "processed_at": prev.get("processed_at") if prev else _ts()}
            continue
        new_or_changed.append({"id":fid,"name":name,"sig":sig})
        known[fid] = {"name": name, "sig": sig, "processed_at": None}
        name_sig_set.add((name, sig))

    manifest["items"] = known; manifest["pending_full"] = pending_full; manifest["updated_at"] = _ts()
    _save_manifest(drive, manifest, mid)

    if not new_or_changed:
        return False, 0

    ss["_auto_booklets_pending"] = True
    ss["_auto_topics_text"] = topics_text
    ss["_auto_booklet_title"] = title
    ss["_auto_make_citations"] = cite
    ss.prep_both_done = False; ss.prep_both_running = True; ss.index_job = None
    st.info(f"prepared/ 신규/변경 {len(new_or_changed)}개 감지 → 인덱싱/최적화 시작")
    st.rerun()
    return True, len(new_or_changed)

# ---- 관리자용(완전 스캔 + 진행률/워치독/취소) ----
def admin_full_scan_and_schedule(topics_text: str, title: str, cite: bool, rename_title: bool=False):
    drive = _build_sa_drive()
    mid, manifest = _load_manifest(drive)
    known: dict = manifest.get("items", {})
    name_sig_set = {(v.get("name"), v.get("sig")) for v in known.values()}

    metas = _list_prepared_all(drive)
    metas = [m for m in metas if m.get("name")!="prepared_manifest.json"]

    # 진행판
    box = st.container(border=True)
    with box:
        st.markdown("### 🛡 관리자: 초기 전체 스캔(정확도 우선)")
        prog = st.progress(0, text="준비 중…")
        info_line = st.empty()
        delay_line = st.empty()
        cancel_col, _ = st.columns([0.25, 0.75])
        with cancel_col:
            if st.button("⛔ 강제 중단", use_container_width=True):
                ss["_admin_scan_cancel"] = True

    total = len(metas); done = 0
    started = time.time(); last_tick = time.time()
    new_or_changed = 0

    for m in metas:
        if ss.get("_admin_scan_cancel"):
            info_line.warning("사용자에 의해 중단됨.")
            break

        fid, name = m["id"], m.get("name","untitled")
        try:
            sig = _file_sig(drive, m, fast=False)  # 항상 확정 해시
        except Exception as e:
            done += 1
            pct = int(done/max(total,1)*100)
            prog.progress(pct, text=f"해시 실패 스킵: {name} → {e}")
            info_line.error(f"해시 실패: {name} → {e}")
            last_tick = time.time()
            time.sleep(YIELD_SLEEP)
            continue

        prev = known.get(fid)
        if not (prev and prev.get("sig")==sig):
            if (name, sig) not in name_sig_set:
                new_or_changed += 1
                known[fid] = {"name": name, "sig": sig, "processed_at": None}
                name_sig_set.add((name, sig))
            else:
                known[fid] = {"name": name, "sig": sig, "processed_at": prev.get("processed_at") if prev else _ts()}

        done += 1
        pct = int(done/max(total,1)*100)
        elapsed = int(time.time()-started)
        mm, ss_ = divmod(elapsed, 60)
        prog.progress(pct, text=f"{pct}% | {done}/{total} | {mm:02d}:{ss_:02d} | {name}")
        info_line.info(f"진행: {done}/{total} (신규/변경 {new_or_changed})")
        # 워치독(지연 감지)
        if time.time() - last_tick > STALL_WARN_SEC:
            delay_line.warning("⏱ 지연(네트워크/쿼터) 발생 가능 — 계속 대기 중…")
        else:
            delay_line.empty()
        last_tick = time.time()
        time.sleep(YIELD_SLEEP)

    # 결과 저장
    manifest["items"] = known; manifest["pending_full"] = []; manifest["updated_at"] = _ts()
    _save_manifest(drive, manifest, mid)

    if ss.get("_admin_scan_cancel"):
        ss["_admin_scan_cancel"] = False
        st.stop()

    if new_or_changed == 0:
        with box: st.success("완료 — 신규/변경 없음 (모두 최신).")
        return False, 0

    # 파이프라인 예약
    ss["_auto_booklets_pending"] = True
    ss["_auto_topics_text"] = topics_text
    ss["_auto_booklet_title"] = title
    ss["_auto_make_citations"] = cite
    ss.prep_both_done = False; ss.prep_both_running = True; ss.index_job = None
    with box: st.success(f"완료 — 신규/변경 {new_or_changed}개 → 인덱싱/최적화 시작")
    st.rerun()
    return True, new_or_changed

# ================= 10) 업로드(원본→prepared) ============
with st.expander("📤 관리자: 자료 업로드 (원본→prepared 저장)", expanded=False):
    st.caption("원본은 prepared/에 저장됩니다. (옵션 ON) 업로드 후 자동으로 문법 소책자(최적화본)를 Drive에 저장합니다.")
    auto_title = st.toggle("AI 제목 자동 생성(업로드 후 이름 바꾸기)", value=True)
    title_hint = st.text_input("제목 힌트(선택)", placeholder="예: 고1 영어 문법 / 학원 교재 / 중간고사 대비 등")
    auto_optimize = st.toggle("업로드 후 자동 최적화(문법 소책자 Drive 저장)", value=True)

    default_topics = [
        "Parts of Speech(품사)","Articles(관사)","Nouns & Pronouns(명사/대명사)",
        "Verbs & Tenses(시제)","Modals(조동사)","Passive(수동태)","Gerunds & Infinitives(동명사/부정사)",
        "Adjectives & Adverbs(형용사/부사/비교급)","Prepositions(전치사)","Phrasal Verbs(구동사)",
        "Conjunctions & Clauses(접속사/절)","Conditionals(조건문)","Relative Clauses(관계사절)",
        "Reported Speech(화법전환)","Questions & Negation(의문문/부정문)","Sentence Structure(문장구조·어순)"
    ]
    topics_text = st.text_area("토픽 목록(줄바꿈, 자동 최적화용)", value="\n".join(default_topics), height=150, disabled=not auto_optimize)
    booklet_title = st.text_input("소책자 세트 제목", value="Grammar Booklets", disabled=not auto_optimize)
    make_citations = st.checkbox("소책자에 참고자료(출처) 섹션 포함", value=True, disabled=not auto_optimize)

    SUPPORTED_TYPES = ["pdf","docx","doc","pptx","ppt","md","txt","rtf","odt","html","epub","xlsx","xls","csv"]
    files = st.file_uploader("로컬 파일 선택 (여러 개 가능)", type=SUPPORTED_TYPES, accept_multiple_files=True)
    gdocs_urls = st.text_area("Google Docs/Slides/Sheets URL (줄바꿈으로 여러 개)", height=80)

    prog = st.progress(0, text="대기 중…"); status_area = st.empty(); result_area = st.empty()

    if st.button("업로드/가져오기 → prepared", type="primary", use_container_width=True):
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseUpload
        from googleapiclient.errors import HttpError
        from src.rag_engine import _normalize_sa
        from src.google_oauth import is_signed_in, build_drive_service

        creds_sa = _normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON)
        drive_sa = build("drive","v3",credentials=creds_sa)
        drive_oauth = build_drive_service() if is_signed_in() else None

        created, total = [], 1
        if files: total += len(files)
        url_list = [u.strip() for u in (gdocs_urls.splitlines() if gdocs_urls else []) if u.strip()]
        if url_list: total += len(url_list)

        progress_state = {"done": 0, "total": total}
        def _tick(msg: str):
            progress_state["done"] += 1; pct = int(progress_state["done"]/max(progress_state["total"],1)*100)
            prog.progress(pct, text=msg); status_area.info(msg)

        try:
            # a) 로컬 파일
            if files:
                for f in files:
                    data = f.read(); buf = io.BytesIO(data)
                    base = _safe_name(f.name)
                    ext = (base.rsplit(".",1)[-1].lower() if "." in base else "")
                    name = f"{_ts()}__{base}"
                    mime = _guess_mime_by_ext(base)
                    media = MediaIoBaseUpload(buf, mimetype=mime, resumable=False)
                    meta = {"name": name, "parents":[settings.GDRIVE_FOLDER_ID]}
                    _tick(f"업로드 중: {name}")
                    res = drive_sa.files().create(body=meta, media_body=media, fields="id,webViewLink").execute()
                    created.append({"id":res["id"],"name":name,"link":res.get("webViewLink",""),"ext":ext,"orig_base":base})

            # b) Google 문서 링크
            def _parse_gdoc_id(s: str) -> str|None:
                for pat in [r"/d/([-\w]{15,})", r"[?&]id=([-\w]{15,})$", r"^([-\w]{15,})$"]:
                    m = re.search(pat, s.strip()); 
                    if m: return m.group(1)
                return None
            for raw in url_list:
                file_id = _parse_gdoc_id(raw)
                if not file_id: _tick("잘못된 링크 건너뜀"); continue
                drive_ro = drive_oauth or drive_sa
                try:
                    meta = drive_ro.files().get(fileId=file_id, fields="id,name,mimeType").execute()
                    name0, mtype = meta.get("name","untitled"), meta.get("mimeType","")
                except HttpError:
                    if drive_ro is drive_oauth:
                        try:
                            meta = drive_sa.files().get(fileId=file_id, fields="id,name,mimeType").execute()
                            name0, mtype = meta.get("name","untitled"), meta.get("mimeType",""); drive_ro = drive_sa
                        except Exception as e2:
                            status_area.error(f"접근 실패: {e2}"); continue
                    else:
                        status_area.error("접근 실패(공유 필요)"); continue
                GOOGLE_NATIVE = {
                    "application/vnd.google-apps.document": ("application/pdf",".pdf"),
                    "application/vnd.google-apps.presentation": ("application/pdf",".pdf"),
                    "application/vnd.google-apps.spreadsheet": ("application/pdf",".pdf"),
                }
                if mtype in GOOGLE_NATIVE:
                    export_mime, ext = GOOGLE_NATIVE[mtype]
                    _tick(f"내보내는 중: {name0}{ext}")
                    data = drive_ro.files().export(fileId=file_id, mimeType=export_mime).execute()
                    buf = io.BytesIO(data)
                    name = f"{_ts()}__{_safe_name(name0)}{ext}"
                    media = MediaIoBaseUpload(buf, mimetype=export_mime, resumable=False)
                    meta2 = {"name": name, "parents":[settings.GDRIVE_FOLDER_ID]}
                    res2 = drive_sa.files().create(body=meta2, media_body=media, fields="id,webViewLink").execute()
                    created.append({"id":res2["id"],"name":name,"link":res2.get("webViewLink",""),
                                    "ext":ext.strip("."),"orig_base":name0})
                else:
                    _tick(f"복사 중: {name0}")
                    body = {"name": f"{_ts()}__{_safe_name(name0)}", "parents":[settings.GDRIVE_FOLDER_ID]}
                    try:
                        res3 = drive_sa.files().copy(fileId=file_id, body=body, fields="id,webViewLink").execute()
                    except HttpError:
                        if drive_oauth:
                            res3 = drive_oauth.files().copy(fileId=file_id, body=body, fields="id,webViewLink").execute()
                        else:
                            status_area.error("복사 실패(권한 부족)"); continue
                    created.append({"id":res3["id"],"name":body["name"],"link":res3.get("webViewLink",""),
                                    "ext":"", "orig_base": name0})

            # c) (선택) AI 제목 변경
            renamed_rows = []
            if auto_title and created:
                used=set()
                for item in created:
                    fid, old = item["id"], item["name"]
                    ext = f".{item['ext']}" if item.get("ext") else ""
                    ai_title = _ai_title(item.get("orig_base", old), hint=title_hint)
                    new_name = _safe_name(f"{_ts()}__{ai_title}{ext}")
                    k=new_name; n=2
                    while k in used: k=f"{new_name} ({n})"; n+=1
                    try:
                        drive_sa.files().update(fileId=fid, body={"name":k}).execute()
                        renamed_rows.append({"original":old,"renamed_to":k,"open":item["link"]}); used.add(k)
                    except Exception as e:
                        renamed_rows.append({"original":old,"renamed_to":f"(실패) {e}","open":item["link"]})
            else:
                for item in created:
                    renamed_rows.append({"original":item["name"],"renamed_to":"(AI 제목 꺼짐)","open":item["link"]})

            # d) 결과표
            prog.progress(100, text="완료"); status_area.success(f"총 {len(created)}개 항목 처리 완료 (prepared)")
            if renamed_rows:
                df = pd.DataFrame(renamed_rows)
                result_area.dataframe(
                    df, use_container_width=True, hide_index=True,
                    column_config={"original": st.column_config.TextColumn("원래 파일명"),
                                   "renamed_to": st.column_config.TextColumn("변경 후 파일명"),
                                   "open": st.column_config.LinkColumn("열기", display_text="열기")}
                )
            st.toast("업로드/가져오기 완료", icon="✅")

            # e) 자동 최적화 예약 + 인덱싱 자동 시작
            if auto_optimize:
                ss["_auto_booklets_pending"] = True
                ss["_auto_topics_text"] = topics_text
                ss["_auto_booklet_title"] = booklet_title
                ss["_auto_make_citations"] = make_citations
            ss.prep_both_done = False; ss.prep_both_running = True; ss.index_job = None
            st.info("인덱싱/최적화 파이프라인을 시작합니다."); st.rerun()

        except Exception as e:
            prog.progress(0, text="오류"); status_area.error(f"처리 실패: {e}")

# ================= 11) prepared 감시 UI ================
with st.expander("📡 prepared 폴더 감시(Drive 직접 투입 → 최적화/스킵)", expanded=False):
    st.caption("prepared/ 에 파일을 직접 넣어도 자동 감지합니다. (관리자는 첫 방문 시 완전 스캔)")
    ss["_watch_on_load"] = st.toggle("페이지 열릴 때 자동 스캔", value=ss["_watch_on_load"])
    ss["_watch_rename_title"] = st.toggle("감시 시에도 AI 제목 자동 정리(선택)", value=ss["_watch_rename_title"])

    topics_watch = st.text_area("토픽 목록(감시용, 줄바꿈)", value=ss.get("_auto_topics_text") or
                                "Verbs & Tenses(시제)\nPassive(수동태)\nConditionals(조건문)", height=100)
    title_watch = st.text_input("소책자 세트 제목(감시용)", value=ss.get("_auto_booklet_title","Grammar Booklets"))
    cite_watch = st.checkbox("참고자료 섹션 포함", value=ss.get("_auto_make_citations", True))

    if st.button("📡 지금 스캔 & 최적화", use_container_width=True):
        if IS_ADMIN:
            admin_full_scan_and_schedule(topics_watch, title_watch, cite_watch, ss["_watch_rename_title"])
        else:
            watch_scan_and_schedule(topics_watch, title_watch, cite_watch, ss["_watch_rename_title"])

# ================= 12) 인덱싱 보고서 ===================
rep = ss.get("indexing_report")
if rep:
    with st.expander("🧾 인덱싱 보고서 (스킵된 파일 보기)", expanded=False):
        st.write(f"총 파일(매니페스트): {rep.get('total_manifest')}, "
                 f"로딩된 문서 수: {rep.get('loaded_docs')}, "
                 f"스킵: {rep.get('skipped_count')}")
        skipped = rep.get("skipped", [])
        if skipped: st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)
        else: st.caption("스킵된 파일이 없습니다 🎉")

# ================= 13) 두뇌 준비(인덱싱) ===============
st.markdown("---"); st.subheader("🧠 두뇌 준비 — 저장본 로드 ↔ 변경 시 증분 인덱싱 (중간 취소/재개)")

c_g, c_o = st.columns(2)
with c_g: st.caption("Gemini 진행"); g_bar = st.empty(); g_msg = st.empty()
with c_o: st.caption("ChatGPT 진행"); o_bar = st.empty(); o_msg = st.empty()

def _render_progress(slot_bar, slot_msg, pct: int, msg: str | None = None):
    p = max(0, min(100, int(pct)))
    slot_bar.markdown(
        f"<div class='gp-wrap'><div class='gp-fill' style='width:{p}%'></div><div class='gp-label'>{p}%</div></div>",
        unsafe_allow_html=True)
    if msg is not None: slot_msg.markdown(f"<div class='gp-msg'>{msg}</div>", unsafe_allow_html=True)

def _is_cancelled() -> bool: return bool(ss.get("prep_cancel_requested", False))

from src.rag_engine import (
    set_embed_provider, make_llm, get_text_answer, CancelledError,
    start_index_builder, resume_index_builder, cancel_index_builder, llm_complete
)

def run_prepare_both_step():
    # 1) 임베딩 설정
    embed_provider = "openai"
    embed_api = (getattr(settings, "OPENAI_API_KEY", None).get_secret_value()
                 if getattr(settings, "OPENAI_API_KEY", None) else "")
    embed_model = getattr(settings, "OPENAI_EMBED_MODEL", "text-embedding-3-small")
    if not embed_api:
        embed_provider = "google"; embed_api = settings.GEMINI_API_KEY.get_secret_value()
        embed_model = getattr(settings, "EMBED_MODEL", "text-embedding-004")
    try:
        _render_progress(g_bar, g_msg, 3, f"임베딩 설정({embed_provider})")
        _render_progress(o_bar, o_msg, 3, f"임베딩 설정({embed_provider})")
        set_embed_provider(embed_provider, embed_api, embed_model)
    except Exception as e:
        _render_progress(g_bar, g_msg, 100, f"임베딩 실패: {e}")
        _render_progress(o_bar, o_msg, 100, f"임베딩 실패: {e}")
        ss.prep_both_running = False; return

    # 2) 인덱스 스텝
    def upd(p, m=None): _render_progress(g_bar, g_msg, p, m); _render_progress(o_bar, o_msg, p, m)
    def umsg(m): _render_progress(g_bar, g_msg, ss.get("p_shared", 0), m); _render_progress(o_bar, o_msg, ss.get("p_shared", 0), m)
    persist_dir = f"{getattr(settings,'PERSIST_DIR','/tmp/my_ai_teacher/storage_gdrive')}_shared"
    job = ss.get("index_job")
    try:
        if job is None:
            res = start_index_builder(update_pct=upd, update_msg=umsg,
                                      gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
                                      raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
                                      persist_dir=persist_dir,
                                      manifest_path=getattr(settings, "MANIFEST_PATH", "/tmp/my_ai_teacher/drive_manifest.json"),
                                      max_docs=None, is_cancelled=_is_cancelled)
        else:
            res = resume_index_builder(job=job, update_pct=upd, update_msg=umsg, is_cancelled=_is_cancelled, batch_size=6)
        status = res.get("status")
        if status == "running":
            ss.index_job = res["job"]
            _render_progress(g_bar, g_msg, res.get("pct", 8), res.get("msg","진행 중…"))
            _render_progress(o_bar, o_msg, res.get("pct", 8), res.get("msg","진행 중…"))
            time.sleep(0.15); st.rerun(); return
        if status == "cancelled":
            ss.prep_both_running = False; ss.prep_cancel_requested = False; ss.index_job = None
            _render_progress(g_bar, g_msg, res.get("pct", 0), "사용자 취소")
            _render_progress(o_bar, o_msg, res.get("pct", 0), "사용자 취소"); return
        if status != "done":
            _render_progress(g_bar, g_msg, 100, "인덱싱 실패")
            _render_progress(o_bar, o_msg, 100, "인덱싱 실패")
            ss.prep_both_running = False; return
        index = res["index"]; ss.index_job = None
    except Exception as e:
        ss.prep_both_running = False; ss.index_job = None
        _render_progress(g_bar, g_msg, 100, f"에러: {e}")
        _render_progress(o_bar, o_msg, 100, f"에러: {e}"); return

    # 3) QE 생성
    try:
        g_llm = make_llm("google", settings.GEMINI_API_KEY.get_secret_value(),
                         getattr(settings, "LLM_MODEL", "gemini-1.5-pro"),
                         float(ss.get("temperature", 0.0)))
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
            o_llm = make_llm("openai", settings.OPENAI_API_KEY.get_secret_value(),
                             getattr(settings, "OPENAI_LLM_MODEL", "gpt-4o-mini"),
                             float(ss.get("temperature", 0.0)))
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

    # 4) 자동 소책자(예약 시)
    if ss.get("_auto_booklets_pending"):
        try:
            topics = [t.strip() for t in ss.get("_auto_topics_text","").splitlines() if t.strip()]
            topics = topics or ["Verbs & Tenses(시제)","Passive(수동태)","Conditionals(조건문)"]
            title = ss.get("_auto_booklet_title","Grammar Booklets")
            citations = bool(ss.get("_auto_make_citations", True))
            with st.spinner("자동 최적화(문법 소책자) 생성 중…"):
                res_auto = generate_booklets_drive(topics, title, citations)
            st.success(f"자동 최적화 완료 → prepared_volumes/{res_auto['folder_name']}")
            if res_auto.get("rows"):
                st.dataframe(pd.DataFrame(res_auto["rows"]),
                             use_container_width=True, hide_index=True,
                             column_config={"topic": st.column_config.TextColumn("토픽"),
                                            "open": st.column_config.LinkColumn("열기", display_text="열기")})
        except Exception as e:
            st.warning(f"자동 최적화 실패: {e}")
        finally:
            ss["_auto_booklets_pending"] = False

    ss.prep_both_running = False; ss.prep_both_done = True
    time.sleep(0.2); st.rerun()

# 실행/취소 버튼
left, right = st.columns([0.7, 0.3])
with left:
    clicked = st.button("🚀 한 번에 준비하기", key="prepare_both", use_container_width=True,
                        disabled=ss.prep_both_running or ss.prep_both_done)
with right:
    cancel_clicked = st.button("⛔ 준비 취소", key="cancel_prepare", use_container_width=True, type="secondary",
                               disabled=not ss.prep_both_running)
from src.rag_engine import cancel_index_builder
if cancel_clicked and ss.prep_both_running:
    ss.prep_cancel_requested = True
    if ss.get("index_job"): cancel_index_builder(ss.index_job)
    st.rerun()
if clicked and not (ss.prep_both_running or ss.prep_both_done):
    ss.prep_cancel_requested = False; ss.prep_both_running = True; ss.index_job = None; st.rerun()

# 자동 스캔: 페이지 최초 1회
if ss["_watch_on_load"] and not ss["_watch_ran_once"] and not ss.prep_both_running:
    ss["_watch_ran_once"] = True
    topics0 = ss.get("_auto_topics_text") or "Verbs & Tenses(시제)\nPassive(수동태)\nConditionals(조건문)"
    title0  = ss.get("_auto_booklet_title","Grammar Booklets")
    cite0   = bool(ss.get("_auto_make_citations", True))
    try:
        if IS_ADMIN and not ss.get("_admin_initial_fullscan_done", False):
            ss["_admin_initial_fullscan_done"] = True
            admin_full_scan_and_schedule(topics0, title0, cite0, ss["_watch_rename_title"])
        else:
            watch_scan_and_schedule(topics0, title0, cite0, ss["_watch_rename_title"])
    except Exception as e:
        st.warning(f"초기 자동 스캔 실패: {e}")

if ss.prep_both_running:
    run_prepare_both_step()

st.caption("준비 버튼을 다시 활성화하려면 아래 재설정 버튼을 누르세요.")
if st.button("🔧 재설정(버튼 다시 활성화)", disabled=not ss.prep_both_done):
    ss.prep_both_done = False; ss["_watch_ran_once"] = False; ss["_admin_initial_fullscan_done"] = False; st.rerun()

# ================= 14) 그룹토론 UI =====================
st.markdown("---")
st.subheader("💬 그룹토론 — 학생 ↔ 🤖Gemini(친절/꼼꼼) ↔ 🤖ChatGPT(유머러스/보완)")

ready_google = "qe_google" in ss
ready_openai = "qe_openai" in ss
if ss.session_terminated: st.warning("세션이 종료된 상태입니다. 새로고침으로 다시 시작하세요."); st.stop()
if not ready_google: st.info("먼저 **[🚀 한 번에 준비하기]**로 두뇌를 준비하세요. (OpenAI 키 없으면 Gemini만 응답)"); st.stop()

# 과거 메시지 렌더
for m in ss.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

def _strip_sources(text: str) -> str: return re.sub(r"\n+---\n\*참고 자료:.*$", "", text, flags=re.DOTALL)
def _build_context(messages, limit_pairs=2, max_chars=2000) -> str:
    pairs, buf_user = [], None
    for m in reversed(messages):
        role, content = m.get("role"), str(m.get("content","")).strip()
        if role == "assistant":
            content = re.sub(r"^\*\*🤖 .*?\*\*\s*\n+", "", content).strip()
            if buf_user is not None:
                pairs.append((buf_user, content)); buf_user = None
                if len(pairs) >= limit_pairs: break
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
CHATGPT_STYLE = ("당신은 유머러스하지만 정확한 동료 교사입니다. 동료(Gemini)의 답을 읽고 "
                 "빠진 부분을 보완/교정하고 마지막에 <최종 정리>로 요약하세요. 과한 농담 금지.")
mode = st.radio("학습 모드", ["💬 이유문법 설명", "🔎 구문 분석", "📚 독해 및 요약"], horizontal=True, key="mode_select")

from src import chat_store
from src.drive_log import get_chatlog_folder_id, save_chatlog_markdown_oauth
def _jsonl_log(items):
    if not ss.save_logs: return
    try:
        parent_id = (getattr(settings, "CHATLOG_FOLDER_ID", None) or settings.GDRIVE_FOLDER_ID)
        sa = settings.GDRIVE_SERVICE_ACCOUNT_JSON
        if isinstance(sa, str):
            try: sa = json.loads(sa)
            except Exception: pass
        sub_id = get_chatlog_folder_id(parent_folder_id=parent_id, sa_json=sa)
        chat_store.append_jsonl(folder_id=sub_id, sa_json=sa, items=items)
        st.toast("대화 JSONL 저장 완료", icon="💾")
    except Exception as e:
        st.warning(f"대화 JSONL 저장 실패: {e}")

user_input = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
if user_input:
    ss.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"): st.markdown(user_input)
    _jsonl_log([chat_store.make_entry(ss.session_id, "user", "user", user_input, mode, model="user")])

    from src.rag_engine import get_text_answer, llm_complete
    with st.spinner("🤖 Gemini 선생님이 먼저 답합니다…"):
        prev_ctx = _build_context(ss.messages[:-1], limit_pairs=2, max_chars=2000)
        gemini_query = (f"[이전 대화]\n{prev_ctx}\n\n" if prev_ctx else "") + f"[학생 질문]\n{user_input}"
        ans_g = get_text_answer(ss["qe_google"], gemini_query, _persona() + "\n" + GEMINI_STYLE)

    content_g = f"**🤖 Gemini**\n\n{ans_g}"
    ss.messages.append({"role":"assistant","content":content_g})
    with st.chat_message("assistant"): st.markdown(content_g)
    _jsonl_log([chat_store.make_entry(ss.session_id, "assistant", "Gemini", content_g, mode, model=getattr(settings,"LLM_MODEL","gemini"))])

    if "qe_openai" in ss:
        review_directive = ("역할: 동료 AI 영어교사\n"
            "목표: [이전 대화], [학생 질문], [동료의 1차 답변]을 읽고 사실오류/빠진점/모호함을 보완.\n"
            "지침: 1)핵심 간결 재정리 2)틀린 부분 근거와 교정 3)예문 2~3개 4)<최종 정리>로 요약. 외부검색 금지.")
        prev_all = _build_context(ss.messages, limit_pairs=2, max_chars=2000)
        augmented = ((f"[이전 대화]\n{prev_all}\n\n" if prev_all else "") +
                     f"[학생 질문]\n{user_input}\n\n"
                     f"[동료의 1차 답변(Gemini)]\n{_strip_sources(ans_g)}\n\n"
                     "[당신의 작업]\n위 기준으로만 보완/검증.")
        with st.spinner("🤝 ChatGPT 선생님이 보완/검증 중…"):
            ans_o = llm_complete(ss.get("llm_openai"),
                                 _persona() + "\n" + CHATGPT_STYLE + "\n\n" + review_directive + "\n\n" + augmented)
        content_o = f"**🤖 ChatGPT**\n\n{ans_o}"
        ss.messages.append({"role":"assistant","content":content_o})
        with st.chat_message("assistant"): st.markdown(content_o)
    else:
        with st.chat_message("assistant"):
            st.info("ChatGPT 키가 없어 Gemini만 응답했습니다. OPENAI_API_KEY를 추가하면 보완/검증이 활성화됩니다.")

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
