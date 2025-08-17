# src/rag_engine.py
from __future__ import annotations
import os, json, shutil, io, zipfile, hashlib, re
from typing import Callable, Any, Mapping, List, Tuple, Dict
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None  # 폴백

import streamlit as st
from src.config import settings, APP_DATA_DIR, QUALITY_REPORT_PATH

# ====== 백업 파일명 규칙(날짜 포함) ===========================================
INDEX_BACKUP_PREFIX = "ai_brain_cache"  # 결과: ai_brain_cache-YYYYMMDD-HHMMSS.zip

def _now_kst_str() -> str:
    try:
        if ZoneInfo:
            return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S")
    except Exception:
        pass
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

def _build_backup_filename(prefix: str = INDEX_BACKUP_PREFIX) -> str:
    return f"{prefix}-{_now_kst_str()}.zip"

# ====== 체크포인트 경로 ========================================================
CHECKPOINT_PATH = str((APP_DATA_DIR / "index_checkpoint.json").resolve())

# ====== LLM/Embedding 설정 ====================================================
def init_llama_settings(api_key: str, llm_model: str, embed_model: str, temperature: float = 0.0):
    from llama_index.core import Settings
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    from llama_index.llms.google_genai import GoogleGenAI
    Settings.llm = GoogleGenAI(model=llm_model, api_key=api_key, temperature=temperature)
    Settings.embed_model = GoogleGenAIEmbedding(model_name=embed_model, api_key=api_key)
    try:
        _ = Settings.embed_model.get_text_embedding("ping")
    except Exception as e:
        st.error("임베딩 모델 점검 실패 — API 키/모델명/네트워크를 확인하세요.")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

# ====== 로컬 저장본 로딩 =======================================================
@st.cache_resource(show_spinner=False)
def _load_index_from_disk(persist_dir: str):
    from llama_index.core import StorageContext, load_index_from_storage
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index

# ====== Google Drive helpers ==================================================
def _build_drive_service(creds_dict: Mapping[str, Any], write: bool = False):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    if write:
        scopes = ["https://www.googleapis.com/auth/drive"]
    creds = service_account.Credentials.from_service_account_info(dict(creds_dict), scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def _fetch_drive_manifest(creds_dict: Mapping[str, Any], folder_id: str) -> dict:
    svc = _build_drive_service(creds_dict, write=False)
    files = []
    page_token = None
    q = f"'{folder_id}' in parents and trashed=false"
    fields = "nextPageToken, files(id,name,mimeType,modifiedTime,md5Checksum,size)"
    while True:
        resp = svc.files().list(
            q=q, fields=fields, pageToken=page_token,
            pageSize=1000, supportsAllDrives=True, includeItemsFromAllDrives=True
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return {
        f["id"]: {
            "name": f.get("name"),
            "mimeType": f.get("mimeType"),
            "modifiedTime": f.get("modifiedTime"),
            "md5": f.get("md5Checksum"),
            "size": f.get("size"),
        } for f in files
    }

# ====== 서비스계정 정규화/검증 =================================================
def _normalize_sa(raw_sa: Any | None) -> Mapping[str, Any] | None:
    if raw_sa is None:
        return None
    if isinstance(raw_sa, Mapping):
        d = dict(raw_sa)
    elif isinstance(raw_sa, str) and raw_sa.strip():
        try:
            d = json.loads(raw_sa)
        except Exception:
            return None
    else:
        return None
    for k in ("service_account", "serviceAccount"):
        if isinstance(d.get(k), Mapping):
            d = dict(d[k])
    pk = d.get("private_key")
    if isinstance(pk, str) and "\\n" in pk and "\n" not in pk:
        d["private_key"] = pk.replace("\\n", "\n")
    return d

def _validate_sa(creds: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not creds:
        st.error("GDRIVE 서비스계정 자격증명이 비었습니다. secrets에 JSON을 넣어주세요.")
        st.stop()
    required = {"type","project_id","private_key_id","private_key","client_email","client_id","token_uri"}
    missing = [k for k in required if k not in creds]
    if missing:
        st.error("서비스계정 JSON에 필수 키가 누락되었습니다: " + ", ".join(missing))
        with st.expander("진단 정보(보유 키 목록)"):
            st.write(sorted(list(creds.keys())))
        st.stop()
    return creds

# ====== ZIP 백업/복원 + 보관정책 =============================================
def _zip_dir(src_dir: str, zip_path: str) -> None:
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for name in files:
                full = os.path.join(root, name)
                rel = os.path.relpath(full, src_dir)
                zf.write(full, rel)

def _list_backups(svc, folder_id: str, prefix: str = INDEX_BACKUP_PREFIX) -> List[dict]:
    q = (
        f"'{folder_id}' in parents and trashed=false and "
        f"mimeType='application/zip' and name contains '{prefix}-'"
    )
    resp = svc.files().list(
        q=q, orderBy="modifiedTime desc", pageSize=100,
        fields="files(id,name,modifiedTime,size,md5Checksum)"
    ).execute()
    return resp.get("files", [])

def prune_old_backups(creds: Mapping[str, Any], folder_id: str, keep: int = 5,
                      prefix: str = INDEX_BACKUP_PREFIX) -> List[Tuple[str, str]]:
    if keep <= 0:
        keep = 1
    svc = _build_drive_service(creds, write=True)
    items = _list_backups(svc, folder_id, prefix)
    to_delete = items[keep:] if len(items) > keep else []
    deleted: List[Tuple[str, str]] = []
    for it in to_delete:
        try:
            svc.files().delete(fileId=it["id"]).execute()
            deleted.append((it["id"], it.get("name", "")))
        except Exception:
            pass
    return deleted

def export_brain_to_drive(creds: Mapping[str, Any], persist_dir: str, dest_folder_id: str,
                          filename: str | None = None) -> Tuple[str, str]:
    if not os.path.exists(persist_dir):
        raise FileNotFoundError("persist_dir가 없습니다. 먼저 두뇌를 생성하세요.")
    fname = filename or _build_backup_filename()
    tmp_zip = os.path.join("/tmp", fname)
    _zip_dir(persist_dir, tmp_zip)

    svc = _build_drive_service(creds, write=True)
    from googleapiclient.http import MediaFileUpload
    meta = {"name": fname, "parents": [dest_folder_id]}
    media = MediaFileUpload(tmp_zip, mimetype="application/zip", resumable=True)
    created = svc.files().create(body=meta, media_body=media, fields="id,name,parents").execute()
    return created["id"], created["name"]

def import_brain_from_drive(creds: Mapping[str, Any], persist_dir: str, src_folder_id: str,
                            prefix: str = INDEX_BACKUP_PREFIX) -> bool:
    svc = _build_drive_service(creds, write=True)
    items = _list_backups(svc, src_folder_id, prefix)
    if not items:
        return False
    file_id = items[0]["id"]
    from googleapiclient.http import MediaIoBaseDownload
    req = svc.files().get_media(fileId=file_id)
    tmp_zip = os.path.join("/tmp", items[0]["name"])
    with io.FileIO(tmp_zip, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            status, done = downloader.next_chunk()

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        zf.extractall(persist_dir)
    return True

def try_restore_index_from_drive(creds: Mapping[str, Any], persist_dir: str, folder_id: str) -> bool:
    if os.path.exists(persist_dir):
        return True
    try:
        return import_brain_from_drive(creds, persist_dir, folder_id, INDEX_BACKUP_PREFIX)
    except Exception:
        return False

# ====== 체크포인트 유틸 =======================================================
def _load_checkpoint(path: str = CHECKPOINT_PATH) -> Dict[str, Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_checkpoint(data: Dict[str, Dict], path: str = CHECKPOINT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)

def _mark_done(cp: Dict[str, Dict], file_id: str, entry: Dict) -> None:
    cp[file_id] = {
        "md5": entry.get("md5"),
        "modifiedTime": entry.get("modifiedTime"),
        "name": entry.get("name"),
        "ts": _now_kst_str(),
    }
    _save_checkpoint(cp)

def clear_checkpoint(path: str = CHECKPOINT_PATH) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

# ====== 저장본 존재 체크 & 안전한 StorageContext 생성 =========================
def _has_persisted_index(persist_dir: str) -> bool:
    names = ("docstore.json", "index_store.json", "vector_store.json")
    return any(os.path.exists(os.path.join(persist_dir, n)) for n in names)

def _make_storage_context(persist_dir: str):
    from llama_index.core import StorageContext
    try:
        if _has_persisted_index(persist_dir):
            return StorageContext.from_defaults(persist_dir=persist_dir)
    except Exception:
        pass
    return StorageContext.from_defaults()

# ====== 품질 유틸(전처리·중복·리포트) =========================================
_ws_re = re.compile(r"[ \t\f\v]+")
def _clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = _ws_re.sub(" ", s)
    return s.strip()

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _get_opt() -> Dict[str, Any]:
    return {
        "chunk_size": int(st.session_state.get("opt_chunk_size", settings.CHUNK_SIZE)),
        "chunk_overlap": int(st.session_state.get("opt_chunk_overlap", settings.CHUNK_OVERLAP)),
        "min_chars": int(st.session_state.get("opt_min_chars", settings.MIN_CHARS_PER_DOC)),
        "dedup": bool(st.session_state.get("opt_dedup", settings.DEDUP_BY_TEXT_HASH)),
        "skip_low_text": bool(st.session_state.get("opt_skip_low_text", settings.SKIP_LOW_TEXT_DOCS)),
        "pre_summarize": bool(st.session_state.get("opt_pre_summarize", settings.PRE_SUMMARIZE_DOCS)),
    }

def _maybe_summarize_docs(docs: List[Any]) -> None:
    if not docs or not _get_opt()["pre_summarize"]:
        return
    try:
        from llama_index.core import Settings
        for d in docs:
            if "doc_summary" in getattr(d, "metadata", {}):
                continue
            text = (getattr(d, "text", "") or "")[:4000]
            if not text:
                continue
            prompt = (
                "다음 문서 내용을 교사 시각에서 5줄 이내 핵심 bullet로 요약하라.\n"
                "교재 단원/개념/예문/핵심 규칙을 간단히 표시하라.\n\n"
                f"[문서 내용]\n{text}"
            )
            try:
                resp = Settings.llm.complete(prompt)
                summary = getattr(resp, "text", None) or str(resp)
                d.metadata["doc_summary"] = summary.strip()
            except Exception:
                pass
    except Exception:
        pass

def _preprocess_docs(docs: List[Any], seen_hashes: set, min_chars: int, dedup: bool) -> Tuple[List[Any], Dict[str, Any]]:
    kept: List[Any] = []
    stats = {"input_docs": len(docs), "kept": 0, "skipped_low_text": 0, "skipped_dup": 0, "total_chars": 0}
    for d in docs:
        t = _clean_text(getattr(d, "text", "") or "")
        if len(t) < min_chars:
            stats["skipped_low_text"] += 1
            continue
        h = _sha1(t)
        if dedup and h in seen_hashes:
            stats["skipped_dup"] += 1
            continue
        d.text = t
        d.metadata = dict(getattr(d, "metadata", {}) or {})
        d.metadata["text_hash"] = h
        kept.append(d)
        seen_hashes.add(h)
        stats["kept"] += 1
        stats["total_chars"] += len(t)
    return kept, stats

def _load_quality_report(path: str = QUALITY_REPORT_PATH) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"summary": {}, "files": {}}

def _save_quality_report(data: Dict[str, Any], path: str = QUALITY_REPORT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)

# ====== 인덱스 생성(체크포인트 + 최적화 + 중지 지원) ==========================
def _build_index_with_checkpoint(update_pct: Callable[[int, str | None], None],
                                 update_msg: Callable[[str], None],
                                 gdrive_folder_id: str,
                                 gcp_creds: Mapping[str, Any],
                                 persist_dir: str,
                                 remote_manifest: Dict[str, Dict],
                                 should_stop: Callable[[], bool] | None = None):
    """
    파일ID 단위 처리 → 각 파일 완주 후 저장 & 체크포인트 기록.
    전처리(정리/저품질 필터/중복 제거) → SentenceSplitter 청킹 → 인덱스 누적.
    중지 버튼(should_stop=True) 감지 시 '현재 파일까지' 저장 후 안전 종료.
    """
    from llama_index.core import VectorStoreIndex, load_index_from_storage
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.readers.google import GoogleDriveReader

    if should_stop is None:
        should_stop = lambda: False  # 기본: 중지 없음

    opt = _get_opt()

    update_pct(15, "Drive 리더 초기화")
    loader = GoogleDriveReader(service_account_key=gcp_creds)

    cp = _load_checkpoint()
    todo_ids: List[str] = []
    for fid, meta in remote_manifest.items():
        done = cp.get(fid)
        if done and done.get("md5") and meta.get("md5") and done["md5"] == meta["md5"]:
            continue
        todo_ids.append(fid)

    total = len(remote_manifest)
    pending = len(todo_ids)
    done_cnt = total - pending
    update_pct(30, f"문서 목록 불러오는 중 • 전체 {total}개, 이번에 처리 {pending}개")

    os.makedirs(persist_dir, exist_ok=True)
    storage_context = _make_storage_context(persist_dir)
    try:
        _ = load_index_from_storage(storage_context)
    except Exception:
        pass

    # 품질 리포트
    qrep = _load_quality_report()
    qrep.setdefault("summary", {}).setdefault("total_docs", total)
    for k in ("processed_docs","kept_docs","skipped_low_text","skipped_dup","total_chars"):
        qrep["summary"].setdefault(k, 0)
    qrep.setdefault("files", {})
    seen_hashes = set(h for h in qrep.get("files", {}).values() if isinstance(h, dict) and "text_hash" in h)

    if pending == 0:
        update_pct(95, "변경 없음 → 저장본 그대로 사용")
        try:
            return load_index_from_storage(storage_context)
        except Exception:
            st.error("저장된 두뇌가 없는데 변경도 없다고 감지되었습니다. 초기화 후 다시 시도하세요.")
            st.stop()

    splitter = SentenceSplitter(chunk_size=opt["chunk_size"], chunk_overlap=opt["chunk_overlap"])

    for i, fid in enumerate(todo_ids, start=1):
        # 중지 요청이 '다음 파일로 넘어가기 전'에 들어온 경우: 바로 종료
        if should_stop():
            update_msg("🛑 중지 요청 감지 — 현재까지 저장 후 종료합니다.")
            break

        meta = remote_manifest.get(fid, {})
        fname = meta.get("name") or fid
        update_msg(f"전처리 • {fname} ({done_cnt + i}/{total})")

        # 1) 파일 로드
        try:
            docs = loader.load_data(file_ids=[fid])
        except TypeError:
            st.error("GoogleDriveReader 버전이 오래되어 file_ids 옵션을 지원하지 않습니다. requirements 업데이트가 필요합니다.")
            st.stop()

        # 2) 전처리/필터/중복
        kept, stats = _preprocess_docs(
            docs, seen_hashes,
            min_chars=opt["min_chars"], dedup=opt["dedup"]
        )
        _maybe_summarize_docs(kept)

        # 3) 품질 리포트 기록(파일 단위)
        qrep["files"][fid] = {
            "name": fname,
            "md5": meta.get("md5"),
            "modifiedTime": meta.get("modifiedTime"),
            "kept": stats["kept"],
            "skipped_low_text": stats["skipped_low_text"],
            "skipped_dup": stats["skipped_dup"],
            "total_chars": stats["total_chars"],
        }
        qs = qrep["summary"]
        qs["processed_docs"] += 1
        qs["kept_docs"] += stats["kept"]
        qs["skipped_low_text"] += stats["skipped_low_text"]
        qs["skipped_dup"] += stats["skipped_dup"]
        qs["total_chars"] += stats["total_chars"]
        _save_quality_report(qrep)

        if stats["kept"] == 0:
            # 텍스트가 없거나 중복만 → 완료 체크만 하고 다음 파일
            _mark_done(cp, fid, meta)
            pct = 30 + int((i / max(1, pending)) * 60)
            update_pct(pct, f"건너뜀 • {fname} (저품질/중복)")
            # 중지 요청이 이 시점에 들어왔으면 여기서 종료
            if should_stop():
                update_msg("🛑 중지 요청 감지 — 현재까지 저장 후 종료합니다.")
                break
            continue

        # 4) 인덱스에 누적 추가(청킹 적용)
        update_msg(f"인덱스 생성 • {fname} ({done_cnt + i}/{total})")
        try:
            VectorStoreIndex.from_documents(
                kept, storage_context=storage_context, show_progress=False,
                transformations=[splitter]
            )
            storage_context.persist(persist_dir=persist_dir)  # 부분 진행 저장
            _mark_done(cp, fid, meta)  # 파일 '완료'로 체크포인트 기록
        except Exception as e:
            st.error(f"인덱스 생성 중 오류: {fname}")
            with st.expander("자세한 오류 보기"):
                st.exception(e)
            st.stop()

        pct = 30 + int((i / max(1, pending)) * 60)
        update_pct(pct, f"완료 • {fname}")

        # 5) 파일 경계에서 중지 요청 확인 → 안전 종료
        if should_stop():
            update_msg("🛑 중지 요청 감지 — 현재 파일까지 저장 완료, 종료합니다.")
            break

    # 최종 저장 및 인덱스 반환(부분 진행이어도 안전)
    update_pct(95, "두뇌 저장 중")
    try:
        storage_context.persist(persist_dir=persist_dir)
    except Exception as e:
        st.error("인덱스 저장 중 오류가 발생했습니다.")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

    update_pct(100, "완료")
    from llama_index.core import load_index_from_storage
    return load_index_from_storage(storage_context)

# ====== 변경 감지 → 인덱스 준비(체크포인트 포함) ==============================
def get_or_build_index(update_pct: Callable[[int, str | None], None],
                       update_msg: Callable[[str], None],
                       gdrive_folder_id: str,
                       raw_sa: Any | None,
                       persist_dir: str,
                       manifest_path: str,
                       should_stop: Callable[[], bool] | None = None):
    """Drive 변경을 감지해 저장본을 쓰거나, 변경 시에만 재인덱싱(체크포인트 & 중지 지원)."""
    update_pct(5, "드라이브 변경 확인 중…")
    gcp_creds = _validate_sa(_normalize_sa(raw_sa))

    remote = _fetch_drive_manifest(gcp_creds, gdrive_folder_id)

    local = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fp:
                local = json.load(fp)
        except Exception:
            local = None

    def _manifests_differ(local_m: dict | None, remote_m: dict) -> bool:
        if local_m is None:
            return True
        if set(local_m.keys()) != set(remote_m.keys()):
            return True
        for fid, r in remote_m.items():
            l = local_m.get(fid, {})
            if l.get("md5") and r.get("md5"):
                if l["md5"] != r["md5"]:
                    return True
            if l.get("modifiedTime") != r.get("modifiedTime"):
                return True
        return False

    if os.path.exists(persist_dir) and not _manifests_differ(local, remote):
        update_pct(25, "변경 없음 → 저장된 두뇌 로딩")
        from llama_index.core import StorageContext, load_index_from_storage
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        idx = load_index_from_storage(storage_context)
        update_pct(100, "완료!")
        return idx

    update_pct(40, "변경 감지 → 전처리/청킹/인덱스 생성 (체크포인트)")
    idx = _build_index_with_checkpoint(
        update_pct, update_msg, gdrive_folder_id, gcp_creds, persist_dir, remote,
        should_stop=should_stop
    )

    # 새 매니페스트 저장 & 체크포인트 정리(완주했을 때만)
    if not (should_stop and should_stop()):
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as fp:
            json.dump(remote, fp, ensure_ascii=False, indent=2, sort_keys=True)
        # 모든 파일 완료했을 때만 체크포인트를 비움
        # (중지한 경우에는 남겨두어 재개 지점으로 사용)
        if os.path.exists(CHECKPOINT_PATH):
            try:
                with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                    cp = json.load(f)
                # 모두 완료인지 빠르게 확인
                all_done = set(cp.keys()) == set(remote.keys())
                if all_done:
                    os.remove(CHECKPOINT_PATH)
            except Exception:
                pass

    update_pct(100, "완료!")
    return idx

# ====== QA 유틸 ===============================================================
def get_text_answer(query_engine, question: str, system_prompt: str) -> str:
    try:
        full_query = (
            f"{system_prompt}\n\n"
            "[지시사항] 반드시 업로드된 강의 자료를 최우선으로 참고하여 답변하고, "
            "근거를 찾을 수 없다면 그 사실을 명확히 밝혀라.\n\n"
            f"[학생의 질문]\n{question}"
        )
        response = query_engine.query(full_query)
        answer_text = str(response)
        try:
            files = [n.metadata.get("file_name", "알 수 없음") for n in getattr(response, "source_nodes", [])]
            source_files = ", ".join(sorted(list(set(files)))) if files else "출처 정보 없음"
        except Exception:
            source_files = "출처 정보 없음"
        return f"{answer_text}\n\n---\n*참고 자료: {source_files}*"
    except Exception as e:
        return f"텍스트 답변 생성 중 오류 발생: {e}"
