# src/rag_engine.py
from __future__ import annotations
import os, json, shutil, io, zipfile
from typing import Callable, Any, Mapping, List, Tuple, Dict
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None  # 폴백

import streamlit as st
from src.config import settings, APP_DATA_DIR

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
    """LlamaIndex 저장본의 핵심 파일이 하나라도 있으면 True."""
    names = ("docstore.json", "index_store.json", "vector_store.json")
    return any(os.path.exists(os.path.join(persist_dir, n)) for n in names)

def _make_storage_context(persist_dir: str):
    """저장본이 있으면 로드, 없으면 빈 컨텍스트로 시작(초기 인덱싱에서 FileNotFound 방지)."""
    from llama_index.core import StorageContext
    try:
        if _has_persisted_index(persist_dir):
            return StorageContext.from_defaults(persist_dir=persist_dir)
    except Exception:
        # 파손/부분 저장본일 수 있음 → 깨끗한 컨텍스트로 시작
        pass
    return StorageContext.from_defaults()

# ====== 인덱스 생성(체크포인트 지원) ==========================================
def _build_index_with_checkpoint(update_pct: Callable[[int, str | None], None],
                                 update_msg: Callable[[str], None],
                                 gdrive_folder_id: str,
                                 gcp_creds: Mapping[str, Any],
                                 persist_dir: str,
                                 remote_manifest: Dict[str, Dict]):
    """
    Google Drive 파일을 '파일ID 단위'로 처리 → 각 파일 완주 후 저장 & 체크포인트 기록.
    재실행 시 이미 완료한 파일(ID+md5 동일)은 건너뜀.
    """
    from llama_index.core import VectorStoreIndex, load_index_from_storage
    from llama_index.readers.google import GoogleDriveReader

    # 준비
    update_pct(15, "Drive 리더 초기화")
    loader = GoogleDriveReader(service_account_key=gcp_creds)

    # 체크포인트/대상 목록 계산
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

    # 스토리지 컨텍스트(있으면 이어쓰기, 없으면 새로 시작)
    os.makedirs(persist_dir, exist_ok=True)
    storage_context = _make_storage_context(persist_dir)

    # 기존 저장본을 우선 로딩(있다면)
    try:
        _ = load_index_from_storage(storage_context)
    except Exception:
        pass  # 처음 생성일 수 있음/부분 저장본일 수 있음 → 무시하고 이어쓰기

    if pending == 0:
        update_pct(95, "변경 없음 → 저장본 그대로 사용")
        # 기존 저장본 반환(없더라도 위에서 예외 처리)
        try:
            return load_index_from_storage(storage_context)
        except Exception:
            # 저장본이 아예 없으면 빈 상태이므로, 그대로 진행하지 말고 에러 안내
            st.error("저장된 두뇌가 없는데 변경도 없다고 감지되었습니다. 초기화 후 다시 시도하세요.")
            st.stop()

    # 파일 단위로 이어서 인덱싱
    for i, fid in enumerate(todo_ids, start=1):
        meta = remote_manifest.get(fid, {})
        pretty_name = meta.get("name") or fid
        update_msg(f"인덱스 생성 • {pretty_name} ({done_cnt + i}/{total})")

        try:
            docs = loader.load_data(file_ids=[fid])  # 파일 1개만 로드
        except TypeError:
            st.error("GoogleDriveReader 버전이 오래되어 file_ids 옵션을 지원하지 않습니다. requirements 업데이트가 필요합니다.")
            st.stop()

        try:
            VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=False)
            storage_context.persist(persist_dir=persist_dir)  # 부분 진행 즉시 저장
            _mark_done(cp, fid, meta)  # 체크포인트 갱신
        except Exception as e:
            st.error(f"인덱스 생성 중 오류: {pretty_name}")
            with st.expander("자세한 오류 보기"):
                st.exception(e)
            st.stop()

        pct = 30 + int((i / max(1, pending)) * 60)  # 30→90 사이
        update_pct(pct, f"문서 {done_cnt + i}/{total} 로드 → 인덱스 생성")

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

# (참고) 기존 일괄 빌드 함수(미사용)
def _build_index_with_progress(update_pct: Callable[[int, str | None], None],
                               update_msg: Callable[[str], None],
                               gdrive_folder_id: str,
                               gcp_creds: Mapping[str, Any],
                               persist_dir: str):
    from llama_index.core import VectorStoreIndex
    from llama_index.readers.google import GoogleDriveReader

    update_pct(15, "Drive 리더 초기화")
    loader = GoogleDriveReader(service_account_key=gcp_creds)

    update_pct(30, "문서 목록 불러오는 중")
    documents = loader.load_data(folder_id=gdrive_folder_id)
    if not documents:
        st.error("강의 자료 폴더가 비었거나 권한 문제입니다. folder_id/공유권한을 확인하세요.")
        st.stop()

    update_pct(60, f"문서 {len(documents)}개 로드 → 인덱스 생성")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    update_pct(90, "두뇌 저장 중")
    index.storage_context.persist(persist_dir=persist_dir)
    update_pct(100, "완료")
    return index

# ====== 변경 감지 → 인덱스 준비(체크포인트 포함) ==============================
def get_or_build_index(update_pct: Callable[[int, str | None], None],
                       update_msg: Callable[[str], None],
                       gdrive_folder_id: str,
                       raw_sa: Any | None,
                       persist_dir: str,
                       manifest_path: str):
    """
    Drive 변경을 감지해 저장본을 쓰거나, 변경 시에만 재인덱싱.
    재인덱싱은 '파일 단위 체크포인트'를 사용하여 중간 재개를 지원.
    """
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

    # 변경 없음 → 저장본 바로 로드
    if os.path.exists(persist_dir) and not _manifests_differ(local, remote):
        update_pct(25, "변경 없음 → 저장된 두뇌 로딩")
        from llama_index.core import StorageContext, load_index_from_storage
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        idx = load_index_from_storage(storage_context)
        update_pct(100, "완료!")
        return idx

    # 변경 있음 → 체크포인트 이어서 빌드
    update_pct(40, "변경 감지 → 문서 로드/인덱스 생성 (체크포인트 사용)")
    idx = _build_index_with_checkpoint(update_pct, update_msg, gdrive_folder_id, gcp_creds, persist_dir, remote)

    # 새 매니페스트 저장 & 체크포인트 정리
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(remote, fp, ensure_ascii=False, indent=2, sort_keys=True)
    clear_checkpoint()  # 모든 파일 완료했으니 체크포인트 비움

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
