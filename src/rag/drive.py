# src/rag/drive.py
from __future__ import annotations
import os, io, json, zipfile, shutil
from typing import Any, Mapping, List, Tuple, Dict

# 백업 파일 이름 접두사(날짜가 뒤에 붙음)
INDEX_BACKUP_PREFIX = "ai_brain_cache"

# ── 서비스계정 JSON 정규화/검증 ────────────────────────────────────────────────
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
    import streamlit as st
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

# ── Google Drive API 래퍼 ─────────────────────────────────────────────────────
def _build_drive_service(creds_dict: Mapping[str, Any], write: bool = False):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    scopes = ["https://www.googleapis.com/auth/drive.readonly"] if not write else ["https://www.googleapis.com/auth/drive"]
    creds = service_account.Credentials.from_service_account_info(dict(creds_dict), scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def fetch_drive_manifest(creds_dict: Mapping[str, Any], folder_id: str) -> dict:
    """폴더 내 파일 메타데이터(이름/md5/수정시각 등) 맵을 반환."""
    svc = _build_drive_service(creds_dict, write=False)
    files = []; page_token = None
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

# ── ZIP 백업/복원 + 보관정책 ──────────────────────────────────────────────────
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
    fname = filename or f"{INDEX_BACKUP_PREFIX}.zip"  # 외부에서 날짜를 붙여 호출해도 됨
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
