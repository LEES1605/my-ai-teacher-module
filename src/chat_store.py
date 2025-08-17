# src/chat_store.py — Google Drive JSONL 저장(서비스계정)

from __future__ import annotations
import io, json, datetime as dt
from typing import Any, Mapping, List
from googleapiclient.http import MediaInMemoryUpload
from googleapiclient.discovery import build
from google.oauth2 import service_account

def _build_svc(sa_json: Mapping[str, Any]):
    scopes = ["https://www.googleapis.com/auth/drive"]
    creds = service_account.Credentials.from_service_account_info(sa_json, scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def make_entry(session_id: str, role: str, speaker: str, content: str, mode: str, model: str) -> dict:
    return {
        "session_id": session_id,
        "role": role,
        "speaker": speaker,
        "content": content,
        "mode": mode,
        "model": model,
        "ts": dt.datetime.utcnow().isoformat() + "Z",
    }

def append_jsonl(folder_id: str, sa_json: Mapping[str, Any], items: List[dict]) -> str:
    svc = _build_svc(sa_json)
    ts = dt.datetime.utcnow().strftime("%Y-%m-%d")
    name = f"chat_log_{ts}.jsonl"

    # 파일 조회
    q = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    res = svc.files().list(q=q, fields="files(id,name)", includeItemsFromAllDrives=True, supportsAllDrives=True).execute()
    files = res.get("files", [])
    if files:
        file_id = files[0]["id"]
        # 기존 내용 읽기
        content = svc.files().get_media(fileId=file_id).execute()
        text = content.decode("utf-8")
        for it in items:
            text += json.dumps(it, ensure_ascii=False) + "\n"
        media = MediaInMemoryUpload(text.encode("utf-8"), mimetype="application/json", resumable=False)
        svc.files().update(fileId=file_id, media_body=media, supportsAllDrives=True).execute()
        return file_id
    else:
        text = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
        media = MediaInMemoryUpload(text.encode("utf-8"), mimetype="application/json", resumable=False)
        meta = {"name": name, "parents": [folder_id]}
        created = svc.files().create(body=meta, media_body=media, fields="id", supportsAllDrives=True).execute()
        return created["id"]
