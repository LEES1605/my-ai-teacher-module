# src/drive_log.py — OAuth Markdown 저장 + 서비스계정 chat_log 폴더 보장(JSONL용)

from __future__ import annotations
import io, json, datetime as dt
from typing import Any, Mapping, Optional

from googleapiclient.http import MediaInMemoryUpload
from googleapiclient.discovery import build

# ---------- OAuth 기반: 내 드라이브에 Markdown 저장 ----------
def _ensure_folder(service, name: str, parent_id: Optional[str]) -> str:
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    res = service.files().list(
        q=q, fields="files(id,name)", includeItemsFromAllDrives=True, supportsAllDrives=True
    ).execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    created = service.files().create(body=meta, fields="id", supportsAllDrives=True).execute()
    return created["id"]

def save_chatlog_markdown_oauth(session_id: str, messages: list[dict], svc, parent_id: Optional[str] = None) -> str:
    """
    내 드라이브에 my-ai-teacher-data/chat_log/ 하위에 Markdown으로 저장.
    parent_id를 주면 그 폴더 아래에 my-ai-teacher-data/chat_log를 생성.
    """
    root = _ensure_folder(svc, "my-ai-teacher-data", parent_id)
    sub = _ensure_folder(svc, "chat_log", root)

    ts = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    name = f"chat_{session_id}_{ts}.md"
    md_lines = []
    for m in messages:
        role = m.get("role")
        text = str(m.get("content", ""))
        md_lines.append(f"### {role}\n\n{text}\n")
    data = "\n".join(md_lines).encode("utf-8")
    media = MediaInMemoryUpload(data, mimetype="text/markdown", resumable=False)
    meta = {"name": name, "parents": [sub]}
    created = svc.files().create(
        body=meta, media_body=media, fields="id,webViewLink", supportsAllDrives=True
    ).execute()
    return created["id"]

# ---------- 서비스계정: chat_log 하위 폴더 보장(JSONL 저장용) ----------
def _build_sa_service(sa_json: Mapping[str, Any]):
    from google.oauth2 import service_account
    scopes = ["https://www.googleapis.com/auth/drive"]
    creds = service_account.Credentials.from_service_account_info(sa_json, scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def get_chatlog_folder_id(parent_folder_id: str, sa_json: Mapping[str, Any]) -> str:
    """parent_folder_id 아래에 chat_log 폴더가 없으면 생성 후 ID 반환"""
    svc = _build_sa_service(sa_json)
    q = (
        f"name='chat_log' and mimeType='application/vnd.google-apps.folder' "
        f"and trashed=false and '{parent_folder_id}' in parents"
    )
    res = svc.files().list(q=q, fields="files(id,name)", includeItemsFromAllDrives=True, supportsAllDrives=True).execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]
    meta = {
        "name": "chat_log",
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    created = svc.files().create(body=meta, fields="id", supportsAllDrives=True).execute()
    return created["id"]
