# src/drive_log.py
# -----------------------------------------------------------
# 대화를 Google Drive에 저장하는 유틸리티
# - 상위 폴더: 전달받은 parent_folder_id (없으면 settings.GDRIVE_FOLDER_ID)
# - 하위 폴더: chat_log (없으면 생성)
# - 파일명: YYYYMMDD_session-<session_id>.md
# - Shared Drive 대응: supportsAllDrives/includeItemsFromAllDrives 활용
# -----------------------------------------------------------

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any, Iterable

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

from src.config import settings

SCOPES_WRITE = ["https://www.googleapis.com/auth/drive.file"]
MIME_FOLDER = "application/vnd.google-apps.folder"

def _load_creds_write() -> Credentials:
    raw = settings.GDRIVE_SERVICE_ACCOUNT_JSON
    info = json.loads(raw) if isinstance(raw, str) else dict(raw)
    return Credentials.from_service_account_info(info, scopes=SCOPES_WRITE)

def _drive():
    # cache_discovery=False: Streamlit Cloud 캐시 이슈 회피
    return build("drive", "v3", credentials=_load_creds_write(), cache_discovery=False)

def _ensure_subfolder(parent_id: str, name: str) -> str:
    svc = _drive()
    q = (
        f"mimeType='{MIME_FOLDER}' and name='{name}' "
        f"and '{parent_id}' in parents and trashed=false"
    )
    res = svc.files().list(
        q=q, fields="files(id, name)", pageSize=1,
        includeItemsFromAllDrives=True, supportsAllDrives=True
    ).execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]

    meta = {"name": name, "mimeType": MIME_FOLDER, "parents": [parent_id]}
    folder = svc.files().create(
        body=meta, fields="id", supportsAllDrives=True
    ).execute()
    return folder["id"]

def _find_file(parent_id: str, name: str) -> str | None:
    svc = _drive()
    q = (
        f"name='{name}' and '{parent_id}' in parents and "
        f"mimeType!='{MIME_FOLDER}' and trashed=false"
    )
    res = svc.files().list(
        q=q, fields="files(id, name)", pageSize=1,
        includeItemsFromAllDrives=True, supportsAllDrives=True
    ).execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None

def _create_file(parent_id: str, name: str, content: bytes, mime: str) -> str:
    svc = _drive()
    meta = {"name": name, "parents": [parent_id]}
    media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime, resumable=False)
    file = svc.files().create(
        body=meta, media_body=media, fields="id",
        supportsAllDrives=True
    ).execute()
    return file["id"]

def _update_file(file_id: str, content: bytes, mime: str) -> None:
    svc = _drive()
    media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime, resumable=False)
    svc.files().update(
        fileId=file_id, media_body=media, supportsAllDrives=True
    ).execute()

def _to_markdown(session_id: str, messages: Iterable[dict[str, Any]]) -> str:
    now = datetime.now(timezone.utc).astimezone()  # 로컬 타임존 표시
    head = [
        f"# Chat Log — {now:%Y-%m-%d %H:%M:%S %Z}",
        f"- session_id: `{session_id}`",
        "",
        "---",
        "",
    ]
    lines: list[str] = head
    for m in messages:
        role = m.get("role", "assistant")
        content = str(m.get("content", "")).rstrip()
        title = "👤 User" if role == "user" else "🤖 Assistant"
        lines.append(f"### {title}")
        lines.append("")
        lines.append(content if content else "_(empty)_")
        lines.append("")
    return "\n".join(lines).strip() + "\n"

def save_chatlog_markdown(
    session_id: str,
    messages: Iterable[dict[str, Any]],
    parent_folder_id: str | None = None,
) -> str:
    """
    현재 세션 대화 전체를 chat_log/<YYYYMMDD_session-...>.md 로 저장/업데이트.
    반환값: 파일 ID
    """
    parent_root = (parent_folder_id or
                   getattr(settings, "CHATLOG_FOLDER_ID", "") or
                   settings.GDRIVE_FOLDER_ID)
    if not parent_root:
        raise RuntimeError("GDRIVE_FOLDER_ID/CHATLOG_FOLDER_ID가 비어 있습니다.")

    # 하위 폴더 chat_log 보장
    chatlog_folder_id = _ensure_subfolder(parent_root, "chat_log")

    # 파일명 (하루에 한 파일)
    today = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d")
    fname = f"{today}_session-{session_id}.md"

    md = _to_markdown(session_id, messages)
    data = md.encode("utf-8")
    mime = "text/markdown"

    file_id = _find_file(chatlog_folder_id, fname)
    if file_id:
        _update_file(file_id, data, mime)
        return file_id
    else:
        return _create_file(chatlog_folder_id, fname, data, mime)
