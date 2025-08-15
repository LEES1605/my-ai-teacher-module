# src/drive_log.py
from __future__ import annotations
from typing import Any, Mapping
from datetime import datetime

# Google Drive API
from googleapiclient.discovery import build
from google.oauth2 import service_account

# Streamlit (토스트/메시지 용, 실패해도 동작하도록 선택적 사용)
try:
    import streamlit as st
except Exception:
    st = None

CHAT_SUBFOLDER_NAME = "chat_log"

def _build_service(sa_json: Mapping[str, Any]) -> Any:
    scopes = ["https://www.googleapis.com/auth/drive"]
    creds = service_account.Credentials.from_service_account_info(sa_json, scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def _ensure_subfolder(service, parent_folder_id: str, name: str) -> str:
    # 같은 이름 폴더가 있으면 그걸 쓰고, 없으면 생성
    q = (
        f"('{parent_folder_id}' in parents) and "
        f"name = '{name}' and "
        f"mimeType = 'application/vnd.google-apps.folder' and trashed=false"
    )
    res = service.files().list(
        q=q,
        fields="files(id,name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        pageSize=10,
    ).execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]

    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    created = service.files().create(
        body=meta,
        fields="id",
        supportsAllDrives=True,
    ).execute()
    return created["id"]

def get_chatlog_folder_id(parent_folder_id: str, sa_json: Mapping[str, Any]) -> str:
    """상위 데이터 폴더 아래에 chat_log/ 서브폴더를 보장 생성하고 그 ID를 반환."""
    svc = _build_service(sa_json)
    return _ensure_subfolder(svc, parent_folder_id, CHAT_SUBFOLDER_NAME)

def save_chatlog_markdown(session_id: str, messages: list[dict], parent_folder_id: str, sa_json: Mapping[str, Any] | None = None):
    """
    대화 전체를 Markdown으로 저장.
    - parent_folder_id 아래에 chat_log/ 서브폴더를 보장 생성
    - 파일명: YYYY-MM-DD__{session_id}.md
    """
    if sa_json is None:
        raise ValueError("save_chatlog_markdown: service account json 필요")

    svc = _build_service(sa_json)
    sub_id = _ensure_subfolder(svc, parent_folder_id, CHAT_SUBFOLDER_NAME)

    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    filename = f"{date_str}__{session_id}.md"

    # Markdown 조립 (아주 단순)
    lines = ["# Chat Log", f"- session: `{session_id}`", ""]
    for m in messages:
        role = m.get("role", "")
        content = str(m.get("content", "")).strip()
        lines.append(f"## {role}")
        lines.append(content)
        lines.append("")
    data = "\n".join(lines).encode("utf-8")

    media = {"mimeType": "text/markdown"}
    file_metadata = {"name": filename, "parents": [sub_id]}
    svc.files().create(
        body=file_metadata,
        media_body=None,
        fields="id",
        supportsAllDrives=True,
    ).execute()

    # 실제 콘텐츠 업로드 (resumable 말고 간단 업로드)
    upload = svc.files().create(
        body={"name": filename, "parents": [sub_id]},
        media_body=None,
        fields="id",
        supportsAllDrives=True,
    )
    # 위에서 본문 없이 만들었으면, 간단히 update로 컨텐츠 업로드
    # 간단화를 위해 아래처럼 한 번 더 호출해도 무방 (드라이브는 이름 중복 허용)
    svc.files().create(
        body={"name": filename, "parents": [sub_id]},
        media_body=data,
        fields="id",
        supportsAllDrives=True,
    ).execute()

    if st: st.toast("Drive에 Markdown 대화 저장 완료 (chat_log/)", icon="💾")
# src/drive_log.py - 하단에 추가

from googleapiclient.http import MediaInMemoryUpload

def _ensure_folder(svc, name: str, parent_id: str | None = None) -> str:
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        q += f" and '{parent_id}' in parents"
    res = svc.files().list(q=q, fields="files(id,name)").execute()
    if res.get("files"):
        return res["files"][0]["id"]
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        meta["parents"] = [parent_id]
    f = svc.files().create(body=meta, fields="id").execute()
    return f["id"]

def _messages_to_markdown(session_id: str, messages: list[dict]) -> str:
    lines = [f"# Chat session {session_id}"]
    for m in messages:
        role = m.get("role", "user")
        lines.append(f"\n## {role}\n\n{m.get('content','')}")
    return "\n".join(lines).strip() + "\n"

def save_chatlog_markdown_oauth(session_id: str, messages: list[dict], svc, parent_folder_id: str | None = None) -> str:
    """OAuth(사용자 소유)로 MD 저장. 반환: 파일 ID"""
    if not svc:
        raise RuntimeError("Drive service is None")

    # 상위 폴더 결정: 지정 없으면 my-ai-teacher-data/chat_log 생성
    root_id = parent_folder_id or _ensure_folder(svc, "my-ai-teacher-data")
    chat_id = _ensure_folder(svc, "chat_log", root_id)

    from datetime import datetime, timezone, timedelta
    KST = timezone(timedelta(hours=9))
    stamp = datetime.now(KST).strftime("%Y-%m-%d__%H%M%S")
    filename = f"{stamp}.md"

    md = _messages_to_markdown(session_id, messages).encode("utf-8")
    media = MediaInMemoryUpload(md, mimetype="text/markdown", resumable=False)
    meta = {"name": filename, "parents": [chat_id]}
    f = svc.files().create(body=meta, media_body=media, fields="id").execute()
    return f["id"]
