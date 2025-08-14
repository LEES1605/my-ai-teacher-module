# src/chat_store.py
# Google Drive에 대화 로그를 일자별 .jsonl 파일로 저장(append)
# - 서비스계정 JSON(secrets) 사용
# - 파일이 없으면 생성, 있으면 내용을 읽어 추가 후 업데이트
# - 작은 로그부터 안정적으로 사용 (수 MB 이상 커지면 회전 권장)

from __future__ import annotations
from typing import Any, Mapping, Iterable
import io, json, re, datetime as dt

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

# ==== 내부 유틸 =================================================================

def _normalize_sa(raw_sa: Any | None) -> Mapping[str, Any] | None:
    """서비스계정 JSON 문자열/딕트를 dict로 정규화(+private_key 개행 보정)"""
    if raw_sa is None:
        return None
    if isinstance(raw_sa, Mapping):
        return dict(raw_sa)
    if isinstance(raw_sa, str):
        s = raw_sa.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            pass
        # private_key 개행 보정
        try:
            m = re.search(r'"private_key"\s*:\s*"(?P<key>.*?)"', s, re.DOTALL)
            if m:
                key = m.group("key")
                key_fixed = key.replace("\r\n", "\n").replace("\n", "\\n")
                s_fixed = s[:m.start("key")] + key_fixed + s[m.end("key"):]
                return json.loads(s_fixed)
        except Exception:
            pass
    return None

def _drive_service_for_write(sa_dict: Mapping[str, Any]):
    """쓰기 가능한 최소 권한 scope: drive.file"""
    scopes = ["https://www.googleapis.com/auth/drive.file"]
    creds = service_account.Credentials.from_service_account_info(sa_dict, scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

# ==== 공개 API ==================================================================

def ensure_daily_log_file(folder_id: str, sa_json: Any, date: dt.date | None = None) -> str:
    """해당 날짜의 로그 파일이 없으면 생성하고, 파일 ID를 반환."""
    date = date or dt.date.today()
    fname = f"chat_log_{date.isoformat()}.jsonl"

    sa_dict = _normalize_sa(sa_json)
    if not sa_dict:
        raise RuntimeError("서비스계정 JSON 파싱 실패(GOOGLE_SERVICE_ACCOUNT_JSON).")

    svc = _drive_service_for_write(sa_dict)
    q = f"name = '{fname}' and '{folder_id}' in parents and trashed = false"
    fields = "files(id,name)"
    resp = svc.files().list(
        q=q, fields=fields, pageSize=1, supportsAllDrives=True, includeItemsFromAllDrives=True
    ).execute()
    files = resp.get("files", [])
    if files:
        return files[0]["id"]

    # 없으면 새로 생성(빈 파일)
    meta = {"name": fname, "parents": [folder_id], "mimeType": "text/plain"}
    media = MediaIoBaseUpload(io.BytesIO(b""), mimetype="text/plain", resumable=False)
    created = svc.files().create(
        body=meta, media_body=media, fields="id", supportsAllDrives=True
    ).execute()
    return created["id"]

def _download_bytes(svc, file_id: str) -> bytes:
    req = svc.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()

def append_jsonl(folder_id: str, sa_json: Any, items: Iterable[dict]) -> None:
    """items(딕트들)를 JSON Lines로 해당 날짜 로그 파일 끝에 append."""
    sa_dict = _normalize_sa(sa_json)
    if not sa_dict:
        raise RuntimeError("서비스계정 JSON 파싱 실패(GOOGLE_SERVICE_ACCOUNT_JSON).")

    svc = _drive_service_for_write(sa_dict)
    file_id = ensure_daily_log_file(folder_id, sa_json)

    # 기존 내용 다운로드 → 끝에 새 줄들 추가 → 업데이트
    try:
        old = _download_bytes(svc, file_id)
    except Exception:
        # 새 파일일 수도 있으니 실패 시 빈 바이트로 간주
        old = b""

    new_tail = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items).encode("utf-8")
    merged = old + new_tail

    media = MediaIoBaseUpload(io.BytesIO(merged), mimetype="text/plain", resumable=False)
    svc.files().update(
        fileId=file_id, media_body=media, supportsAllDrives=True
    ).execute()

def make_entry(session_id: str, role: str, agent: str, text: str, mode: str, model: str | None = None) -> dict:
    """로그 한 줄 포맷(UTC ISO 타임스탬프 포함)."""
    return {
        "ts": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "session": session_id,
        "role": role,       # 'user' | 'assistant'
        "agent": agent,     # 'user' | 'Gemini' | 'ChatGPT'
        "model": model or agent,
        "mode": mode,       # '이유문법 설명' | '구문 분석' | ...
        "text": text,
    }
