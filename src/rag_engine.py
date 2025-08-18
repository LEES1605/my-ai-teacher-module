# ===== [01] TOP ==============================================================
# RAG Engine — 에러메시지 사용자 친화화 + 단계별 진행상태 콜백 + 상세 진단 힌트
from __future__ import annotations
import os, json, traceback
from pathlib import Path
from typing import Callable, Any

# ===== [02] CONFIG BRIDGE ====================================================
try:
    from src.config import settings, PERSIST_DIR
except Exception:
    from config import settings, PERSIST_DIR  # 폴백

# ===== [03] 사용자 친화 예외 정의 ============================================
class RAGEngineError(Exception):
    def __init__(self, public_msg: str, debug: str | None = None):
        super().__init__(public_msg)
        self.public_msg = public_msg
        self.debug = debug

class SecretsMissing(RAGEngineError): ...
class ServiceAccountInvalid(RAGEngineError): ...
class FolderIdMissing(RAGEngineError): ...
class DriveRestoreFailed(RAGEngineError): ...
class LocalIndexMissing(RAGEngineError): ...
class IndexLoadFailed(RAGEngineError): ...
class LlamaInitFailed(RAGEngineError): ...
class QueryEngineNotReady(RAGEngineError): ...

# ===== [04] 콜백 유틸 ========================================================
def _safe(cb: Callable[..., Any] | None, *a, **kw):
    try:
        if cb: cb(*a, **kw)
    except Exception:
        pass

def _emit(update_pct: Callable[[int], None] | None = None,
          update_msg: Callable[[str], None] | None = None,
          pct: int | None = None, msg: str | None = None):
    if pct is not None:
        _safe(update_pct, pct)
    if msg:
        _safe(update_msg, msg)

# ===== [05] SA JSON normalize / validate ====================================
def _normalize_sa(raw_sa: Any) -> str:
    if raw_sa is None:
        raise SecretsMissing("서비스 계정 키가 없습니다.", "GDRIVE_SERVICE_ACCOUNT_JSON is None")

    try:
        from pydantic.types import SecretStr
        if isinstance(raw_sa, SecretStr):
            raw_sa = raw_sa.get_secret_value()
    except Exception:
        pass

    if isinstance(raw_sa, dict):
        return json.dumps(raw_sa, ensure_ascii=False)
    if isinstance(raw_sa, str):
        s = raw_sa.strip()
        if not s:
            raise SecretsMissing("서비스 계정 키가 비어 있습니다.")
        return s
    try:
        return str(raw_sa)
    except Exception:
        raise ServiceAccountInvalid("서비스 계정 키 형식을 알 수 없습니다.")

def _validate_sa(json_str: str) -> dict:
    try:
        data = json.loads(json_str)
    except Exception as e:
        raise ServiceAccountInvalid(
            "서비스 계정 키(JSON) 파싱 실패. private_key 줄바꿈(\\n)과 TOML 따옴표(''' ... ''')를 확인하세요.",
            debug=repr(e)
        )
    needed = {"type","private_key","client_email"}
    if not needed.issubset(set(data.keys())):
        raise ServiceAccountInvalid("서비스 계정 키에 type/private_key/client_email 항목이 없습니다.")
    return data

# ===== [06] LLM/임베딩 초기화 ===============================================
def init_llama_settings(api_key: str, llm_model: str, embed_model: str, temperature: float = 0.0):
    try:
        if not api_key:
            raise ValueError("Gemini API Key가 비어 있습니다.")
        if not llm_model:
            raise ValueError("LLM 모델명이 비어 있습니다.")
        if not embed_model:
            raise ValueError("임베딩 모델명이 비어 있습니다.")
        return True
    except Exception as e:
        raise LlamaInitFailed("LLM/임베딩 초기화 중 오류. API Key/모델명을 확인하세요.", debug=repr(e))

# ===== [07] 인덱스 로드/생성 =================================================
def _index_exists(persist_dir: str | Path) -> bool:
    p = Path(persist_dir)
    if not p.exists(): return False
    try:
        return any(p.iterdir())
    except Exception:
        return False

def _load_index_from_disk(persist_dir: str | Path):
    try:
        if not _index_exists(persist_dir):
            raise LocalIndexMissing("로컬 인덱스가 없습니다. 먼저 복구/생성하세요.")
        class _DummyIndex:
            def as_query_engine(self, **kw):
                class _QE:
                    def query(self, q):
                        return type("A", (), {"response": f"[더미응답] {q}"})
                return _QE()
        return _DummyIndex()
    except RAGEngineError:
        raise
    except Exception as e:
        raise IndexLoadFailed("인덱스 로드 중 오류가 발생했습니다. 인덱스 폴더가 손상되었을 수 있습니다.", debug=repr(e))

def get_or_build_index(
    update_pct: Callable[[int], None] | None = None,
    update_msg: Callable[[str], None] | None = None,
    gdrive_folder_id: str | None = None,
    raw_sa: Any | None = None,
    persist_dir: str | Path = PERSIST_DIR,
    manifest_path: str | None = None,
):
    _emit(update_pct, update_msg, 2, "설정 확인 중…")

    # 1) 로컬
    if _index_exists(persist_dir):
        _emit(update_pct, update_msg, 15, "로컬 인덱스 감지 → 로드합니다…")
        return _load_index_from_disk(persist_dir)

    # 2) Drive 복구
    _emit(update_pct, update_msg, 25, "로컬에 없어요. Drive에서 복구 시도…")
    if not gdrive_folder_id:
        raise FolderIdMissing("Google Drive 폴더 ID가 비어 있습니다.", "settings.GDRIVE_FOLDER_ID/BACKUP_FOLDER_ID 확인")

    creds = _validate_sa(_normalize_sa(raw_sa))
    ok, note = try_restore_index_from_drive(creds, persist_dir, gdrive_folder_id, update_msg=update_msg)
    if not ok:
        sa_email = creds.get("client_email", "(unknown)")
        # ⬇️ 폴더 ID/SA 이메일/확인 리스트까지 포함해서 사용자 문장으로 안내
        hint = (
            f"Drive에서 인덱스를 찾지 못했거나 권한이 없습니다.\n"
            f"- 폴더 ID: {gdrive_folder_id}\n"
            f"- 서비스 계정: {sa_email}\n"
            f"- 확인:\n"
            f"  1) 해당 폴더의 공유 대상에 위 서비스 계정 이메일을 추가했는지\n"
            f"  2) 폴더 안에 인덱스 파일들이 실제로 존재하는지\n"
            f"  3) 권한이 최소 보기(Reader) 이상인지\n"
            f"{f'  4) 참고: {note}' if note else ''}"
        )
        raise DriveRestoreFailed(hint)

    _emit(update_pct, update_msg, 65, "복구됨 → 로드합니다…")
    return _load_index_from_disk(persist_dir)

# ===== [08] Drive 복구(실제 다운로드 구현) ===================================
def try_restore_index_from_drive(
    creds: dict,
    persist_dir: str | Path,
    folder_id: str,
    update_msg: Callable[[str], None] | None = None,
) -> tuple[bool, str | None]:
    """
    Google Drive v3 API를 사용해 folder_id 하위의 파일/폴더를 재귀적으로 내려받아
    persist_dir에 동일한 구조로 복구합니다.
    - creds: 서비스 계정 JSON(dict)
    - persist_dir: 로컬 저장 경로
    - folder_id: 백업 폴더 ID
    반환: (성공여부, 참고메모)
    """
    from pathlib import Path
    import io
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    def _emit_msg(m: str):
        try:
            if update_msg: update_msg(m)
        except Exception:
            pass

    try:
        if not folder_id or not str(folder_id).strip():
            raise FolderIdMissing("폴더 ID가 비어 있습니다.")
        if "client_email" not in creds:
            raise ServiceAccountInvalid("서비스 계정 키에 client_email이 없습니다.")

        # [08-1] 자격증명/드라이브 클라이언트
        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        credentials = Credentials.from_service_account_info(creds, scopes=scopes)
        svc = build("drive", "v3", credentials=credentials, cache_discovery=False)

        # [08-2] 헬퍼: 폴더/파일 구분 및 리스트
        def _list_children(fid: str) -> list[dict]:
            files = []
            page_token = None
            query = f"'{fid}' in parents and trashed=false"
            while True:
                res = svc.files().list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, size)",
                    pageToken=page_token,
                ).execute()
                files.extend(res.get("files", []))
                page_token = res.get("nextPageToken")
                if not page_token:
                    break
            return files

        def _is_folder(item: dict) -> bool:
            return item.get("mimeType") == "application/vnd.google-apps.folder"

        # [08-3] 헬퍼: Google Docs류는 export, 일반 파일은 download
        # 인덱스는 보통 일반 파일(json, pkl, bin, txt 등)이므로 우선 일반 다운로드에 초점
        GOOGLE_DOC_EXPORT = {
            "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
            "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
            "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
        }

        def _download_file(file_id: str, name: str, mime_type: str, out_path: Path):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if mime_type in GOOGLE_DOC_EXPORT:
                export_mime, ext = GOOGLE_DOC_EXPORT[mime_type]
                request = svc.files().export_media(fileId=file_id, mimeType=export_mime)
                fh = io.FileIO(str(out_path.with_suffix(ext)), "wb")
            else:
                request = svc.files().get_media(fileId=file_id)
                fh = io.FileIO(str(out_path), "wb")
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _status, done = downloader.next_chunk()
            fh.close()

        # [08-4] 재귀 내려받기
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        downloaded_count = 0

        def _walk_and_download(cur_folder_id: str, dst_dir: Path):
            nonlocal downloaded_count
            items = _list_children(cur_folder_id)
            for it in items:
                fname = it.get("name", "unnamed")
                fid   = it.get("id")
                mime  = it.get("mimeType", "")
                if _is_folder(it):
                    _emit_msg(f"폴더: {fname} 내려받는 중…")
                    _walk_and_download(fid, dst_dir / fname)
                else:
                    _emit_msg(f"파일: {fname} 내려받는 중…")
                    _download_file(fid, fname, mime, dst_dir / fname)
                    downloaded_count += 1

        _emit_msg("Drive에서 백업 파일을 내려받는 중…")
        _walk_and_download(folder_id, persist_dir)

        if downloaded_count == 0:
            # 폴더는 있었지만 내부가 비었거나 권한 부족
            return (False, "폴더에 다운로드할 파일이 없거나 접근 권한이 부족합니다.")
        _emit_msg(f"다운로드 완료: {downloaded_count}개 파일")
        return (True, f"{downloaded_count} files downloaded")

    except RAGEngineError:
        raise
    except Exception as e:
        raise DriveRestoreFailed("Drive 복구 중 예기치 못한 오류가 발생했습니다.", debug=repr(e))

# ===== [09] 질의/답변 ========================================================
def get_text_answer(query_engine: Any, prompt: str, sys_prompt: str) -> str:
    if not query_engine:
        raise QueryEngineNotReady("질의 엔진이 준비되지 않았습니다. 먼저 ‘AI 두뇌 준비’를 실행하세요.")
    try:
        q = f"{sys_prompt}\n\n사용자: {prompt}"
        res = query_engine.query(q)
        if hasattr(res, "response"):
            return res.response
        return str(res)
    except RAGEngineError:
        raise
    except Exception as e:
        raise RAGEngineError("답변 생성 중 오류가 발생했습니다. 우측 Traceback과 관리자 설정을 확인하세요.", debug=repr(e))

# ===== [10] 디버그 헬퍼 ======================================================
def _format_debug(e: Exception) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))
