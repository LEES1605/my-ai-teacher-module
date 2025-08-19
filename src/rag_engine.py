# ===== [01] TOP ==============================================================
# RAG Engine — 사용자 친화 에러 + Drive 복구/백업(P0 스텁 포함)
from __future__ import annotations

import io
import json
import traceback
from pathlib import Path
from typing import Any, Callable, Optional, List, Dict, Tuple

# ===== [02] CONFIG BRIDGE ====================================================
try:
    from src.config import settings, PERSIST_DIR
except Exception:
    # 루트 폴백
    from config import settings, PERSIST_DIR

# ===== [03] 사용자 친화 예외 ==================================================
class RAGEngineError(Exception):
    def __init__(self, public_msg: str, debug: Optional[str] = None):
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
def _safe(cb: Optional[Callable[..., Any]], *a: Any, **kw: Any) -> None:
    try:
        if cb:
            cb(*a, **kw)
    except Exception:
        # 콜백 오류는 삼킨다(진행 방해 X)
        pass

def _emit(
    update_pct: Optional[Callable[[int], None]] = None,
    update_msg: Optional[Callable[[str], None]] = None,
    pct: Optional[int] = None,
    msg: Optional[str] = None,
) -> None:
    if pct is not None:
        _safe(update_pct, int(pct))
    if msg:
        _safe(update_msg, msg)

# ===== [05] Secret/Service Account normalize & validate ======================
def _normalize_sa(raw_sa: Any) -> str:
    if raw_sa is None:
        raise SecretsMissing("서비스 계정 키가 없습니다.", "GDRIVE_SERVICE_ACCOUNT_JSON is None")

    # pydantic SecretStr 지원
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

def _validate_sa(json_str: str) -> Dict[str, Any]:
    try:
        data = json.loads(json_str)
    except Exception as e:
        raise ServiceAccountInvalid(
            "서비스 계정 키(JSON) 파싱 실패. private_key 줄바꿈(\\n)과 TOML 따옴표(''' ... ''')를 확인하세요.",
            debug=repr(e),
        )
    needed = {"type", "private_key", "client_email"}
    if not needed.issubset(set(data.keys())):
        raise ServiceAccountInvalid("서비스 계정 키에 type/private_key/client_email 항목이 없습니다.")
    return data

# ===== [06] LLM/임베딩 초기화 ===============================================
def init_llama_settings(api_key: str, llm_model: str, embed_model: str, temperature: float = 0.0) -> bool:
    """
    실제 라이브러리 초기화가 있다면 이곳에 연결.
    P0: 파라미터 유효성만 확인하여 최소 동작 보장.
    """
    try:
        if not api_key:
            raise ValueError("Gemini API Key가 비어 있습니다.")
        if not llm_model:
            raise ValueError("LLM 모델명이 비어 있습니다.")
        if not embed_model:
            raise ValueError("임베딩 모델명이 비어 있습니다.")
        _ = float(temperature)
        return True
    except Exception as e:
        raise LlamaInitFailed("LLM/임베딩 초기화 중 오류. API Key/모델명을 확인하세요.", debug=repr(e))

# ===== [07] 인덱스 로드/존재 체크 ===========================================
def _index_exists(persist_dir: str | Path) -> bool:
    p = Path(persist_dir)
    if not p.exists():
        return False
    try:
        return any(p.iterdir())
    except Exception:
        return False

def _load_index_from_disk(persist_dir: str | Path) -> Any:
    """
    실제 프로젝트의 인덱스 로딩 코드를 이 함수에 연결하세요.
    지금은 안전한 더미 인덱스를 반환해 최소 동작 보장합니다.
    """
    try:
        if not _index_exists(persist_dir):
            raise LocalIndexMissing("로컬 인덱스가 없습니다. 먼저 복구/생성하세요.")

        # === 여기에 '진짜' 인덱스 로드 코드를 붙여주세요 ===
        class _DummyIndex:
            def as_query_engine(self, **kw: Any) -> Any:
                class _QE:
                    def query(self, q: str) -> Any:
                        return type("A", (), {"response": f"[더미응답] {q}"})
                return _QE()
        return _DummyIndex()
    except RAGEngineError:
        raise
    except Exception as e:
        raise IndexLoadFailed(
            "인덱스 로드 중 오류가 발생했습니다. 인덱스 폴더가 손상되었을 수 있습니다.",
            debug=repr(e),
        )

# ===== [08] Drive 복구(실제 다운로드 구현) ===================================
def try_restore_index_from_drive(
    creds: Dict[str, Any],
    persist_dir: str | Path,
    folder_id: str,
    update_msg: Optional[Callable[[str], None]] = None,
) -> tuple[bool, Optional[str]]:
    """
    Google Drive v3 API를 사용해 folder_id 하위의 파일/폴더를 재귀적으로 내려받아
    persist_dir에 동일한 구조로 복구합니다.

    requirements.txt 에 다음 패키지 필요:
      - google-api-python-client
      - google-auth
    """
    def _emit_msg(m: str) -> None:
        try:
            if update_msg:
                update_msg(m)
        except Exception:
            pass

    # 의존성 체크
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
    except Exception as e:
        raise DriveRestoreFailed(
            "Google Drive 클라이언트가 설치되어 있지 않습니다. requirements.txt에 "
            "`google-api-python-client`와 `google-auth`를 추가하고 다시 배포하세요.",
            debug=repr(e),
        )

    try:
        if not folder_id or not str(folder_id).strip():
            raise FolderIdMissing("Google Drive 폴더 ID가 비어 있습니다.")
        if "client_email" not in creds:
            raise ServiceAccountInvalid("서비스 계정 키에 client_email이 없습니다.")

        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        credentials = Credentials.from_service_account_info(creds, scopes=scopes)
        svc = build("drive", "v3", credentials=credentials, cache_discovery=False)

        def _list_children(fid: str) -> List[Dict[str, Any]]:
            files: List[Dict[str, Any]] = []
            page_token: Optional[str] = None
            query = f"'{fid}' in parents and trashed=false"
            while True:
                res = svc.files().list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageToken=page_token,
                ).execute()
                files.extend(res.get("files", []))
                page_token = res.get("nextPageToken")
                if not page_token:
                    break
            return files

        def _is_folder(item: Dict[str, Any]) -> bool:
            return item.get("mimeType") == "application/vnd.google-apps.folder"

        GOOGLE_DOC_EXPORT: Dict[str, tuple[str, str]] = {
            "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
            "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
            "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
        }

        def _download_file(file_id: str, name: str, mime_type: str, out_path: Path) -> None:
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
                _, done = downloader.next_chunk()
            fh.close()

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        downloaded_count = 0

        def _walk(fid: str, dst: Path) -> None:
            nonlocal downloaded_count
            for it in _list_children(fid):
                name = it.get("name", "unnamed")
                mime = it.get("mimeType", "")
                if _is_folder(it):
                    _emit_msg(f"폴더: {name} 내려받는 중…")
                    _walk(it["id"], dst / name)
                else:
                    _emit_msg(f"파일: {name} 내려받는 중…")
                    _download_file(it["id"], name, mime, dst / name)
                    downloaded_count += 1

        _emit_msg("Drive에서 백업 파일을 내려받는 중…")
        _walk(folder_id, persist_dir)

        if downloaded_count == 0:
            return (False, "폴더가 비었거나 권한이 부족합니다.")
        _emit_msg(f"다운로드 완료: {downloaded_count}개")
        return (True, f"{downloaded_count} files downloaded")

    except RAGEngineError:
        raise
    except Exception as e:
        raise DriveRestoreFailed("Drive 복구 중 예기치 못한 오류가 발생했습니다.", debug=repr(e))

# ===== [09] 백업/내보내기(스텁) & 정리 =======================================
INDEX_BACKUP_PREFIX = "rag_index_backup"

def prune_old_backups(
    backups: Optional[Dict[str, Any]] = None,
    *args: Any,
    **kwargs: Any,
) -> Tuple[int, int]:
    """
    P0 stub: 오래된 백업 정리. 호출부 호환성 100% 목표.
    - 두 번째 '위치 인자'를 prefix로 받아도 OK
    - 키워드 prefix=..., keep=..., max_to_keep=... 모두 허용
    return: (정리 대상 개수(before), 정리 후 남은 실제 개수(after))
    """
    if backups is None:
        return (0, 0)

    # 1) prefix 추출: 2번째 위치 인자 → args[0], 또는 kwargs["prefix"]
    prefix: Optional[str] = None
    if args and isinstance(args[0], str):
        prefix = args[0]
    if "prefix" in kwargs and isinstance(kwargs["prefix"], str):
        prefix = kwargs["prefix"]

    # 2) keep/max_to_keep 호환
    keep_kw = kwargs.get("keep")
    mtk_kw = kwargs.get("max_to_keep")
    if isinstance(keep_kw, int):
        max_to_keep = keep_kw
    elif isinstance(mtk_kw, int):
        max_to_keep = mtk_kw
    else:
        max_to_keep = 3

    keys = list(backups.keys())
    if prefix:
        keys = [k for k in keys if k.startswith(prefix)]
    before = len(keys)

    if before <= max_to_keep:
        return (before, len(backups))

    to_delete = keys[:-max_to_keep]
    for k in to_delete:
        backups.pop(k, None)

    after = len(backups)
    return (before, after)

def export_brain_to_drive(
    creds: Dict[str, Any],
    persist_dir: str | Path,
    folder_id: str,
    filename: Optional[str] = None,
) -> Tuple[str, str]:
    """
    P0 스텁: 인덱스 디렉토리를 압축/업로드했다고 가정하고, (file_id, file_name)을 반환.
    실제 구현에서는 googleapiclient를 사용해 files().create(...) 호출.
    """
    ts_name = filename or f"{INDEX_BACKUP_PREFIX}.zip"
    # 실제 업로드 대신 성공 시그니처만 반환
    fake_file_id = "file_stub_id"
    return fake_file_id, ts_name

# ===== [10] 오케스트레이션: 인덱스 확보 =====================================
def get_or_build_index(
    update_pct: Optional[Callable[[int], None]] = None,
    update_msg: Optional[Callable[[str], None]] = None,
    gdrive_folder_id: Optional[str] = None,
    raw_sa: Optional[Any] = None,
    persist_dir: str | Path = PERSIST_DIR,
    manifest_path: Optional[str] = None,
    should_stop: Optional[Callable[[], bool]] = None,  # P0: 호출부와 시그니처 정합성
) -> Any:
    """
    - 로컬에 인덱스가 있으면 로드
    - 없으면 Drive에서 복구 시도 후 로드
    - P0: should_stop 콜백은 전달만 받고, 실제 사용은 선택적으로 수행
    """
    _emit(update_pct, update_msg, 2, "설정 확인 중…")

    # (옵션) 중단 신호 체크 — P0에서는 안전 호출만
    if should_stop:
        try:
            if should_stop():
                raise RAGEngineError("작업이 중단되었습니다. (should_stop)")
        except Exception:
            pass

    # 1) 로컬 먼저
    if _index_exists(persist_dir):
        _emit(update_pct, update_msg, 15, "로컬 인덱스 감지 → 로드합니다…")
        return _load_index_from_disk(persist_dir)

    # 2) Drive 복구
    _emit(update_pct, update_msg, 25, "로컬에 없어요. Drive에서 복구 시도…")
    if not gdrive_folder_id:
        raise FolderIdMissing(
            "Google Drive 폴더 ID가 비어 있습니다.",
            "settings.GDRIVE_FOLDER_ID/BACKUP_FOLDER_ID 확인",
        )

    creds = _validate_sa(_normalize_sa(raw_sa))
    ok, note = try_restore_index_from_drive(creds, persist_dir, gdrive_folder_id, update_msg=update_msg)
    if not ok:
        sa_email = creds.get("client_email", "(unknown)")
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

# ===== [11] 질의/답변 래퍼 ===================================================
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
        raise RAGEngineError(
            "답변 생성 중 오류가 발생했습니다. 우측 Traceback과 관리자 설정을 확인하세요.",
            debug=repr(e),
        )

# ===== [12] 디버그 헬퍼 ======================================================
def _format_debug(e: Exception) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))
