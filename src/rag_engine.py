# ===== [01] TOP ==============================================================
# RAG Engine — 에러메시지 사용자 친화화 + 단계별 진행상태 콜백
# - app.py에서 호출하는 모든 공개함수 호환:
#   init_llama_settings, get_or_build_index, _load_index_from_disk,
#   try_restore_index_from_drive, get_text_answer,
#   _normalize_sa, _validate_sa
#
# 주의: 외부 라이브러리는 지연(import)하고, 실패 시 사람이 읽기 쉬운 메시지로 변환

from __future__ import annotations
import os, json, time, traceback
from pathlib import Path
from typing import Callable, Any

# ===== [02] CONFIG BRIDGE ====================================================
# settings, PERSIST_DIR은 src.config에서 가져오되, 예외적으로 루트 폴백 제공
try:
    from src.config import settings, PERSIST_DIR
except Exception:
    from config import settings, PERSIST_DIR  # 폴백

# ===== [03] 사용자 친화 예외 정의 ============================================
class RAGEngineError(Exception):
    """사용자에게 그대로 보여도 되는 메시지 중심의 예외"""
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

# ===== [04] 콜백 유틸(진행률/메시지 안전 호출) ===============================
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
    """
    secrets의 값이 pydantic SecretStr, dict, JSON str 등 어떤 형태든
    'JSON 문자열'로 정상화해서 반환.
    """
    if raw_sa is None:
        raise SecretsMissing(
            "서비스 계정 키가 없습니다.",
            "GDRIVE_SERVICE_ACCOUNT_JSON이 None"
        )

    # SecretStr 지원
    try:
        from pydantic.types import SecretStr
        if isinstance(raw_sa, SecretStr):
            raw_sa = raw_sa.get_secret_value()
    except Exception:
        pass

    # dict -> json
    if isinstance(raw_sa, dict):
        return json.dumps(raw_sa, ensure_ascii=False)

    # str이면 그대로
    if isinstance(raw_sa, str):
        s = raw_sa.strip()
        if not s:
            raise SecretsMissing("서비스 계정 키가 비어 있습니다.")
        return s

    # 그 외 타입
    try:
        return str(raw_sa)
    except Exception:
        raise ServiceAccountInvalid("서비스 계정 키 형식을 알 수 없습니다.")

def _validate_sa(json_str: str) -> dict:
    """
    구글 라이브러리 없이도 기본 키 유효성만 검사.
    실제 Drive API 호출은 try_restore_index_from_drive 내부에서 처리(있다면).
    """
    try:
        data = json.loads(json_str)
    except Exception as e:
        raise ServiceAccountInvalid(
            "서비스 계정 키(JSON)를 파싱하지 못했습니다. 비공개키에 줄바꿈(\\n) 처리와 TOML 따옴표(''' ... ''')를 확인하세요.",
            debug=repr(e)
        )
    # 최소 필수키 점검
    needed = {"type","private_key","client_email"}
    if not needed.issubset(set(data.keys())):
        raise ServiceAccountInvalid(
            "서비스 계정 키에 필수 항목(type, private_key, client_email)이 없습니다."
        )
    return data

# ===== [06] LLM/임베딩 초기화 ===============================================
def init_llama_settings(api_key: str, llm_model: str, embed_model: str, temperature: float = 0.0):
    """
    llama_index(혹은 유사) 설정을 지연 로딩. 실패 시 LlamaInitFailed로 변환.
    """
    try:
        # ★ 프로젝트별 초기화 로직을 여기에 연결
        # 예: from llama_index.core import Settings
        #     from llama_index.llms.gemini import Gemini
        #     Settings.llm = Gemini(api_key=api_key, model=llm_model, temperature=temperature)
        #     Settings.embed_model = ...
        # 여기서는 라이브러리 미존재 환경도 고려해 '의미있는 검증'만 수행
        if not api_key:
            raise ValueError("Gemini API Key가 비어 있습니다.")
        if not llm_model:
            raise ValueError("LLM 모델명이 비어 있습니다.")
        if not embed_model:
            raise ValueError("임베딩 모델명이 비어 있습니다.")
        # 실제 프로젝트에서는 위에 주석 코드처럼 Settings 구성하세요.
        return True
    except Exception as e:
        raise LlamaInitFailed(
            "LLM/임베딩 초기화 중 오류가 발생했습니다. API Key/모델명을 확인하세요.",
            debug=repr(e)
        )

# ===== [07] 인덱스 로드/생성 =================================================
def _index_exists(persist_dir: str | Path) -> bool:
    p = Path(persist_dir)
    if not p.exists(): return False
    # 프로젝트마다 파일명이 다르므로 “폴더가 비어있지 않음” 기준으로 최소 검증
    try:
        return any(p.iterdir())
    except Exception:
        return False

def _load_index_from_disk(persist_dir: str | Path):
    """
    디스크에서 인덱스를 로드. 실패 시 IndexLoadFailed.
    실제 구현은 프로젝트의 인덱스 포맷에 맞게 import/로드 코드를 연결하세요.
    """
    try:
        if not _index_exists(persist_dir):
            raise LocalIndexMissing("로컬 인덱스가 없습니다. 먼저 복구/생성하세요.")
        # ★ 실제 로더 예시 (주석):
        # from llama_index.core import StorageContext, load_index_from_storage
        # sc = StorageContext.from_defaults(persist_dir=str(persist_dir))
        # return load_index_from_storage(sc)
        # 여기서는 “더미” 객체로 쿼리엔진 인터페이스만 흉내냄
        class _DummyIndex:
            def as_query_engine(self, **kw):
                class _QE:
                    def query(self, q):  # 프로젝트에선 실제 질의가 수행됨
                        return type("A", (), {"response": f"[더미응답] {q}"})
                return _QE()
        return _DummyIndex()
    except RAGEngineError:
        raise
    except Exception as e:
        raise IndexLoadFailed(
            "인덱스 로드 중 오류가 발생했습니다. 인덱스 폴더가 손상되었을 수 있습니다.",
            debug=repr(e)
        )

def get_or_build_index(
    update_pct: Callable[[int], None] | None = None,
    update_msg: Callable[[str], None] | None = None,
    gdrive_folder_id: str | None = None,
    raw_sa: Any | None = None,
    persist_dir: str | Path = PERSIST_DIR,
    manifest_path: str | None = None,
):
    """
    1) 로컬 인덱스 있으면 로드
    2) 없고 폴더ID/SA 있으면 Drive 복구 시도
    3) 그래도 없으면 새로 빌드(옵션)
    """
    _emit(update_pct, update_msg, 2, "설정 확인 중…")

    # Llama 설정은 app.py에서 이미 호출됨 (여기선 생략)

    # 1) 로컬
    if _index_exists(persist_dir):
        _emit(update_pct, update_msg, 15, "로컬 인덱스 감지 → 로드합니다…")
        return _load_index_from_disk(persist_dir)

    # 2) Drive 복구
    _emit(update_pct, update_msg, 25, "로컬에 없어요. Drive에서 복구 시도…")
    if not gdrive_folder_id:
        raise FolderIdMissing(
            "Google Drive 폴더 ID가 비어 있습니다.",
            "settings.GDRIVE_FOLDER_ID 또는 BACKUP_FOLDER_ID 확인"
        )
    creds = _validate_sa(_normalize_sa(raw_sa))
    ok = try_restore_index_from_drive(creds, persist_dir, gdrive_folder_id, update_msg=update_msg)
    if not ok:
        raise DriveRestoreFailed(
            "Drive에서 인덱스를 찾지 못했거나 권한이 없습니다.",
            "폴더 공유 대상에 서비스 계정 이메일 추가 + 폴더ID 오탈자 확인"
        )
    _emit(update_pct, update_msg, 65, "복구됨 → 로드합니다…")
    return _load_index_from_disk(persist_dir)

# ===== [08] Drive 복구(필요 시 구현) ========================================
def try_restore_index_from_drive(
    creds: dict,
    persist_dir: str | Path,
    folder_id: str,
    update_msg: Callable[[str], None] | None = None,
) -> bool:
    """
    실제 구현은 Google Drive API를 사용해 folder_id 하위의 인덱스 파일 묶음을
    persist_dir로 내려받아야 합니다.
    - 여기서는 의존성 없는 ‘자체 점검’ + 친절한 오류 가이드만 제공.
    - 프로젝트에 Drive 연동이 이미 있다면 이 함수 내용을 교체하세요.
    """
    try:
        # 최소 점검
        if not folder_id or not str(folder_id).strip():
            raise FolderIdMissing("폴더 ID가 비어 있습니다.")
        if "client_email" not in creds:
            raise ServiceAccountInvalid("서비스 계정 키에 client_email이 없습니다.")

        # 여기에 실제 다운로드 로직을 넣으세요.
        # 예시: pydrive2 또는 googleapiclient로 파일 리스트 → 다운로드
        # 여기서는 데모로 '폴더가 이미 마운트되어 있다' 가정 후 폴더만 생성
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        _emit(None, update_msg, msg="(데모) Drive 복구 함수가 호출되었습니다 — 실제 다운로드 로직을 연결하세요.")
        # 실제 복구 성공 여부 반환
        return _index_exists(persist_dir)
    except RAGEngineError:
        raise
    except Exception as e:
        raise DriveRestoreFailed(
            "Drive 복구 중 예기치 못한 오류가 발생했습니다.",
            debug=repr(e)
        )

# ===== [09] 질의/답변 ========================================================
def get_text_answer(query_engine: Any, prompt: str, sys_prompt: str) -> str:
    """
    프로젝트의 실제 query_engine을 그대로 사용.
    엔진이 없거나 실패하면 명확한 메시지로 변환.
    """
    if not query_engine:
        raise QueryEngineNotReady("질의 엔진이 준비되지 않았습니다. 먼저 ‘AI 두뇌 준비’를 실행하세요.")

    try:
        # 시스템 프롬프트를 전처리해서 엔진에 전달하는 방식은 프로젝트에 맞게 확장
        q = f"{sys_prompt}\n\n사용자: {prompt}"
        res = query_engine.query(q)  # llama_index 스타일
        # llama_index는 res.response 속성에 결과 문자열을 둠
        if hasattr(res, "response"): 
            return res.response
        return str(res)
    except RAGEngineError:
        raise
    except Exception as e:
        # 여기서의 에러는 주로 인덱스/LLM 설정 문제
        raise RAGEngineError(
            "답변 생성 중 오류가 발생했습니다. 우측 Traceback과 관리자 설정을 확인하세요.",
            debug=repr(e)
        )

# ===== [10] 디버그 헬퍼(선택) ===============================================
def _format_debug(e: Exception) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))

# ===== [11] 모듈 끝 ==========================================================
