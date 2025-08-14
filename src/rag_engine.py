# src/rag_engine.py — RAG 유틸(임베딩 1회 + LLM 2개) + 취소(캔슬) 지원
#                     + tqdm 콘솔 진행바 억제(연결 리셋/로그 스팸 방지)

from __future__ import annotations
import os, json, shutil, re
from typing import Callable, Any, Mapping

# 🔇 tqdm(콘솔 진행바) 억제 — Streamlit Cloud 로그 스팸/워커부하 완화
os.environ.setdefault("TQDM_DISABLE", "1")

import streamlit as st
from src.config import settings

# (선택) llama_index 로그 억제 — 과도한 디버그 출력 방지
import logging
logging.getLogger("llama_index").setLevel(logging.WARNING)

# 취소 신호용 예외
class CancelledError(Exception):
    """사용자가 '취소' 버튼을 눌러 실행을 중단했음을 나타냅니다."""
    pass

# ============================================================================
# 0) 공통: 서비스계정 JSON 정규화 + Drive 서비스
# ============================================================================

def _normalize_sa(raw_sa: Any | None) -> Mapping[str, Any] | None:
    """서비스계정 JSON을 dict로 정규화(+private_key 개행 보정)"""
    if raw_sa is None:
        return None
    if isinstance(raw_sa, Mapping):
        return dict(raw_sa)

    if isinstance(raw_sa, str):
        s = raw_sa.strip()
        if not s:
            return None
        # 1) 정상 JSON 시도
        try:
            return json.loads(s)
        except Exception:
            pass
        # 2) private_key 개행 보정
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

def _build_drive_service(creds_dict):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

# ============================================================================
# 1) 임베딩/LLM 초기화 (임베딩 1회, LLM은 필요 수만큼)
# ============================================================================

def set_embed_provider(provider: str, api_key: str, embed_model: str):
    """임베딩 공급자만 지정해 Settings.embed_model 설정"""
    from llama_index.core import Settings
    p = (provider or "google").lower()
    try:
        if p == "openai":
            from llama_index.embeddings.openai import OpenAIEmbedding as _EMB
            Settings.embed_model = _EMB(model=embed_model, api_key=api_key)
        else:
            from llama_index.embeddings.google_genai import GoogleGenAIEmbedding as _EMB
            Settings.embed_model = _EMB(model_name=embed_model, api_key=api_key)

        # 스모크 테스트
        _ = Settings.embed_model.get_text_embedding("ping")
    except Exception as e:
        st.error("임베딩 초기화 실패 — 키/모델/패키지 설치를 확인하세요.")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

def make_llm(provider: str, api_key: str, llm_model: str, temperature: float = 0.0):
    """공급자의 LLM 인스턴스만 만들어 반환(Settings는 건드리지 않음)"""
    p = (provider or "google").lower()
    try:
        if p == "openai":
            from llama_index.llms.openai import OpenAI as _LLM
            return _LLM(model=llm_model, api_key=api_key, temperature=temperature)
        else:
            from llama_index.llms.google_genai import GoogleGenAI as _LLM
            return _LLM(model=llm_model, api_key=api_key, temperature=temperature)
    except Exception as e:
        st.error(f"{provider} LLM 초기화 실패 — 키/모델을 확인하세요.")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

# ============================================================================
# 2) Drive 테스트/미리보기
# ============================================================================

def smoke_test_drive() -> tuple[bool, str]:
    folder_id = settings.GDRIVE_FOLDER_ID
    if not str(folder_id).strip():
        return (False, "GDRIVE_FOLDER_ID가 비었습니다. Secrets에 값을 추가하세요.")

    sa = _normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON)
    if not sa:
        return (False, "GOOGLE_SERVICE_ACCOUNT_JSON(서비스계정) 파싱 실패")

    try:
        svc = _build_drive_service(sa)
        svc.files().get(fileId=folder_id, fields="id").execute()
        return (True, "Google Drive 연결 OK")
    except Exception as e:
        return (False, f"Drive 연결 점검 실패: {e}")

def preview_drive_files(max_items: int = 10) -> tuple[bool, str, list[dict]]:
    folder_id = settings.GDRIVE_FOLDER_ID
    sa = _normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON)
    if not sa or not str(folder_id).strip():
        return (False, "서비스계정/폴더 ID 설정이 부족합니다.", [])

    try:
        svc = _build_drive_service(sa)
        q = f"'{folder_id}' in parents and trashed=false"
        fields = "files(id,name,mimeType,modifiedTime), nextPageToken"
        resp = svc.files().list(
            q=q,
            orderBy="modifiedTime desc",
            pageSize=max_items,
            fields=fields,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files = resp.get("files", [])
        rows = [{
            "name": f.get("name"),
            "link": f"https://drive.google.com/file/d/{f.get('id')}/view",
            "mime": f.get("mimeType"),
            "modified": f.get("modifiedTime"),
        } for f in files]
        return (True, f"{len(rows)}개 파일", rows)
    except Exception as e:
        return (False, f"목록 조회 실패: {e}", [])

# ============================================================================
# 3) 인덱스 로딩/빌드 & 변경 감지(매니페스트) + 취소 지원
# ============================================================================

def _fetch_drive_manifest(creds_dict, folder_id: str, is_cancelled: Callable[[], bool] | None = None) -> dict:
    svc = _build_drive_service(creds_dict)
    files = []
    page_token = None
    q = f"'{folder_id}' in parents and trashed=false"
    fields = "nextPageToken, files(id,name,mimeType,modifiedTime,md5Checksum,size)"
    while True:
        if is_cancelled and is_cancelled():
            raise CancelledError("사용자 취소(매니페스트 조회 중)")
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

def _load_local_manifest(path: str) -> dict | None:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            return None
    return None

def _save_local_manifest(path: str, m: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(m, fp, ensure_ascii=False, indent=2, sort_keys=True)

def _manifests_differ(local: dict | None, remote: dict) -> bool:
    if local is None:
        return True
    if set(local.keys()) != set(remote.keys()):
        return True
    for fid, r in remote.items():
        l = local.get(fid, {})
        if l.get("md5") and r.get("md5"):
            if l["md5"] != r["md5"]:
                return True
        if l.get("modifiedTime") != r.get("modifiedTime"):
            return True
    return False

@st.cache_resource(show_spinner=False)
def _load_index_from_disk(persist_dir: str):
    from llama_index.core import StorageContext, load_index_from_storage
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    return load_index_from_storage(storage_context)

def _build_index_with_progress(update_pct: Callable[[int, str | None], None],
                               update_msg: Callable[[str], None],
                               gdrive_folder_id: str,
                               gcp_creds: Mapping[str, Any],
                               persist_dir: str,
                               max_docs: int | None = None,
                               is_cancelled: Callable[[], bool] | None = None):
    """인덱스 신규 생성(취소 체크 지원)"""
    from llama_index.core import VectorStoreIndex
    from llama_index.readers.google import GoogleDriveReader

    if is_cancelled and is_cancelled():
        raise CancelledError("사용자 취소(초기 단계)")

    update_pct(5, "Google Drive 인증 준비")
    if not gcp_creds:
        st.error("❌ 서비스계정 JSON을 읽을 수 없습니다.")
        st.stop()

    if is_cancelled and is_cancelled():
        raise CancelledError("사용자 취소(리더 초기화 전)")

    update_pct(15, "Drive 리더 초기화")
    try:
        try:
            loader = GoogleDriveReader(service_account_key=gcp_creds)   # 신형
        except TypeError:
            loader = GoogleDriveReader(gcp_creds_dict=gcp_creds)        # 구형
    except Exception as e:
        st.error("Google Drive 리더 초기화 실패")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

    if is_cancelled and is_cancelled():
        raise CancelledError("사용자 취소(문서 로드 전)")

    update_pct(30, "문서 로드 중…")
    try:
        documents = loader.load_data(folder_id=gdrive_folder_id)
    except Exception as e:
        st.error("Google Drive에서 문서를 불러오는 중 오류가 발생했습니다.")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

    if is_cancelled and is_cancelled():
        raise CancelledError("사용자 취소(문서 로드 후)")

    # 빠른 모드: 개수 제한
    if max_docs and len(documents) > max_docs:
        documents = documents[:max_docs]
        update_msg(f"빠른 모드: 처음 {max_docs}개 문서만 인덱싱")

    update_pct(60, f"문서 {len(documents)}개 → 인덱스 생성")
    if is_cancelled and is_cancelled():
        raise CancelledError("사용자 취소(인덱스 생성 전)")

    try:
        # ✅ tqdm 콘솔 진행바 끄기: show_progress=False
        index = VectorStoreIndex.from_documents(documents, show_progress=False)
    except Exception as e:
        st.error("인덱스 생성 중 오류가 발생했습니다.")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

    if is_cancelled and is_cancelled():
        raise CancelledError("사용자 취소(인덱스 생성 후)")

    update_pct(90, "두뇌 저장 중")
    try:
        index.storage_context.persist(persist_dir=persist_dir)
    except Exception as e:
        st.error("인덱스 저장 중 오류가 발생했습니다.")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

    update_pct(100, "완료")
    return index

def get_or_build_index(update_pct: Callable[[int, str | None], None],
                       update_msg: Callable[[str], None],
                       gdrive_folder_id: str,
                       raw_sa: Any | None,
                       persist_dir: str,
                       manifest_path: str,
                       max_docs: int | None = None,
                       is_cancelled: Callable[[], bool] | None = None):
    """변경 없으면 로딩, 있으면 빌드(취소 체크 지원)"""
    gcp_creds = _normalize_sa(raw_sa)
    if not gcp_creds:
        st.error("서비스계정 JSON 파싱에 실패했습니다.")
        st.stop()

    update_pct(5, "드라이브 변경 확인 중…")
    remote = _fetch_drive_manifest(gcp_creds, gdrive_folder_id, is_cancelled=is_cancelled)
    local = _load_local_manifest(manifest_path)

    if is_cancelled and is_cancelled():
        raise CancelledError("사용자 취소(변경 확인 단계)")

    if os.path.exists(persist_dir) and not _manifests_differ(local, remote):
        update_pct(25, "변경 없음 → 저장된 두뇌 로딩")
        idx = _load_index_from_disk(persist_dir)
        update_pct(100, "완료!")
        return idx

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    update_pct(40, "변경 감지 → 인덱스 생성")
    idx = _build_index_with_progress(
        update_pct, update_msg, gdrive_folder_id, gcp_creds, persist_dir,
        max_docs=max_docs, is_cancelled=is_cancelled
    )

    _save_local_manifest(manifest_path, remote)
    update_pct(100, "완료!")
    return idx

# ============================================================================
# 4) QA 유틸
# ============================================================================

def get_text_answer(query_engine, question: str, system_prompt: str) -> str:
    """선택된 페르소나 지침 + 사용자의 질문을 합쳐 쿼리하고, 출처 파일명을 함께 반환."""
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
            files = [n.metadata.get("file_name", "알 수 없음")
                     for n in getattr(response, "source_nodes", [])]
            source_files = ", ".join(sorted(set(files))) if files else "출처 정보 없음"
        except Exception:
            source_files = "출처 정보 없음"

        return f"{answer_text}\n\n---\n*참고 자료: {source_files}*"
    except Exception as e:
        return f"텍스트 답변 생성 중 오류 발생: {e}"
