# src/rag_engine.py
from __future__ import annotations
import os, json, time
from typing import Callable, Any, Mapping

import streamlit as st
from src.config import settings

# --------------------------------------------------------------------
# 0) 공통: 서비스계정 JSON 파싱 (str JSON / dict 모두 허용)
# --------------------------------------------------------------------
def _parse_service_account(raw_sa: Any | None) -> Mapping[str, Any] | None:
    if raw_sa is None:
        return None
    if isinstance(raw_sa, Mapping):
        return raw_sa
    if isinstance(raw_sa, str) and raw_sa.strip():
        try:
            return json.loads(raw_sa)
        except Exception:
            return None
    return None

# --------------------------------------------------------------------
# 1) Google Drive 스모크 테스트 & 미리보기
# --------------------------------------------------------------------
def _build_drive_service(creds_dict):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    # cache_discovery=False: Streamlit Cloud의 파일시스템에서 캐시 충돌 방지
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def smoke_test_drive() -> tuple[bool, str]:
    """
    서비스계정 JSON & 폴더 ID 유무를 확인하고, 간단 호출이 가능한지 점검.
    UI에서는 (ok, msg)로 받아 success/warning 출력.
    """
    raw_sa = settings.GDRIVE_SERVICE_ACCOUNT_JSON
    folder_id = settings.GDRIVE_FOLDER_ID
    if not str(folder_id).strip():
        return (False, "GDRIVE_FOLDER_ID가 비었습니다. Secrets에 값을 추가하세요.")
    sa = _parse_service_account(raw_sa)
    if not sa:
        return (False, "GOOGLE_SERVICE_ACCOUNT_JSON(서비스계정)이 비었습니다.")
    try:
        svc = _build_drive_service(sa)
        # 호출 가능여부만 빠르게 확인
        svc.files().get(fileId=folder_id, fields="id").execute()
        return (True, "Google Drive 연결 OK")
    except Exception as e:
        return (False, f"Drive 연결 점검 실패: {e}")

def preview_drive_files(max_items: int = 10) -> tuple[bool, str, list[dict]]:
    """
    지정 폴더의 최신 파일 목록 몇 개를 테이블로 보여주기 위한 헬퍼.
    반환 rows 예시: [{"name":..., "link":..., "mime":..., "modified":...}, ...]
    """
    raw_sa = settings.GDRIVE_SERVICE_ACCOUNT_JSON
    folder_id = settings.GDRIVE_FOLDER_ID
    sa = _parse_service_account(raw_sa)
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
        rows = []
        for f in files:
            rows.append({
                "name": f.get("name"),
                "link": f"https://drive.google.com/file/d/{f.get('id')}/view",
                "mime": f.get("mimeType"),
                "modified": f.get("modifiedTime"),
            })
        return (True, f"{len(rows)}개 파일", rows)
    except Exception as e:
        return (False, f"목록 조회 실패: {e}", [])

# --------------------------------------------------------------------
# 2) LLM/Embedding 설정 (지연 초기화)
# --------------------------------------------------------------------
def init_llama_settings(api_key: str, llm_model: str, embed_model: str, temperature: float = 0.0):
    """
    LlamaIndex 전역 Settings에 Google GenAI LLM/임베딩을 설정.
    임베딩 1회 호출로 키/네트워크 스모크 테스트.
    """
    from llama_index.core import Settings
    from llama_index.llms.google_genai import GoogleGenAI
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

    Settings.llm = GoogleGenAI(model=llm_model, api_key=api_key, temperature=temperature)
    Settings.embed_model = GoogleGenAIEmbedding(model_name=embed_model, api_key=api_key)

    try:
        _ = Settings.embed_model.get_text_embedding("ping")
    except Exception as e:
        st.error("임베딩 모델 점검 실패 — API 키/모델명/네트워크를 확인하세요.")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

# --------------------------------------------------------------------
# 3) 인덱스 로딩/빌드 & 변경 감지(매니페스트)
# --------------------------------------------------------------------
def _load_index_from_disk(persist_dir: str):
    from llama_index.core import StorageContext, load_index_from_storage
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    return load_index_from_storage(storage_context)

def _save_index_to_disk(index, persist_dir: str):
    try:
        index.storage_context.persist(persist_dir=persist_dir)
    except Exception as e:
        st.error("인덱스 저장 중 오류가 발생했습니다.")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

def _load_local_manifest(path: str) -> dict | None:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            return None
    return None

def _save_local_manifest(path: str, data: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"로컬 매니페스트 저장 실패: {e}")

def _fetch_drive_manifest(creds_dict, folder_id: str) -> dict:
    """
    Drive 폴더의 파일 스냅샷(id/name/mime/modified/md5/size)을 dict로 반환.
    변경 감지에 사용.
    """
    svc = _build_drive_service(creds_dict)
    files = []
    page_token = None
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

def _manifests_differ(local: dict | None, remote: dict) -> bool:
    """간단 비교: 로컬이 없거나, 파일 개수/수정시각/해시가 다르면 변경으로 간주."""
    if local is None:
        return True
    if set(local.keys()) != set(remote.keys()):
        return True
    for fid, r in remote.items():
        l = local.get(fid, {})
        if l.get("modifiedTime") != r.get("modifiedTime"):
            return True
        # md5가 있는 파일이면 md5도 비교
        if r.get("md5") and l.get("md5") != r.get("md5"):
            return True
    return False

def _build_index_with_progress(update_pct: Callable[[int, str | None], None],
                               update_msg: Callable[[str], None],
                               gdrive_folder_id: str,
                               gcp_creds: Mapping[str, Any],
                               persist_dir: str):
    """
    Drive → 문서 로드 → 인덱스 생성 → 디스크 저장
    """
    from llama_index.core import VectorStoreIndex
    from llama_index.readers.google import GoogleDriveReader

    update_pct(5, "Google Drive 인증 준비")
    if not gcp_creds:
        st.error("❌ GDRIVE_SERVICE_ACCOUNT_JSON이 비었습니다.")
        st.stop()

    update_pct(15, "Drive 리더 초기화")
    loader = GoogleDriveReader(gcp_creds_dict=gcp_creds)

    update_pct(30, "문서 목록 불러오는 중")
    try:
        documents = loader.load_data(folder_id=gdrive_folder_id)
    except Exception as e:
        st.error("Google Drive에서 문서를 불러오는 중 오류가 발생했습니다.")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

    if not documents:
        st.error("강의 자료 폴더가 비었거나 권한 문제입니다. folder_id/공유권한을 확인하세요.")
        st.stop()

    update_pct(60, f"문서 {len(documents)}개 로드 → 인덱스 생성")
    try:
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
    except Exception as e:
        st.error("인덱스 생성 중 오류가 발생했습니다.")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

    update_pct(85, "인덱스 저장 중")
    _save_index_to_disk(index, persist_dir)

    update_pct(100, "완료")
    return index

def get_or_build_index(update_pct: Callable[[int, str | None], None],
                       update_msg: Callable[[str], None],
                       gdrive_folder_id: str,
                       raw_sa: Any | None,
                       persist_dir: str,
                       manifest_path: str):
    """
    1) Drive 매니페스트 가져와 변경 여부 판단
    2) 변경 없으면 디스크에서 인덱스 로드
    3) 변경 있으면 새로 빌드 후 저장 & 매니페스트 갱신
    """
    gcp_creds = _parse_service_account(raw_sa)
    if not gcp_creds:
        st.error("서비스계정 JSON이 비었습니다.")
        st.stop()

    update_pct(10, "Drive 매니페스트 확인")
    remote = _fetch_drive_manifest(gcp_creds, gdrive_folder_id)
    local = _load_local_manifest(manifest_path)

    if os.path.exists(persist_dir) and not _manifests_differ(local, remote):
        update_pct(40, "변경 없음 → 저장소에서 불러오기")
        try:
            idx = _load_index_from_disk(persist_dir)
            update_pct(100, "완료!")
            return idx
        except Exception:
            # 저장소가 망가진 경우 재빌드 시도
            update_msg("저장된 인덱스가 손상되어 재빌드합니다…")

    update_pct(40, "변경 감지 → 문서 로드/인덱스 생성")
    idx = _build_index_with_progress(update_pct, update_msg, gdrive_folder_id, gcp_creds, persist_dir)

    _save_local_manifest(manifest_path, remote)
    update_pct(100, "완료!")
    return idx

# --------------------------------------------------------------------
# 4) QA 유틸
# --------------------------------------------------------------------
def get_text_answer(query_engine, question: str, system_prompt: str) -> str:
    """
    선택된 페르소나 지침 + 사용자의 질문을 합쳐 쿼리하고, 출처 파일명을 함께 반환.
    """
    try:
        full_query = (
            f"{system_prompt}\n\n"
            "[지시사항] 반드시 업로드된 강의 자료를 최우선으로 참고하고, "
            "근거를 찾을 수 없다면 그 사실을 명확히 밝혀라.\n\n"
            f"[학생의 질문]\n{question}"
        )
        response = query_engine.query(full_query)
        answer_text = str(response)

        # 출처 파일명 모으기
        try:
            files = [n.metadata.get("file_name", "알 수 없음")
                     for n in getattr(response, "source_nodes", [])]
            source_files = ", ".join(sorted(set(files))) if files else "출처 정보 없음"
        except Exception:
            source_files = "출처 정보 없음"

        return f"{answer_text}\n\n---\n*참고 자료: {source_files}*"
    except Exception as e:
        return f"텍스트 답변 생성 중 오류 발생: {e}"
