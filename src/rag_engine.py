# src/rag_engine.py
from __future__ import annotations
import os, json, shutil, re
from typing import Callable, Any, Mapping

import streamlit as st
from src.config import settings

# === LLM/Embedding 설정(지연 초기화) ==========================================
def init_llama_settings(api_key: str, llm_model: str, embed_model: str, temperature: float = 0.0):
    """LlamaIndex Settings를 지연 초기화하고, 임베딩 스모크 테스트로 키/네트워크를 점검."""
    from llama_index.core import Settings
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    from llama_index.llms.google_genai import GoogleGenAI

    Settings.llm = GoogleGenAI(model=llm_model, api_key=api_key, temperature=temperature)
    Settings.embed_model = GoogleGenAIEmbedding(model_name=embed_model, api_key=api_key)

    try:
        _ = Settings.embed_model.get_text_embedding("ping")
    except Exception as e:
        st.error("임베딩 모델 점검 실패 — API 키/모델명/네트워크를 확인하세요.")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

# === 인덱스 로딩/빌드 ==========================================================
@st.cache_resource(show_spinner=False)
def _load_index_from_disk(persist_dir: str):
    """디스크에 저장된 인덱스를 읽어옵니다."""
    from llama_index.core import StorageContext, load_index_from_storage
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index

def _build_drive_service(creds_dict):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def _fetch_drive_manifest(creds_dict, folder_id: str) -> dict:
    """드라이브 폴더 내 파일들의 '스냅샷(매니페스트)'를 가져옵니다."""
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
    """md5 있으면 md5, 없으면 modifiedTime 비교."""
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

# ---- 핵심 수정: 서비스계정 JSON '자동 복구' 파서 ------------------------------
def _normalize_sa(raw_sa: Any | None) -> Mapping[str, Any] | None:
    """
    서비스계정 JSON을 dict로 정규화.
    - dict면 그대로 반환
    - str이면 json.loads 시도
    - private_key가 '실제 줄바꿈'으로 들어온 잘못된 JSON을 자동 보정하여 파싱
    """
    if raw_sa is None:
        return None
    if isinstance(raw_sa, Mapping):
        return dict(raw_sa)

    if isinstance(raw_sa, str):
        s = raw_sa.strip()
        if not s:
            return None

        # 1) 있는 그대로 JSON 파싱
        try:
            return json.loads(s)
        except Exception:
            pass

        # 2) private_key에 '실제 개행'이 들어간 경우 보정
        #    "private_key": "<여러 줄의 PEM>" 구간을 찾아 내부 개행을 \n 으로 이스케이프
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

def _build_index_with_progress(update_pct: Callable[[int, str | None], None],
                               update_msg: Callable[[str], None],
                               gdrive_folder_id: str,
                               gcp_creds: Mapping[str, Any],
                               persist_dir: str):
    """Drive → 문서 로드 → 인덱스 생성 → 디스크 저장"""
    from llama_index.core import VectorStoreIndex
    from llama_index.readers.google import GoogleDriveReader

    update_pct(5, "Google Drive 인증 준비")
    if not gcp_creds:
        st.error("❌ 서비스계정 JSON을 읽을 수 없습니다.")
        with st.expander("문제 해결 가이드", expanded=True):
            st.markdown(
                "- **.streamlit/secrets.toml**의 `GDRIVE_SERVICE_ACCOUNT_JSON`이 유효한 JSON인지 확인하세요.\n"
                "  - 특히 `private_key`는 실제 줄바꿈이 아니라 **`\\\\n` 이스케이프**가 되어 있어야 합니다.\n"
                "  - 예시:\n"
                '```toml\nGDRIVE_SERVICE_ACCOUNT_JSON = """{ "type":"service_account", ..., "private_key":"-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n", ... }"""\n```'
            )
        st.stop()

        update_pct(15, "Drive 리더 초기화")
    # 버전 호환: 신형은 service_account_key, 구형은 gcp_creds_dict
    try:
        loader = GoogleDriveReader(service_account_key=gcp_creds)   # ✅ 신형 API
    except TypeError:
        loader = GoogleDriveReader(gcp_creds_dict=gcp_creds)        # ↩️ 구형 API 호환

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
                       manifest_path: str):
    """Drive 변경을 감지해 저장본을 쓰거나, 변경 시에만 재인덱싱."""
    gcp_creds = _normalize_sa(raw_sa)
    if not gcp_creds:
        st.error("서비스계정 JSON 파싱에 실패했습니다.")
        with st.expander("도움말: 올바른 입력 예", expanded=True):
            st.code(
                'GDRIVE_SERVICE_ACCOUNT_JSON = """{ "type":"service_account", ..., '
                '"private_key":"-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n", ... }"""',
                language="toml",
            )
        st.stop()

    update_pct(5, "드라이브 변경 확인 중…")
    remote = _fetch_drive_manifest(gcp_creds, gdrive_folder_id)
    local = _load_local_manifest(manifest_path)

    if os.path.exists(persist_dir) and not _manifests_differ(local, remote):
        update_pct(25, "변경 없음 → 저장된 두뇌 로딩")
        idx = _load_index_from_disk(persist_dir)
        update_pct(100, "완료!")
        return idx

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    update_pct(40, "변경 감지 → 문서 로드/인덱스 생성")
    idx = _build_index_with_progress(update_pct, update_msg, gdrive_folder_id, gcp_creds, persist_dir)

    _save_local_manifest(manifest_path, remote)
    update_pct(100, "완료!")
    return idx

# === QA 유틸 ==================================================================
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
            files = [n.metadata.get('file_name', '알 수 없음') for n in getattr(response, "source_nodes", [])]
            source_files = ", ".join(sorted(list(set(files)))) if files else "출처 정보 없음"
        except Exception:
            source_files = "출처 정보 없음"

        return f"{answer_text}\n\n---\n*참고 자료: {source_files}*"
    except Exception as e:
        return f"텍스트 답변 생성 중 오류 발생: {e}"
