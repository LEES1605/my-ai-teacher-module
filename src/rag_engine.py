# src/rag_engine.py
from __future__ import annotations
import os, json, shutil
from typing import Callable, Any, Mapping

import streamlit as st
from src.config import settings

# === LLM/Embedding 설정(지연 초기화) ==========================================
def init_llama_settings(api_key: str, llm_model: str, embed_model: str, temperature: float = 0.0):
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
    from llama_index.core import StorageContext, load_index_from_storage
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index

def _build_drive_service(creds_dict: Mapping[str, Any]):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_info(dict(creds_dict), scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def _fetch_drive_manifest(creds_dict: Mapping[str, Any], folder_id: str) -> dict:
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

def _normalize_sa(raw_sa: Any | None) -> Mapping[str, Any] | None:
    """서비스 계정 JSON을 dict로 정규화 (Mapping/AttrDict/문자열/중첩 모두 허용)."""
    if raw_sa is None:
        return None

    # 1) 기본 파싱
    if isinstance(raw_sa, Mapping):
        d = dict(raw_sa)
    elif isinstance(raw_sa, str) and raw_sa.strip():
        try:
            d = json.loads(raw_sa)
        except Exception:
            return None
    else:
        return None

    # 2) 중첩 해제
    for k in ("service_account", "serviceAccount"):
        if isinstance(d.get(k), Mapping):
            d = dict(d[k])

    # 3) private_key 개행 복원
    pk = d.get("private_key")
    if isinstance(pk, str) and "\\n" in pk and "\n" not in pk:
        d["private_key"] = pk.replace("\\n", "\n")

    return d

def _validate_sa(creds: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """필수 키 누락 시 친절한 에러."""
    if not creds:
        st.error("GDRIVE 서비스계정 자격증명이 비어 있습니다. secrets에 JSON을 넣어주세요.")
        st.stop()

    required = {
        "type", "project_id", "private_key_id", "private_key",
        "client_email", "client_id", "token_uri",
    }
    missing = [k for k in required if k not in creds]
    if missing:
        st.error(
            "서비스계정 JSON에 필수 키가 누락되었습니다: "
            + ", ".join(missing)
            + "  (secrets에서 키 이름/철자와 값 형식을 확인하세요. "
              "private_key는 줄바꿈이 복원되어야 합니다.)"
        )
        with st.expander("진단 정보(보유 키 목록)"):
            st.write(sorted(list(creds.keys())))
        st.stop()

    return creds

def _build_index_with_progress(update_pct: Callable[[int, str | None], None],
                               update_msg: Callable[[str], None],
                               gdrive_folder_id: str,
                               gcp_creds: Mapping[str, Any],
                               persist_dir: str):
    """Drive → 문서 로드 → 인덱스 생성 → 디스크 저장"""
    from llama_index.core import VectorStoreIndex
    from llama_index.readers.google import GoogleDriveReader

    update_pct(15, "Drive 리더 초기화")
    # ✅ 최신 LlamaIndex는 service_account_key 인자를 사용함
    try:
        loader = GoogleDriveReader(service_account_key=gcp_creds)
    except Exception as e:
        st.error("GoogleDriveReader 초기화 실패: 서비스계정 키 형식 또는 라이브러리 버전을 확인하세요.")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

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
    update_pct(5, "드라이브 변경 확인 중…")
    gcp_creds = _validate_sa(_normalize_sa(raw_sa))

    remote = _fetch_drive_manifest(gcp_creds, gdrive_folder_id)

    local = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fp:
                local = json.load(fp)
        except Exception:
            local = None

    def _manifests_differ(local_m: dict | None, remote_m: dict) -> bool:
        if local_m is None:
            return True
        if set(local_m.keys()) != set(remote_m.keys()):
            return True
        for fid, r in remote_m.items():
            l = local_m.get(fid, {})
            if l.get("md5") and r.get("md5"):
                if l["md5"] != r["md5"]:
                    return True
            if l.get("modifiedTime") != r.get("modifiedTime"):
                return True
        return False

    if os.path.exists(persist_dir) and not _manifests_differ(local, remote):
        update_pct(25, "변경 없음 → 저장된 두뇌 로딩")
        from llama_index.core import StorageContext, load_index_from_storage
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        idx = load_index_from_storage(storage_context)
        update_pct(100, "완료!")
        return idx

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    update_pct(40, "변경 감지 → 문서 로드/인덱스 생성")
    idx = _build_index_with_progress(update_pct, update_msg, gdrive_folder_id, gcp_creds, persist_dir)

    # 새 매니페스트 저장
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(remote, fp, ensure_ascii=False, indent=2, sort_keys=True)

    update_pct(100, "완료!")
    return idx

# === QA 유틸 ==================================================================
def get_text_answer(query_engine, question: str, system_prompt: str) -> str:
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
            files = [n.metadata.get("file_name", "알 수 없음") for n in getattr(response, "source_nodes", [])]
            source_files = ", ".join(sorted(list(set(files)))) if files else "출처 정보 없음"
        except Exception:
            source_files = "출처 정보 없음"
        return f"{answer_text}\n\n---\n*참고 자료: {source_files}*"
    except Exception as e:
        return f"텍스트 답변 생성 중 오류 발생: {e}"
