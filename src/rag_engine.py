# src/rag_engine.py
from __future__ import annotations
import os, json, shutil, time
from typing import Callable, Any, Mapping, Iterable

import streamlit as st

from src.config import settings

# -----------------------------------------------------------------------------
# 예외 정의
# -----------------------------------------------------------------------------
class CancelledError(Exception):
    pass


# -----------------------------------------------------------------------------
# LlamaIndex Settings: 임베딩/LLM 공급자 설정
# -----------------------------------------------------------------------------
def set_embed_provider(provider: str, api_key: str, model: str) -> None:
    """
    LlamaIndex 전역 Settings.embed_model을 설정.
    provider: "google" | "openai"
    """
    from llama_index.core import Settings
    if provider == "google":
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        Settings.embed_model = GoogleGenAIEmbedding(model=model, api_key=api_key)
    elif provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        # OPENAI_API_KEY 환경변수로도 동작하지만, 여기선 명시 전달
        Settings.embed_model = OpenAIEmbedding(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unknown embed provider: {provider}")


def make_llm(provider: str, api_key: str, model: str, temperature: float = 0.0):
    """
    provider: "google" | "openai"
    반환: LlamaIndex 호환 LLM 객체
    """
    if provider == "google":
        from llama_index.llms.google_genai import GoogleGenAI
        return GoogleGenAI(api_key=api_key, model=model, temperature=temperature)
    elif provider == "openai":
        from llama_index.llms.openai import OpenAI
        return OpenAI(api_key=api_key, model=model, temperature=temperature)
    else:
        raise ValueError(f"Unknown llm provider: {provider}")


# === LLM 무검색(직접 완성) 유틸 ===============================================
def llm_complete(llm, prompt: str, temperature: float = 0.0) -> str:
    """RAG 검색 없이 순수 LLM으로만 결과를 생성."""
    try:
        resp = llm.complete(prompt)
        return getattr(resp, "text", str(resp))
    except AttributeError:
        return llm.predict(prompt)


# -----------------------------------------------------------------------------
# Drive API 도우미 (Shared Drive 대응)
# -----------------------------------------------------------------------------
def _normalize_sa(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, str):
        return json.loads(raw)
    elif isinstance(raw, Mapping):
        return raw
    else:
        return {}


def _build_drive_service(creds_dict):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _fetch_drive_manifest(creds_dict, root_folder_id: str) -> dict:
    """공유드라이브 포함, 폴더 전체(하위 폴더 재귀) 파일 스냅샷을 만든다."""
    svc = _build_drive_service(creds_dict)

    def list_children(folder_id: str) -> tuple[list[dict], list[dict]]:
        files, folders = [], []
        q = (
            f"'{folder_id}' in parents and trashed=false"
        )
        page_token = None
        while True:
            res = svc.files().list(
                q=q,
                fields="nextPageToken, files(id,name,mimeType,modifiedTime,size,md5Checksum)",
                pageSize=200,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                pageToken=page_token,
            ).execute()
            for f in res.get("files", []):
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    folders.append(f)
                else:
                    files.append(f)
            page_token = res.get("nextPageToken")
            if not page_token:
                break
        return files, folders

    all_files: list[dict] = []
    queue = [root_folder_id]
    seen = set()
    while queue:
        fid = queue.pop(0)
        if fid in seen:
            continue
        seen.add(fid)
        files, folders = list_children(fid)
        all_files.extend(files)
        queue.extend([f["id"] for f in folders])

    # 간단한 정규화
    for f in all_files:
        f.setdefault("size", "0")
        f.setdefault("md5Checksum", "")
    return {"root": root_folder_id, "files": all_files, "count": len(all_files)}


def _load_local_manifest(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"files": []}


def _save_local_manifest(path: str, manifest: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _manifests_differ(local: dict, remote: dict) -> bool:
    def sig(d: dict) -> tuple:
        return (
            d.get("id", ""),
            d.get("modifiedTime", ""),
            d.get("size", ""),
            d.get("md5Checksum", ""),
        )
    ls = sorted([sig(x) for x in local.get("files", [])])
    rs = sorted([sig(x) for x in remote.get("files", [])])
    return ls != rs


# -----------------------------------------------------------------------------
# 인덱스 저장/로드
# -----------------------------------------------------------------------------
def _load_index_from_disk(persist_dir: str):
    from llama_index.core import StorageContext, load_index_from_storage
    return load_index_from_storage(StorageContext.from_defaults(persist_dir=persist_dir))


def _persist_index(index, persist_dir: str) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_dir)


# -----------------------------------------------------------------------------
# 인덱스 빌드(파일 단위 로깅 업그레이드)
# -----------------------------------------------------------------------------
def _build_index_with_progress(
    update_pct: Callable[[int, str | None], None],
    update_msg: Callable[[str], None],
    gdrive_folder_id: str,
    gcp_creds: Mapping[str, Any],
    persist_dir: str,
    max_docs: int | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Any:
    """
    Google Drive → 파일 단위 로드(try/except) → 실패 로깅 → VectorStoreIndex 생성 → 저장
    """
    from llama_index.core import VectorStoreIndex, Document
    from llama_index.readers.google import GoogleDriveReader

    if not gcp_creds:
        st.error("❌ GDRIVE_SERVICE_ACCOUNT_JSON이 비었습니다.")
        st.stop()

    # 1) 매니페스트 구성
    update_pct(10, "Drive 파일 목록 불러오는 중…")
    manifest = _fetch_drive_manifest(gcp_creds, gdrive_folder_id)
    files = manifest.get("files", [])
    total = len(files)
    if total == 0:
        update_pct(100, "폴더에 학습할 파일이 없습니다.")
        # 빈 인덱스라도 반환
        return VectorStoreIndex.from_documents([])

    # fast 모드: N개만
    if max_docs is not None and max_docs > 0:
        files = files[:max_docs]
        total = len(files)

    # 2) 파일 단위 로드 (예외 로깅)
    update_pct(20, f"문서 로드 중… (총 {total}개)")
    reader = GoogleDriveReader(service_account_key=gcp_creds, recursive=False)
    loaded_docs: list[Document] = []
    skipped: list[dict] = []

    for i, f in enumerate(files, start=1):
        if is_cancelled and is_cancelled():
            raise CancelledError("사용자 취소(진행 중)")
        fid = f["id"]
        name = f.get("name", "")
        mime = f.get("mimeType", "")
        try:
            # 개별 파일만 로딩
            docs = reader.load_data(file_ids=[fid])
            # 일부 파일은 여러 Document로 나뉠 수 있음
            for d in docs:
                # 원본 파일명 보존(후처리 근거용)
                try:
                    d.metadata["file_name"] = name
                except Exception:
                    pass
            loaded_docs.extend(docs)
            # 진행률
            pct = 20 + int(70 * i / max(total, 1))  # 20→90% 구간 사용
            update_pct(pct, f"로딩 {i}/{total} — {name}")
        except Exception as e:
            skipped.append({"id": fid, "name": name, "mime": mime, "error": str(e)})
            update_msg(f"⚠️ 스킵: {name} ({mime}) — {e}")

    # 3) 인덱스 생성
    if not loaded_docs:
        update_pct(100, "로드된 문서가 없어 빈 인덱스를 반환합니다.")
        index = VectorStoreIndex.from_documents([])
    else:
        update_pct(92, f"인덱스 생성 중… (문서 {len(loaded_docs)}개)")
        index = VectorStoreIndex.from_documents(loaded_docs)

    # 4) 저장 + 보고서 세션에 기록
    update_pct(95, "인덱스 저장 중…")
    _persist_index(index, persist_dir)
    update_pct(100, "완료")

    report = {
        "total_manifest": total,
        "loaded_docs": len(loaded_docs),
        "skipped_count": len(skipped),
        "skipped": skipped,            # [{id,name,mime,error}, ...]
    }
    st.session_state["indexing_report"] = report
    return index


# -----------------------------------------------------------------------------
# 변경 감지 후 로드/재빌드
# -----------------------------------------------------------------------------
def get_or_build_index(
    update_pct: Callable[[int, str | None], None],
    update_msg: Callable[[str], None],
    gdrive_folder_id: str,
    raw_sa: Any | None,
    persist_dir: str,
    manifest_path: str,
    max_docs: int | None = None,
    is_cancelled: Callable[[], bool] | None = None,
):
    """Drive 변경을 감지해 저장본을 쓰거나, 변경 시에만 재인덱싱."""
    gcp_creds = _normalize_sa(raw_sa)

    update_pct(5, "드라이브 변경 확인 중…")
    remote = _fetch_drive_manifest(gcp_creds, gdrive_folder_id)
    local = _load_local_manifest(manifest_path)

    if os.path.exists(persist_dir) and not _manifests_differ(local, remote):
        update_pct(25, "변경 없음 → 저장된 두뇌 로딩")
        idx = _load_index_from_disk(persist_dir)
        update_pct(100, "완료!")
        # 기존 리포트 보존(없으면 기본값)
        st.session_state.setdefault("indexing_report", {"total_manifest": len(remote.get("files", [])),
                                                        "loaded_docs": -1, "skipped_count": 0, "skipped": []})
        return idx

    # 변경 감지: 재빌드
    try:
        # 이전 저장물 제거(깨끗하게)
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir, ignore_errors=True)
    except Exception:
        pass

    # 실제 빌드
    index = _build_index_with_progress(
        update_pct=update_pct,
        update_msg=update_msg,
        gdrive_folder_id=gdrive_folder_id,
        gcp_creds=gcp_creds,
        persist_dir=persist_dir,
        max_docs=max_docs,
        is_cancelled=is_cancelled,
    )

    # 새 매니페스트 저장
    _save_local_manifest(manifest_path, remote)
    return index


# -----------------------------------------------------------------------------
# 텍스트 답변(출처 표시)
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Drive 연결 스모크/미리보기 (앱의 테스트 UI에서 사용)
# -----------------------------------------------------------------------------
def smoke_test_drive() -> tuple[bool, str]:
    try:
        sa = _normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON)
        svc = _build_drive_service(sa)
        fid = settings.GDRIVE_FOLDER_ID
        meta = svc.files().get(fileId=fid, fields="id,name,driveId,parents", supportsAllDrives=True).execute()
        name = meta.get("name", "")
        drive_id = meta.get("driveId", "")
        return True, f"✅ Drive 연결 OK · 폴더명: {name} · driveId: {drive_id or 'MyDrive'}"
    except Exception as e:
        return False, f"Drive 연결/권한 확인 실패: {e}"


def preview_drive_files(max_items: int = 10) -> tuple[bool, str, list[dict]]:
    """상위 폴더의 최신 파일을 미리본다(하위 폴더 포함)."""
    try:
        sa = _normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON)
        manifest = _fetch_drive_manifest(sa, settings.GDRIVE_FOLDER_ID)
        files = manifest.get("files", [])
        # 최신 수정순 정렬
        files.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)
        rows = []
        for f in files[:max_items]:
            fid = f["id"]
            rows.append({
                "name": f.get("name", ""),
                "link": f"https://drive.google.com/file/d/{fid}/view",
                "mime": f.get("mimeType", ""),
                "modified": f.get("modifiedTime", ""),
            })
        return True, "OK", rows
    except Exception as e:
        return False, str(e), []
