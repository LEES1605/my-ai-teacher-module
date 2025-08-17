# src/rag_engine.py — 증분 인덱싱(스텝) + 재개/취소 + 진행조회
#  - Drive 매니페스트 비교로 변경 감지
#  - chat_log 폴더 제외
#  - 인덱스 영구 저장/로드, 체크포인트
#  - LlamaIndex 버전차 호환 insert(문서/노드) 처리
from __future__ import annotations

import os
import json
from typing import Any, Callable, Iterable, Mapping, Optional

import streamlit as st
from src.config import settings

# 파일 상단 import 근처
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

# 모듈 로드 시 한 번만 기본 분할기 지정 (수능 지문에 맞게)
def _configure_index_defaults():
    # 토큰 기준이지만 대략 영문 600~900단어, 한글 800~1200자 정도가 한 청크
    Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=120)
    # 답변 최대 토큰(과도한 출력 방지)
    try:
        Settings.num_output = 512
    except Exception:
        pass

# 모듈 import 시 적용
_configure_index_defaults()

# ================================ 예외 ================================
class CancelledError(Exception):
    """사용자 취소를 나타내는 예외"""
    pass


# ============================ 임베딩/LLM =============================
def set_embed_provider(provider: str, api_key: str, model: str) -> None:
    """llama_index Settings.embed_model 설정"""
    from llama_index.core import Settings
    if provider == "google":
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        Settings.embed_model = GoogleGenAIEmbedding(model=model, api_key=api_key)
    elif provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        Settings.embed_model = OpenAIEmbedding(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unknown embed provider: {provider}")


def make_llm(provider: str, api_key: str, model: str, temperature: float = 0.0):
    """llama_index LLM 생성 (Google / OpenAI)"""
    if provider == "google":
        from llama_index.llms.google_genai import GoogleGenAI
        return GoogleGenAI(api_key=api_key, model=model, temperature=temperature)
    elif provider == "openai":
        from llama_index.llms.openai import OpenAI
        return OpenAI(api_key=api_key, model=model, temperature=temperature)
    else:
        raise ValueError(f"Unknown llm provider: {provider}")


def llm_complete(llm, prompt: str) -> str:
    """검색 없이 LLM 단독으로 완성"""
    try:
        resp = llm.complete(prompt)
        return getattr(resp, "text", str(resp))
    except AttributeError:
        # 일부 드라이버는 .predict만 제공
        return llm.predict(prompt)


# ============================ Drive 유틸 =============================
def _normalize_sa(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, str):
        return json.loads(raw)
    elif isinstance(raw, Mapping):
        return raw
    return {}


def _build_drive_service(creds_dict):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _fetch_drive_manifest(
    creds_dict,
    root_folder_id: str,
    exclude_folder_names: Iterable[str] | None = None,
) -> dict:
    """
    지정 폴더 이하 전체 파일 스냅샷(공유드라이브 포함).
    exclude_folder_names에 포함된 이름의 폴더는 *재귀 진입 제외* (예: chat_log).
    """
    svc = _build_drive_service(creds_dict)
    exclude_l = set(x.strip().lower() for x in (exclude_folder_names or []))

    def list_children(folder_id: str):
        files, folders, page_token = [], [], None
        q = f"'{folder_id}' in parents and trashed=false"
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

    all_files, queue, seen = [], [root_folder_id], set()
    while queue:
        fid = queue.pop(0)
        if fid in seen:
            continue
        seen.add(fid)

        files, folders = list_children(fid)
        # 제외 폴더는 재귀 진입하지 않음
        allowed = []
        for f in folders:
            if f.get("name", "").strip().lower() in exclude_l:
                continue
            allowed.append(f)

        all_files.extend(files)
        queue.extend([f["id"] for f in allowed])

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
    ls = sorted(sig(x) for x in local.get("files", []))
    rs = sorted(sig(x) for x in remote.get("files", []))
    return ls != rs


# ============================ 인덱스 I/O =============================
def _load_index_from_disk(persist_dir: str):
    from llama_index.core import StorageContext, load_index_from_storage
    return load_index_from_storage(StorageContext.from_defaults(persist_dir=persist_dir))


def _persist_index(index, persist_dir: str) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_dir)


def _ckpt_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, "_ingest_progress.json")


def _sig_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, "_index_signature.json")


def _load_ckpt(persist_dir: str) -> dict:
    try:
        with open(_ckpt_path(persist_dir), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"done_ids": []}


def _save_ckpt(persist_dir: str, data: dict) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    with open(_ckpt_path(persist_dir), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _clear_ckpt(persist_dir: str) -> None:
    try:
        os.remove(_ckpt_path(persist_dir))
    except Exception:
        pass


def _load_signature(persist_dir: str) -> dict:
    try:
        with open(_sig_path(persist_dir), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_signature(persist_dir: str, sig: dict) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    with open(_sig_path(persist_dir), "w", encoding="utf-8") as f:
        json.dump(sig, f, ensure_ascii=False, indent=2)


def _insert_docs(index, docs):
    """
    LlamaIndex 버전마다 API가 다른 문제를 흡수:
      - insert_documents(docs)
      - insert_nodes(nodes)
      - insert(docs_or_nodes)
    """
    # 1) try: insert_documents
    try:
        return index.insert_documents(docs)
    except Exception:
        pass
    # 2) try: insert_nodes
    try:
        return index.insert_nodes(docs)
    except Exception:
        pass
    # 3) fallback: insert
    try:
        return index.insert(docs)
    except Exception as e:
        raise e


# ============================ Step Job 객체 ============================
class _StepJob:
    """세션에 그대로 담아두는 간단한 잡 상태 객체(직렬화 불요)"""
    def __init__(
        self,
        gcp_creds: Mapping[str, Any],
        gdrive_folder_id: str,
        persist_dir: str,
        manifest_path: str,
        exclude_folders: Iterable[str],
        to_process: list[dict],
        already_done: set[str],
        total: int,
        max_docs: Optional[int],
    ):
        self.gcp_creds = gcp_creds
        self.gdrive_folder_id = gdrive_folder_id
        self.persist_dir = persist_dir
        self.manifest_path = manifest_path
        self.exclude_folders = list(exclude_folders)
        self.pending = to_process  # list of manifest file dicts
        self.done_ids = set(already_done)
        self.total = int(total)
        self.max_docs = max_docs
        self.cancel = False   # 외부 취소 플래그

        # 런타임 자원
        self.reader = None
        self.index = None
        self._prepared = False

        # 리포트
        self.skipped: list[dict] = []

    # 내부 준비(기존 인덱스 로드/빈 인덱스 생성 + Reader 준비)
    def _prepare(self):
        if self._prepared:
            return
        from llama_index.core import VectorStoreIndex
        from llama_index.readers.google import GoogleDriveReader

        # 기존 인덱스를 불러오거나 빈 인덱스 생성
        try:
            self.index = _load_index_from_disk(self.persist_dir)
        except Exception:
            self.index = VectorStoreIndex.from_documents([])
            _persist_index(self.index, self.persist_dir)

        # Reader 준비
        self.reader = GoogleDriveReader(service_account_key=self.gcp_creds, recursive=False)
        self._prepared = True


# ===================== 인덱싱 엔트리(스텝 시작/재개/취소) =====================
def start_index_builder(
    update_pct: Callable[[int, Optional[str]], None],
    update_msg: Callable[[str], None],
    gdrive_folder_id: str,
    raw_sa: Any | None,
    persist_dir: str,
    manifest_path: str,
    max_docs: int | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict:
    """
    변경이 없으면 즉시 저장된 인덱스를 로드하여 반환.
    변경이 있으면 StepJob을 만들어 'running' 상태를 반환.
    """
    gcp_creds = _normalize_sa(raw_sa)

    # 현재 임베딩 서명(공급자/모델)
    from llama_index.core import Settings
    embed = getattr(Settings, "embed_model", None)
    cur_sig = {
        "embed_provider": (
            "openai" if "openai" in str(type(embed)).lower()
            else ("google" if "google" in str(type(embed)).lower() else "unknown")
        ),
        "embed_model": getattr(embed, "model", getattr(embed, "_model_name", "")),
    }
    old_sig = _load_signature(persist_dir)

    # 드라이브 매니페스트 (chat_log 제외)
    update_pct(4, "드라이브 변경 확인 중…")
    remote = _fetch_drive_manifest(gcp_creds, gdrive_folder_id, exclude_folder_names=["chat_log"])
    local = _load_local_manifest(manifest_path)

    need_rebuild = False
    if old_sig != cur_sig:
        need_rebuild = True
        update_msg("임베딩 설정이 변경되어 재인덱싱합니다.")
    elif _manifests_differ(local, remote):
        need_rebuild = True

    # 변경 없음 → 저장본 로드
    if os.path.exists(persist_dir) and not need_rebuild:
        try:
            update_pct(25, "변경 없음 → 저장된 두뇌 로딩")
            idx = _load_index_from_disk(persist_dir)
            update_pct(100, "완료!")
            st.session_state.setdefault("indexing_report", {
                "total_manifest": len(remote.get("files", [])),
                "loaded_docs": -1, "skipped_count": 0, "skipped": []
            })
            return {"status": "done", "index": idx}
        except Exception as e:
            update_msg(f"저장된 인덱스 로드 실패: {e} → 재인덱싱 시도")
            # 계속 진행해 StepJob 생성

    # 증분 빌드를 위한 StepJob 준비
    files_all = remote.get("files", [])
    if max_docs:
        files_all = files_all[: int(max_docs)]

    ckpt = _load_ckpt(persist_dir)
    done_ids: set[str] = set(ckpt.get("done_ids", []))
    pending = [f for f in files_all if f["id"] not in done_ids]

    job = _StepJob(
        gcp_creds=gcp_creds,
        gdrive_folder_id=gdrive_folder_id,
        persist_dir=persist_dir,
        manifest_path=manifest_path,
        exclude_folders=["chat_log"],
        to_process=pending,
        already_done=done_ids,
        total=len(files_all),
        max_docs=max_docs,
    )
    # 초반 진행률
    pct = 8 if pending else 92
    return {"status": "running", "job": job, "pct": pct, "msg": "인덱싱 시작"}


def resume_index_builder(
    job: _StepJob,
    update_pct: Callable[[int, Optional[str]], None],
    update_msg: Callable[[str], None],
    is_cancelled: Callable[[], bool] | None = None,
    batch_size: int = 8,
) -> dict:
    """
    pending 목록을 batch_size만큼 처리하고 진행률/메시지 반환.
    모든 처리가 끝나면 인덱스를 저장하고 'done' 반환.
    """
    from math import floor

    job._prepare()

    if job.total == 0:
        update_pct(100, "폴더에 학습할 파일이 없습니다.")
        st.session_state["indexing_report"] = {
            "total_manifest": 0, "loaded_docs": 0, "skipped_count": 0, "skipped": []
        }
        return {"status": "done", "index": job.index}

    B = max(1, int(batch_size))
    processed_now = 0

    while processed_now < B and job.pending:
        if job.cancel or (is_cancelled and is_cancelled()):
            return {"status": "cancelled"}

        f = job.pending.pop(0)
        fid, name, mime = f["id"], f.get("name", ""), f.get("mimeType", "")
        try:
            docs = job.reader.load_data(file_ids=[fid])
            # 파일명 메타 보강
            for d in docs:
                try:
                    d.metadata["file_name"] = name
                except Exception:
                    pass

            # 삽입
            _insert_docs(job.index, docs)

            # 주기적 퍼시스트(작게라도 안전)
            if (len(job.done_ids) + processed_now) % 8 == 0:
                _persist_index(job.index, job.persist_dir)

            job.done_ids.add(fid)
            _save_ckpt(job.persist_dir, {"done_ids": list(job.done_ids)})

            processed_now += 1
            processed = len(job.done_ids)
            pct = 8 + int(80 * processed / max(job.total, 1))  # 8% → 88%
            update_pct(pct, f"로딩 {processed}/{job.total} — {name}")
        except Exception as e:
            msg = f"{name} ({mime}) — {type(e).__name__}: {e}"
            job.skipped.append({"name": name, "mime": mime, "reason": str(e)})
            update_msg("⚠️ 스킵: " + msg)

    # 아직 남아있으면 running
    if job.pending:
        processed = len(job.done_ids)
        pct = 8 + int(80 * processed / max(job.total, 1))
        return {"status": "running", "pct": pct, "msg": "진행 중…"}

    # 모두 처리됨 → 저장/정리
    update_pct(92, "인덱스 저장 중…")
    _persist_index(job.index, job.persist_dir)
    _clear_ckpt(job.persist_dir)
    update_pct(100, "완료")

    # 매니페스트/시그니처 저장
    remote = _fetch_drive_manifest(job.gcp_creds, job.gdrive_folder_id, exclude_folder_names=job.exclude_folders)
    _save_local_manifest(job.manifest_path, remote)

    from llama_index.core import Settings
    embed = getattr(Settings, "embed_model", None)
    sig = {
        "embed_provider": (
            "openai" if "openai" in str(type(embed)).lower()
            else ("google" if "google" in str(type(embed)).lower() else "unknown")
        ),
        "embed_model": getattr(embed, "model", getattr(embed, "_model_name", "")),
    }
    _save_signature(job.persist_dir, sig)

    st.session_state["indexing_report"] = {
        "total_manifest": job.total,
        "loaded_docs": len(job.done_ids),
        "skipped_count": len(job.skipped),
        "skipped": job.skipped,
    }
    return {"status": "done", "index": job.index}


def cancel_index_builder(job: Optional[_StepJob]) -> None:
    """외부 취소 플래그 설정(다음 스텝 호출 시 취소)"""
    if job is not None:
        job.cancel = True


def get_index_progress(job: Optional[_StepJob]) -> dict:
    """현재 잡의 진행 상황 요약(간단 조회용)"""
    if job is None:
        return {"running": False}
    return {
        "running": True,
        "total": job.total,
        "done": len(job.done_ids),
        "pending": len(job.pending),
        "skipped_count": len(job.skipped),
    }


# ============================ QA 유틸 =============================
def get_text_answer(query_engine, question: str, system_prompt: str) -> str:
    """
    업로드 자료를 최우선으로 참고하여 답변하고,
    출처(파일명)를 하단에 표시한다.
    """
    try:
        full_query = (
            f"{system_prompt}\n\n"
            "[지시사항] 반드시 업로드된 강의/학습 자료를 최우선으로 참고하여 답변하고, "
            "근거를 찾을 수 없다면 그 사실을 명확히 밝혀라.\n\n"
            f"[학생의 질문]\n{question}"
        )
        response = query_engine.query(full_query)
        answer_text = str(response)

        try:
            files = [n.metadata.get("file_name", "알 수 없음") for n in getattr(response, "source_nodes", [])]
            source_files = ", ".join(sorted(set(files))) if files else "출처 정보 없음"
        except Exception:
            source_files = "출처 정보 없음"

        return f"{answer_text}\n\n---\n*참고 자료: {source_files}*"
    except Exception as e:
        return f"텍스트 답변 생성 중 오류 발생: {type(e).__name__}: {e}"


# ============================ 테스트 유틸 ============================
def smoke_test_drive() -> tuple[bool, str]:
    """서비스계정으로 Drive 폴더에 접근 가능한지 간단 확인"""
    try:
        sa = _normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON)
        svc = _build_drive_service(sa)
        fid = settings.GDRIVE_FOLDER_ID
        meta = svc.files().get(
            fileId=fid,
            fields="id,name,driveId,parents",
            supportsAllDrives=True
        ).execute()
        name = meta.get("name", "")
        drive_id = meta.get("driveId", "")
        return True, f"✅ Drive 연결 OK · 폴더명: {name} · driveId: {drive_id or 'MyDrive'}"
    except Exception as e:
        return False, f"Drive 연결/권한 확인 실패: {e}"


def preview_drive_files(max_items: int = 10) -> tuple[bool, str, list[dict]]:
    """최신 수정 순으로 상위 N개 파일 미리보기"""
    try:
        sa = _normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON)
        manifest = _fetch_drive_manifest(sa, settings.GDRIVE_FOLDER_ID, exclude_folder_names=["chat_log"])
        files = manifest.get("files", [])
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
