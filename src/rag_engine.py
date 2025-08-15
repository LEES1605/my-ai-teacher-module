# src/rag_engine.py — 증분(스텝) 인덱싱 + 체크포인트/Resume + chat_log 제외 + 임베딩서명검사
from __future__ import annotations

import os
import json
import time
import threading
from queue import Queue
from typing import Callable, Any, Mapping, Iterable, List

import streamlit as st
from src.config import settings

# ================================ 예외 ================================
class CancelledError(Exception):
    """사용자 취소를 명시적으로 표현하기 위한 예외"""
    pass


# ============================ 임베딩/LLM =============================
def set_embed_provider(provider: str, api_key: str, model: str) -> None:
    """
    LlamaIndex Settings.embed_model 설정
    """
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
    """
    간단한 LLM 생성기 (Google / OpenAI)
    """
    if provider == "google":
        from llama_index.llms.google_genai import GoogleGenAI
        return GoogleGenAI(api_key=api_key, model=model, temperature=temperature)

    elif provider == "openai":
        from llama_index.llms.openai import OpenAI
        return OpenAI(api_key=api_key, model=model, temperature=temperature)

    else:
        raise ValueError(f"Unknown llm provider: {provider}")


def llm_complete(llm, prompt: str, temperature: float = 0.0) -> str:
    """
    검색/RAG 없이 LLM만으로 바로 완성하는 유틸 (Gemini/OpenAI 모두 호환)
    """
    try:
        resp = llm.complete(prompt, temperature=temperature)  # GoogleGenAI 스타일
        return getattr(resp, "text", str(resp))
    except AttributeError:
        # OpenAI LlamaIndex 구현체는 predict만 있는 경우가 있음
        return llm.predict(prompt)


# ============================ Drive 유틸 =============================
def _normalize_sa(raw: Any) -> Mapping[str, Any]:
    """
    서비스계정 JSON: dict 또는 JSON 문자열 모두 허용 → dict 정규화
    """
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
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
    공유드라이브 포함 재귀 스냅샷. exclude_folder_names는 재귀 진입 제외(예: ["chat_log"])
    반환: {root, files:[{id,name,mimeType,modifiedTime,size,md5Checksum}], count}
    """
    svc = _build_drive_service(creds_dict)
    exclude_l = set([x.strip().lower() for x in (exclude_folder_names or [])])

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

    all_files: List[dict] = []
    queue: List[str] = [root_folder_id]
    seen: set[str] = set()

    while queue:
        fid = queue.pop(0)
        if fid in seen:
            continue
        seen.add(fid)

        files, folders = list_children(fid)
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
    LlamaIndex 버전/구성에 따라 메서드명이 다를 수 있어 호환 래퍼로 처리.
    우선순위: insert(documents) → insert_documents(documents) → insert_nodes(nodes)
    """
    try:
        # 0.12.x 계열에서 주로 동작
        return index.insert(docs)
    except Exception:
        try:
            # 어떤 조합에서는 insert_documents가 제공됨
            return index.insert_documents(docs)
        except Exception:
            # 노드 API로 강제 삽입 (최후의 보루)
            try:
                from llama_index.core.node_parser import SimpleNodeParser
                parser = SimpleNodeParser()
                nodes = []
                for d in docs:
                    nodes.extend(parser.get_nodes_from_documents([d]))
                return index.insert_nodes(nodes)
            except Exception as e:
                raise e


# ============================ 매니페스트 비교 ============================
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
        return (d.get("id",""), d.get("modifiedTime",""), d.get("size",""), d.get("md5Checksum",""))
    ls = sorted([sig(x) for x in local.get("files", [])])
    rs = sorted([sig(x) for x in remote.get("files", [])])
    return ls != rs


# ============================ 일괄(한방) 인덱서 ============================
def _build_index_with_progress(
    update_pct: Callable[[int, str | None], None],
    update_msg: Callable[[str], None],
    gdrive_folder_id: str,
    gcp_creds: Mapping[str, Any],
    persist_dir: str,
    exclude_folder_names: Iterable[str] | None = None,
    max_docs: int | None = None,
    is_cancelled: Callable[[], bool] | None = None,
):
    """
    파일 단위 증분 삽입 + (주기적) persist + 체크포인트 기록 → 도중 종료/리런에도 이어서.
    exclude_folder_names: 예) ["chat_log"]
    """
    from llama_index.core import VectorStoreIndex
    from llama_index.readers.google import GoogleDriveReader

    # 0) 기존 인덱스 로드 or 빈 인덱스
    try:
        index = _load_index_from_disk(persist_dir)
        update_msg("이전 진행분을 불러왔습니다(Resume).")
    except Exception:
        index = VectorStoreIndex.from_documents([])
        _persist_index(index, persist_dir)

    # 1) 매니페스트 & 체크포인트
    update_pct(8, "Drive 파일 목록 불러오는 중…")
    manifest = _fetch_drive_manifest(gcp_creds, gdrive_folder_id, exclude_folder_names=exclude_folder_names)
    files_all = manifest.get("files", [])
    if max_docs:
        files_all = files_all[:max_docs]
    total = len(files_all)

    ckpt = _load_ckpt(persist_dir)
    done_ids: set[str] = set(ckpt.get("done_ids", []))
    pending = [f for f in files_all if f["id"] not in done_ids]
    done_count = len(done_ids)

    if total == 0:
        update_pct(100, "폴더에 학습할 파일이 없습니다.")
        st.session_state["indexing_report"] = {
            "total_manifest": 0, "loaded_docs": 0, "skipped_count": 0, "skipped": []
        }
        return index

    reader = GoogleDriveReader(service_account_key=gcp_creds, recursive=False)

    batch, BATCH_PERSIST = [], 8
    skipped = []

    for i, f in enumerate(pending, start=1):
        if is_cancelled and is_cancelled():
            raise CancelledError("사용자 취소(진행 중)")

        fid, name, mime = f["id"], f.get("name",""), f.get("mimeType","")
        try:
            docs = reader.load_data(file_ids=[fid])
            for d in docs:
                try:
                    d.metadata["file_name"] = name
                except Exception:
                    pass
            batch.extend(docs)

            if len(batch) >= BATCH_PERSIST:
                _insert_docs(index, batch)
                _persist_index(index, persist_dir)
                batch.clear()

            done_ids.add(fid)
            _save_ckpt(persist_dir, {"done_ids": list(done_ids)})

            processed = done_count + i
            pct = 8 + int(80 * processed / max(total, 1))  # 8%→88%
            update_pct(pct, f"로딩 {processed}/{total} — {name}")

        except Exception as e:
            skipped.append({"name": name, "mime": mime, "reason": str(e)})
            update_msg("⚠️ 스킵: " + f"{name} ({mime}) — {e}")

    if batch:
        _insert_docs(index, batch)

    update_pct(92, "인덱스 저장 중…")
    _persist_index(index, persist_dir)

    _clear_ckpt(persist_dir)
    update_pct(100, "완료")
    st.session_state["indexing_report"] = {
        "total_manifest": total,
        "loaded_docs": len(done_ids),
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
    return index


# ======================== 엔트리: 빌드 or 로드 ========================
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
    """
    변경 없으면 저장본 로드, 변경 있으면 일괄 빌드(Resume 지원)
    """
    gcp_creds = _normalize_sa(raw_sa)

    # 현재 임베딩 서명
    from llama_index.core import Settings
    cur_sig = {
        "embed_provider": (
            "openai" if "openai" in str(type(getattr(Settings, "embed_model", None))).lower() else "google"
        ),
        "embed_model": getattr(Settings.embed_model, "model", getattr(Settings.embed_model, "_model_name", "")),
    }
    old_sig = _load_signature(persist_dir)

    # 드라이브 매니페스트 (chat_log 제외)
    update_pct(5, "드라이브 변경 확인 중…")
    remote = _fetch_drive_manifest(gcp_creds, gdrive_folder_id, exclude_folder_names=["chat_log"])
    local = _load_local_manifest(manifest_path)

    need_rebuild = False
    if old_sig != cur_sig:
        need_rebuild = True  # 임베딩 모델/공급자 변경시 풀리빌드
        update_msg("임베딩 설정이 변경되어 재인덱싱합니다.")
    elif _manifests_differ(local, remote):
        need_rebuild = True  # 파일 변경 발생

    if os.path.exists(persist_dir) and not need_rebuild:
        # 변경 없음 → 저장본 로드
        update_pct(25, "변경 없음 → 저장된 두뇌 로딩")
        idx = _load_index_from_disk(persist_dir)
        update_pct(100, "완료!")
        st.session_state.setdefault("indexing_report", {
            "total_manifest": len(remote.get("files", [])),
            "loaded_docs": -1, "skipped_count": 0, "skipped": []
        })
        return idx

    # 변경이 있거나 저장본 없음 → 증분 빌드
    idx = _build_index_with_progress(
        update_pct=update_pct,
        update_msg=update_msg,
        gdrive_folder_id=gdrive_folder_id,
        gcp_creds=gcp_creds,
        persist_dir=persist_dir,
        exclude_folder_names=["chat_log"],
        max_docs=max_docs,
        is_cancelled=is_cancelled,
    )

    _save_local_manifest(manifest_path, remote)
    _save_signature(persist_dir, cur_sig)
    return idx


# ============================ Step-wise Indexer =============================
# 긴 인덱싱을 여러 번의 짧은 스텝으로 나눠서 실행하기 위한 헬퍼입니다.
# - 각 스텝은 최대 N개 파일만 처리하고 즉시 반환
# - 다음 틱에서 다시 step()을 호출하면 이어서 진행
# - cancel 플래그는 다음 스텝에서 즉시 반영
# - 파일 로딩이 오래 걸릴 때를 대비해 per-file timeout 지원

def _load_with_timeout(reader, fid: str, timeout_s: int = 40):
    """GoogleDriveReader.load_data(file_ids=[fid])에 timeout을 거는 래퍼."""
    q: Queue = Queue()
    err: Queue = Queue()

    def _worker():
        try:
            docs = reader.load_data(file_ids=[fid])
            q.put(docs)
        except Exception as e:
            err.put(e)

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        return None, TimeoutError(f"load_data timeout after {timeout_s}s")
    if not err.empty():
        raise err.get()
    return q.get(), None


class IndexBuilder:
    """증분(step) 인덱서를 캡슐화한 객체. Streamlit session_state에 그대로 보관 가능."""
    def __init__(
        self,
        gdrive_folder_id: str,
        gcp_creds: Mapping[str, Any],
        persist_dir: str,
        exclude_folder_names: Iterable[str] | None = None,
        max_docs: int | None = None,
    ):
        from llama_index.core import VectorStoreIndex
        from llama_index.readers.google import GoogleDriveReader

        # 존재하면 로드, 아니면 빈 인덱스 생성
        try:
            self.index = _load_index_from_disk(persist_dir)
            self._resumed = True
        except Exception:
            self.index = VectorStoreIndex.from_documents([])
            _persist_index(self.index, persist_dir)
            self._resumed = False

        self.reader = GoogleDriveReader(service_account_key=gcp_creds, recursive=False)
        self.persist_dir = persist_dir

        # 매니페스트
        self.manifest = _fetch_drive_manifest(
            gcp_creds, gdrive_folder_id, exclude_folder_names=exclude_folder_names
        )
        self.files_all: list[dict] = self.manifest.get("files", [])
        if max_docs:
            self.files_all = self.files_all[:max_docs]
        self.total = len(self.files_all)

        # 체크포인트
        ckpt = _load_ckpt(persist_dir)
        self.done_ids: set[str] = set(ckpt.get("done_ids", []))
        self.skipped: list[dict] = []
        self.batch: list = []
        self.BATCH_PERSIST = 8

        # 진행률 캐시
        self._last_pct = 0

    @property
    def processed(self) -> int:
        return len(self.done_ids)

    @property
    def pending_ids(self) -> list[str]:
        ids_all = [f["id"] for f in self.files_all]
        return [fid for fid in ids_all if fid not in self.done_ids]

    def step(
        self,
        max_files: int = 5,
        per_file_timeout_s: int = 40,
        is_cancelled: Callable[[], bool] | None = None,
        on_pct: Callable[[int, str | None], None] | None = None,
        on_msg: Callable[[str], None] | None = None,
    ) -> str:
        """
        최대 max_files 개만 처리하고 즉시 반환.
        return: "running" | "done"
        """
        if self.total == 0:
            if on_pct: on_pct(100, "폴더에 학습할 파일이 없습니다.")
            return "done"

        # 남은 파일 없으면 마무리 저장
        pend = self.pending_ids
        if not pend:
            if self.batch:
                _insert_docs(self.index, self.batch)
                self.batch.clear()
            _persist_index(self.index, self.persist_dir)
            _clear_ckpt(self.persist_dir)
            if on_pct: on_pct(100, "완료")
            return "done"

        # 이번 스텝 작업 목록
        work = pend[:max_files]
        for fid in work:
            if is_cancelled and is_cancelled():
                raise CancelledError("사용자 취소(스텝 중)")

            # 진행 퍼센트/메시지
            name = next((f.get("name","") for f in self.files_all if f["id"] == fid), fid)
            pct = 8 + int(80 * (self.processed / max(self.total, 1)))  # 8 → 88%
            if on_pct and pct != self._last_pct:
                self._last_pct = pct
                on_pct(pct, f"로딩 {self.processed}/{self.total} — {name}")

            try:
                docs, timeout_err = _load_with_timeout(self.reader, fid, timeout_s=per_file_timeout_s)
                if timeout_err is not None:
                    raise timeout_err

                for d in docs:
                    try:
                        d.metadata["file_name"] = name
                    except Exception:
                        pass
                self.batch.extend(docs)

                if len(self.batch) >= self.BATCH_PERSIST:
                    _insert_docs(self.index, self.batch)
                    _persist_index(self.index, self.persist_dir)
                    self.batch.clear()

                self.done_ids.add(fid)
                _save_ckpt(self.persist_dir, {"done_ids": list(self.done_ids)})

            except Exception as e:
                self.skipped.append({"name": name, "reason": str(e)})
                if on_msg: on_msg("⚠️ 스킵: " + f"{name} — {e}")

            if is_cancelled and is_cancelled():
                raise CancelledError("사용자 취소(스텝 사이)")

        # 스텝 종료 진행률
        pct = 8 + int(80 * (self.processed / max(self.total, 1)))
        if on_pct: on_pct(pct, f"로딩 {self.processed}/{self.total}")

        return "running"


def start_index_builder(
    gdrive_folder_id: str,
    gcp_creds: Mapping[str, Any],
    persist_dir: str,
    exclude_folder_names: Iterable[str] | None = None,
    max_docs: int | None = None,
) -> IndexBuilder:
    """IndexBuilder 인스턴스를 초기화해서 돌려준다."""
    return IndexBuilder(
        gdrive_folder_id=gdrive_folder_id,
        gcp_creds=gcp_creds,
        persist_dir=persist_dir,
        exclude_folder_names=exclude_folder_names,
        max_docs=max_docs,
    )


# ============================ QA 유틸 =============================
def get_text_answer(query_engine, question: str, system_prompt: str) -> str:
    """
    업로드된 자료를 최우선으로 참고하는 답변을 생성하고, 참조 파일명을 하단에 덧붙입니다.
    """
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


# ============================ 테스트 유틸 ============================
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
