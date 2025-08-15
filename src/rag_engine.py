# src/rag_engine.py — 스텝 인덱싱 + Resume/Cancel + chat_log 제외 + 삽입호환
from __future__ import annotations
import os, json, time
from typing import Callable, Any, Mapping, Iterable, Tuple, Optional

import streamlit as st
from src.config import settings

# ================================ 예외 ================================
class CancelledError(Exception):
    pass

# ============================ 임베딩/LLM =============================
def set_embed_provider(provider: str, api_key: str, model: str) -> None:
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
    if provider == "google":
        from llama_index.llms.google_genai import GoogleGenAI
        return GoogleGenAI(api_key=api_key, model=model, temperature=temperature)
    elif provider == "openai":
        from llama_index.llms.openai import OpenAI
        return OpenAI(api_key=api_key, model=model, temperature=temperature)
    else:
        raise ValueError(f"Unknown llm provider: {provider}")

# LLM 무검색(직접 완성)
def llm_complete(llm, prompt: str, temperature: float = 0.0) -> str:
    try:
        resp = llm.complete(prompt)
        return getattr(resp, "text", str(resp))
    except AttributeError:
        return llm.predict(prompt)

# ============================ Drive 유틸 =============================
def _normalize_sa(raw: Any) -> Mapping[str, Any]:
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
    """공유드라이브 포함 재귀 스냅샷. exclude_folder_names는 재귀 진입 제외(예: chat_log)."""
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

    all_files, queue, seen = [], [root_folder_id], set()
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

# ---- LlamaIndex 0.12 호환 삽입 ----
def _insert_docs(index, docs):
    """insert / insert_nodes 중 가능한 메서드로 삽입. 비어있는 문서 스킵."""
    if not docs:
        return
    cleaned = []
    for d in docs:
        try:
            txt = getattr(d, "text", None)
            if txt is None and hasattr(d, "get_text"):
                txt = d.get_text()
            if txt is None:
                txt = ""
            if str(txt).strip():
                cleaned.append(d)
        except Exception:
            pass
    if not cleaned:
        return

    if hasattr(index, "insert"):
        index.insert(cleaned)
        return

    if hasattr(index, "insert_nodes"):
        try:
            from llama_index.core.node_parser import SimpleNodeParser
            nodes = SimpleNodeParser.from_defaults().get_nodes_from_documents(cleaned)
            index.insert_nodes(nodes)
            return
        except Exception:
            pass

    raise RuntimeError("Index has no supported insert method (insert / insert_nodes).")

# ====================== 진행도 계산(스텝 빌더 공용) ======================
def _progress_pct(persist_dir: str, total: int) -> int:
    if total <= 0:
        return 100
    done = len(_load_ckpt(persist_dir).get("done_ids", []))
    base = 8  # 매니페스트 이후 시작지점
    return base + int(80 * min(done, total) / max(total, 1))

# ======================== 스텝 빌더 (시작/재개/취소) =====================
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
    """스텝 빌더 시작. 필요 없으면 즉시 done + index 반환."""
    gcp = _normalize_sa(raw_sa)

    # 현재 임베딩 서명
    from llama_index.core import Settings
    cur_sig = {
        "embed_provider": ("openai" if "openai" in str(type(getattr(Settings, "embed_model", None))).lower() else "google"),
        "embed_model": getattr(Settings.embed_model, "model", getattr(Settings.embed_model, "_model_name", "")),
    }
    old_sig = _load_signature(persist_dir)

    update_pct(5, "드라이브 변경 확인 중…")
    remote = _fetch_drive_manifest(gcp, gdrive_folder_id, exclude_folder_names=["chat_log"])
    files_all = remote.get("files", [])
    if max_docs:
        files_all = files_all[:max_docs]

    local = _load_local_manifest(manifest_path)
    need_rebuild = (old_sig != cur_sig) or _manifests_differ(local, {"files": files_all})

    if os.path.exists(persist_dir) and not need_rebuild:
        # 변경 없음 → 저장본 로드
        update_pct(25, "변경 없음 → 저장된 두뇌 로딩")
        idx = _load_index_from_disk(persist_dir)
        update_pct(100, "완료!")
        st.session_state.setdefault("indexing_report", {
            "total_manifest": len(files_all),
            "loaded_docs": -1, "skipped_count": 0, "skipped": []
        })
        return {"status": "done", "index": idx}

    # 재빌드 필요 → 체크포인트 초기화 보장(없으면 생성)
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir, exist_ok=True)
    ck = _load_ckpt(persist_dir)
    if "done_ids" not in ck:
        _save_ckpt(persist_dir, {"done_ids": []})

    # 빈 인덱스 보장
    try:
        _load_index_from_disk(persist_dir)
        update_msg("이전 진행분을 불러왔습니다(Resume).")
    except Exception:
        from llama_index.core import VectorStoreIndex
        idx = VectorStoreIndex.from_documents([])
        _persist_index(idx, persist_dir)

    # 아직 진행 중
    return {
        "status": "running",
        "job": {
            "gdrive_folder_id": gdrive_folder_id,
            "persist_dir": persist_dir,
            "manifest_path": manifest_path,
            "max_docs": max_docs,
            "gcp": gcp,
        },
        "total": len(files_all),
        "pct": _progress_pct(persist_dir, len(files_all)),
        "msg": "인덱싱 준비 중…",
    }

def resume_index_builder(
    job: dict,
    update_pct: Callable[[int, Optional[str]], None],
    update_msg: Callable[[str], None],
    is_cancelled: Callable[[], bool] | None = None,
    batch_size: int = 6,
) -> dict:
    """한 스텝만 진행. done이면 index 반환."""
    gcp = job["gcp"]
    gdrive_folder_id = job["gdrive_folder_id"]
    persist_dir = job["persist_dir"]
    manifest_path = job["manifest_path"]
    max_docs = job.get("max_docs")

    if is_cancelled and is_cancelled():
        return {"status": "cancelled", "msg": "사용자 취소"}

    # 매니페스트/대상 파일
    remote = _fetch_drive_manifest(gcp, gdrive_folder_id, exclude_folder_names=["chat_log"])
    files_all = remote.get("files", [])
    if max_docs:
        files_all = files_all[:max_docs]
    total = len(files_all)

    # 진행상태
    ck = _load_ckpt(persist_dir)
    done_ids: set[str] = set(ck.get("done_ids", []))
    pending = [f for f in files_all if f["id"] not in done_ids]

    # 모두 끝난 경우 → 마무리
    if not pending:
        try:
            index = _load_index_from_disk(persist_dir)
        except Exception:
            from llama_index.core import VectorStoreIndex
            index = VectorStoreIndex.from_documents([])
            _persist_index(index, persist_dir)

        update_pct(92, "인덱스 저장 중…")
        _persist_index(index, persist_dir)

        # 서명/매니페스트 저장
        from llama_index.core import Settings
        cur_sig = {
            "embed_provider": ("openai" if "openai" in str(type(getattr(Settings, "embed_model", None))).lower() else "google"),
            "embed_model": getattr(Settings.embed_model, "model", getattr(Settings.embed_model, "_model_name", "")),
        }
        _save_signature(persist_dir, cur_sig)
        _save_local_manifest(manifest_path, {"files": files_all})

        _clear_ckpt(persist_dir)
        update_pct(100, "완료")
        return {"status": "done", "index": index}

    # 이번 스텝에서 처리할 파일들
    step_files = pending[:batch_size]

    from llama_index.readers.google import GoogleDriveReader
    reader = GoogleDriveReader(service_account_key=gcp, recursive=False)

    # 인덱스 로드
    try:
        index = _load_index_from_disk(persist_dir)
    except Exception:
        from llama_index.core import VectorStoreIndex
        index = VectorStoreIndex.from_documents([])
        _persist_index(index, persist_dir)

    batch, skipped = [], []
    for f in step_files:
        if is_cancelled and is_cancelled():
            return {"status": "cancelled", "msg": "사용자 취소"}

        fid, name, mime = f["id"], f.get("name", ""), f.get("mimeType", "")
        try:
            docs = reader.load_data(file_ids=[fid])
            for d in docs:
                try:
                    d.metadata["file_name"] = name
                except Exception:
                    pass
            batch.extend(docs)
            done_ids.add(fid)
            _save_ckpt(persist_dir, {"done_ids": list(done_ids)})

            pct = _progress_pct(persist_dir, total)
            update_pct(pct, f"로딩 {len(done_ids)}/{total} — {name}")
        except Exception as e:
            skipped.append({"name": name, "mime": mime, "reason": str(e)})
            update_msg(f"⚠️ 스킵: {name} ({mime}) — {e}")

    if batch:
        try:
            _insert_docs(index, batch)
            _persist_index(index, persist_dir)
        except Exception as e:
            update_msg(f"⚠️ 인덱스 삽입 오류: {e}")

    # 보고서 누적 저장(세션에)
    rep = st.session_state.get("indexing_report", {"total_manifest": total, "loaded_docs": 0, "skipped_count": 0, "skipped": []})
    rep["total_manifest"] = total
    rep["loaded_docs"] = len(done_ids)
    rep["skipped"].extend(skipped)
    rep["skipped_count"] = len(rep["skipped"])
    st.session_state["indexing_report"] = rep

    return {"status": "running", "pct": _progress_pct(persist_dir, total), "msg": "진행 중…"}

def cancel_index_builder(job: dict) -> None:
    """단순히 체크포인트만 제거하면 다음 시작 시 처음부터 혹은 Resume가 깔끔해집니다."""
    persist_dir = job["persist_dir"]
    _clear_ckpt(persist_dir)

def get_index_progress(job: dict) -> Tuple[int, str]:
    """현재 진행률(%)과 간단 메시지."""
    gcp = job["gcp"]
    remote = _fetch_drive_manifest(gcp, job["gdrive_folder_id"], exclude_folder_names=["chat_log"])
    files = remote.get("files", [])
    if job.get("max_docs"):
        files = files[: job["max_docs"]]
    pct = _progress_pct(job["persist_dir"], len(files))
    return pct, f"{pct}%"

# ============================ QA 유틸 =============================
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
