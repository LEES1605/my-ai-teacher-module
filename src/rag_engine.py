# src/rag_engine.py — 증분 인덱싱(스텝/재개/취소) + prepared 전용 + chat_log 제외
from __future__ import annotations
import os, json, time
from typing import Callable, Any, Mapping, Iterable, List, Dict

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

# LLM 직답(검색 없음)
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
        # 학습 제외 폴더는 재귀 제외
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

def _ensure_documents(docs) -> list:
    """GoogleDriveReader가 반환한 객체가 list[str]/dict일 때도 안전하게 Document로 정규화."""
    from llama_index.core import Document
    if not docs:
        return []
    norm = []
    for d in docs:
        if hasattr(d, "id_") and hasattr(d, "text"):
            norm.append(d)
        elif isinstance(d, dict) and "text" in d:
            norm.append(Document(text=d.get("text",""), metadata=d.get("metadata", {})))
        else:
            norm.append(Document(text=str(d)))
    return norm

def _insert_docs(index, docs):
    """LlamaIndex 버전차 호환 (insert / insert_nodes / insert_documents)"""
    docs = _ensure_documents(docs)
    if not docs:
        return
    try:
        index.insert(docs)
        return
    except Exception:
        pass
    try:
        index.insert_nodes(docs)
        return
    except Exception:
        pass
    try:
        index.insert_documents(docs)
        return
    except Exception as e:
        raise e

# ============================ 매니페스트 비교 ============================
def _manifests_differ(local: dict, remote: dict) -> bool:
    def sig(d: dict) -> tuple:
        return (d.get("id",""), d.get("modifiedTime",""), d.get("size",""), d.get("md5Checksum",""))
    ls = sorted([sig(x) for x in local.get("files", [])])
    rs = sorted([sig(x) for x in remote.get("files", [])])
    return ls != rs

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

# ====================== 스텝 인덱서 (시작/재개/취소) ======================
def start_index_builder(
    update_pct: Callable[[int, str | None], None],
    update_msg: Callable[[str], None],
    gdrive_folder_id: str,
    raw_sa: Any | None,
    persist_dir: str,
    manifest_path: str,
    max_docs: int | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict:
    """변경 없으면 저장본 로드 → 즉시 done. 변경 있으면 job 시작."""
    from llama_index.core import VectorStoreIndex
    from llama_index.readers.google import GoogleDriveReader

    gcp_creds = _normalize_sa(raw_sa)

    # 현재 임베딩 서명
    from llama_index.core import Settings
    cur_sig = {
        "embed_provider": ("openai" if "openai" in str(type(getattr(Settings, "embed_model", None))).lower() else "google"),
        "embed_model": getattr(Settings.embed_model, "model", getattr(Settings.embed_model, "_model_name", "")),
    }
    old_sig = _load_signature(persist_dir)

    update_pct(5, "드라이브 변경 확인 중…")
    # prepared 폴더만 인덱싱, chat_log는 재귀에서 제외
    remote = _fetch_drive_manifest(gcp_creds, gdrive_folder_id, exclude_folder_names=["chat_log"])
    local = _load_local_manifest(manifest_path)

    # 변경 없음 → 저장본 로드
    if os.path.exists(persist_dir) and (old_sig == cur_sig) and (not _manifests_differ(local, remote)):
        update_pct(25, "변경 없음 → 저장된 두뇌 로딩")
        idx = _load_index_from_disk(persist_dir)
        update_pct(100, "완료!")
        st.session_state.setdefault("indexing_report", {
            "total_manifest": len(remote.get("files", [])),
            "loaded_docs": -1, "skipped_count": 0, "skipped": []
        })
        return {"status": "done", "index": idx}

    # 새 빌드 준비
    try:
        index = _load_index_from_disk(persist_dir)
        update_msg("이전 진행분을 불러왔습니다(Resume).")
    except Exception:
        index = VectorStoreIndex.from_documents([])
        _persist_index(index, persist_dir)

    manifest = remote
    files_all = manifest.get("files", [])
    if max_docs:
        files_all = files_all[:max_docs]
    total = len(files_all)

    if total == 0:
        update_pct(100, "폴더에 학습할 파일이 없습니다.")
        st.session_state["indexing_report"] = {
            "total_manifest": 0, "loaded_docs": 0, "skipped_count": 0, "skipped": []
        }
        return {"status": "done", "index": index}

    ckpt = _load_ckpt(persist_dir)
    done_ids: set[str] = set(ckpt.get("done_ids", []))
    pending = [f for f in files_all if f["id"] not in done_ids]

    job = {
        "persist_dir": persist_dir,
        "manifest_path": manifest_path,
        "signature": cur_sig,
        "manifest_remote": manifest,
        "reader": GoogleDriveReader(service_account_key=gcp_creds, recursive=False),
        "index": index,
        "pending": pending,
        "done_ids": done_ids,
        "skipped": [],
        "total": total,
        "processed": len(done_ids),
        "last_pct": 8,
    }
    update_pct(8, "인덱싱 준비 중…")
    return {"status": "running", "job": job, "pct": 8, "msg": "준비 중…"}

def resume_index_builder(
    job: dict,
    update_pct: Callable[[int, str | None], None],
    update_msg: Callable[[str], None],
    is_cancelled: Callable[[], bool] | None = None,
    batch_size: int = 6,
) -> dict:
    """파일 몇 개씩 처리하고 돌아옴 → UI에서 st.rerun으로 스텝 진행."""
    reader = job["reader"]
    index = job["index"]
    pending = job["pending"]
    done_ids = job["done_ids"]
    skipped = job["skipped"]
    total = job["total"]
    processed = job["processed"]
    persist_dir = job["persist_dir"]

    if is_cancelled and is_cancelled():
        return {"status": "cancelled"}

    # 한 스텝(batch_size) 만큼 처리
    for _ in range(min(batch_size, len(pending))):
        if is_cancelled and is_cancelled():
            return {"status": "cancelled"}
        f = pending.pop(0)
        fid, name, mime = f["id"], f.get("name",""), f.get("mimeType","")
        try:
            docs = reader.load_data(file_ids=[fid])
            docs = _ensure_documents(docs)
            # 파일명 메타
            for d in docs:
                try:
                    d.metadata["file_name"] = name
                except Exception:
                    pass
            _insert_docs(index, docs)

            processed += 1
            done_ids.add(fid)
            _save_ckpt(persist_dir, {"done_ids": list(done_ids)})

            pct = 8 + int(80 * processed / max(total, 1))  # 8→88
            update_pct(pct, f"로딩 {processed}/{total} — {name}")
            job["last_pct"] = pct
            time.sleep(0.01)
        except Exception as e:
            skipped.append({"name": name, "mime": mime, "reason": str(e)})
            update_msg(f"⚠️ 스킵: {name} ({mime}) — {e}")

    # 중간 저장
    _persist_index(index, persist_dir)
    job["processed"] = processed

    if not pending:
        # 마무리
        update_pct(92, "인덱스 저장 중…")
        _persist_index(index, persist_dir)
        _clear_ckpt(persist_dir)
        _save_signature(persist_dir, job["signature"])
        _save_local_manifest(job["manifest_path"], job["manifest_remote"])
        update_pct(100, "완료")

        st.session_state["indexing_report"] = {
            "total_manifest": total,
            "loaded_docs": len(done_ids),
            "skipped_count": len(skipped),
            "skipped": skipped,
        }
        return {"status": "done", "index": index}

    return {
        "status": "running",
        "pct": job.get("last_pct", 8),
        "msg": f"진행 중… ({processed}/{total})"
    }

def cancel_index_builder(job: dict) -> None:
    # 현재는 세션 내 작업을 중단 표기만. (외부 스레드 없음)
    job["pending"].clear()

# ============================ QA 유틸 =============================
def get_text_answer(query_engine, question: str, system_prompt: str) -> str:
    """
    원칙: 업로드 자료를 최우선으로 검색하되, 충분한 근거가 없을 때는
    일반 지식을 보완적으로 사용(그 사실을 답변에 명시).
    """
    try:
        full_query = (
            f"{system_prompt}\n\n"
            "[지시사항]\n"
            "1) 업로드된 강의/교재 자료를 최우선으로 찾아 근거와 함께 답해라.\n"
            "2) 자료에서 충분한 근거가 없으면, 일반 지식으로 보완하되 '자료 근거 부족'을 명시해라.\n"
            "3) 최종에 '참고 자료'로 파일명을 나열해라.\n\n"
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
