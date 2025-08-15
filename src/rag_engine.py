# src/rag_engine.py — 증분 인덱싱(스텝/취소/재개) + 체크포인트 + chat_log 제외 + 서명검사
from __future__ import annotations
import os, json, time
from typing import Callable, Any, Mapping, Iterable, Optional

import streamlit as st
from src.config import settings

# ================================ 예외 ================================
class CancelledError(Exception):
    """사용자 취소 등으로 중단"""
    pass

# ============================ 임베딩/LLM =============================
def set_embed_provider(provider: str, api_key: str, model: str) -> None:
    """llama_index 전역 Settings.embed_model 지정"""
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

def _insert_docs(index, docs):
    """LlamaIndex 버전차 호환 (insert / insert_documents)"""
    try:
        index.insert(docs)
    except Exception:
        index.insert_documents(docs)

# ============================ 전체 빌드(단발) =============================
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
                try: d.metadata["file_name"] = name
                except Exception: pass
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
            msg = f"{name} ({mime}) — {e}"
            skipped.append({"name": name, "mime": mime, "reason": str(e)})
            update_msg("⚠️ 스킵: " + msg)

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
    gcp_creds = _normalize_sa(raw_sa)

    # 현재 임베딩 서명
    from llama_index.core import Settings
    cur_sig = {
        "embed_provider": ("openai" if "openai" in str(type(getattr(Settings, "embed_model", None))).lower() else "google"),
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

# ============================ 스텝 빌더 ==============================
# app.py가 사용하는 API:
# - start_index_builder(...)
# - resume_index_builder()  → {"done": bool, "bursts":[{"pct":int,"msg":str}, ...], "index": obj|None}
# - cancel_index_builder()
# - get_index_progress()    → 마지막 pct/msg 조회

_BUILDER: dict[str, Any] = {
    "gen": None, "index": None, "done": False, "bursts": [],
    "last_pct": 0, "last_msg": "", "error": None, "cancel": False, "params": None,
}

def _step_generator(params: dict):
    """진행 이벤트(pct,msg)를 yield 하다가 완료 시 return index"""
    from llama_index.core import VectorStoreIndex
    from llama_index.readers.google import GoogleDriveReader

    # unpack
    gdrive_folder_id = params["gdrive_folder_id"]
    gcp_creds = _normalize_sa(params["raw_sa"])
    persist_dir = params["persist_dir"]
    manifest_path = params["manifest_path"]
    max_docs = params.get("max_docs")
    is_cancelled = params.get("is_cancelled")
    exclude_folder_names = ["chat_log"]

    # 임베딩 서명
    from llama_index.core import Settings
    cur_sig = {
        "embed_provider": ("openai" if "openai" in str(type(getattr(Settings, "embed_model", None))).lower() else "google"),
        "embed_model": getattr(Settings.embed_model, "model", getattr(Settings.embed_model, "_model_name", "")),
    }
    old_sig = _load_signature(persist_dir)

    # 매니페스트 비교
    yield {"pct": 5, "msg": "드라이브 변경 확인 중…"}
    remote = _fetch_drive_manifest(gcp_creds, gdrive_folder_id, exclude_folder_names=exclude_folder_names)
    local = _load_local_manifest(manifest_path)
    need_rebuild = (old_sig != cur_sig) or _manifests_differ(local, remote)

    # 변경 없으면 저장본 로드
    if os.path.exists(persist_dir) and not need_rebuild:
        yield {"pct": 25, "msg": "변경 없음 → 저장된 두뇌 로딩"}
        idx = _load_index_from_disk(persist_dir)
        yield {"pct": 100, "msg": "완료!"}
        st.session_state.setdefault("indexing_report", {
            "total_manifest": len(remote.get("files", [])),
            "loaded_docs": -1, "skipped_count": 0, "skipped": []
        })
        return idx

    # 새로/증분 빌드
    # 0) 기존 인덱스 로드 or 빈 인덱스
    try:
        index = _load_index_from_disk(persist_dir)
        yield {"pct": 7, "msg": "이전 진행분을 불러왔습니다(Resume)."}
    except Exception:
        index = VectorStoreIndex.from_documents([])
        _persist_index(index, persist_dir)

    # 1) 파일 목록
    yield {"pct": 8, "msg": "Drive 파일 목록 불러오는 중…"}
    files_all = remote.get("files", [])
    if max_docs:
        files_all = files_all[:max_docs]
    total = len(files_all)
    if total == 0:
        st.session_state["indexing_report"] = {
            "total_manifest": 0, "loaded_docs": 0, "skipped_count": 0, "skipped": []
        }
        yield {"pct": 100, "msg": "폴더에 학습할 파일이 없습니다."}
        return index

    ckpt = _load_ckpt(persist_dir)
    done_ids: set[str] = set(ckpt.get("done_ids", []))
    pending = [f for f in files_all if f["id"] not in done_ids]
    done_count = len(done_ids)

    reader = GoogleDriveReader(service_account_key=gcp_creds, recursive=False)
    batch, BATCH_PERSIST = [], 8
    skipped = []

    def _maybe_cancel():
        if _BUILDER.get("cancel"):  # 외부 취소
            raise CancelledError("사용자 취소(버튼)")
        if callable(is_cancelled) and is_cancelled():  # 앱 측 콜백
            raise CancelledError("사용자 취소(콜백)")

    for i, f in enumerate(pending, start=1):
        _maybe_cancel()
        fid, name, mime = f["id"], f.get("name",""), f.get("mimeType","")
        try:
            docs = reader.load_data(file_ids=[fid])
            for d in docs:
                try: d.metadata["file_name"] = name
                except Exception: pass
            batch.extend(docs)

            if len(batch) >= BATCH_PERSIST:
                _insert_docs(index, batch)
                _persist_index(index, persist_dir)
                batch.clear()

            done_ids.add(fid)
            _save_ckpt(persist_dir, {"done_ids": list(done_ids)})

            processed = done_count + i
            pct = 8 + int(80 * processed / max(total, 1))  # 8→88
            yield {"pct": pct, "msg": f"로딩 {processed}/{total} — {name}"}
        except Exception as e:
            skipped.append({"name": name, "mime": mime, "reason": str(e)})
            yield {"pct": _BUILDER.get("last_pct", 8), "msg": f"⚠️ 스킵: {name} ({mime}) — {e}"}

        # 한 파일마다 잠깐 양보 → UI 반응성
        time.sleep(0.01)

    if batch:
        _insert_docs(index, batch)
    yield {"pct": 92, "msg": "인덱스 저장 중…"}
    _persist_index(index, persist_dir)

    _clear_ckpt(persist_dir)
    yield {"pct": 98, "msg": "메타데이터 저장 중…"}
    _save_local_manifest(manifest_path, remote)
    _save_signature(persist_dir, cur_sig)
    st.session_state["indexing_report"] = {
        "total_manifest": total,
        "loaded_docs": len(done_ids),
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
    yield {"pct": 100, "msg": "완료!"}
    return index

def start_index_builder(
    update_pct: Callable[[int, str | None], None],
    update_msg: Callable[[str], None],
    gdrive_folder_id: str,
    raw_sa: Any | None,
    persist_dir: str,
    manifest_path: str,
    max_docs: int | None = None,
    is_cancelled: Callable[[], bool] | None = None,
):
    """스텝 빌더 시작(또는 재시작)"""
    _BUILDER.update({
        "gen": None, "index": None, "done": False, "bursts": [],
        "last_pct": 0, "last_msg": "", "error": None, "cancel": False,
        "params": {
            "gdrive_folder_id": gdrive_folder_id,
            "raw_sa": raw_sa,
            "persist_dir": persist_dir,
            "manifest_path": manifest_path,
            "max_docs": max_docs,
            "is_cancelled": is_cancelled,
            "update_pct": update_pct,
            "update_msg": update_msg,
        },
    })
    _BUILDER["gen"] = _step_generator(_BUILDER["params"])

def resume_index_builder(steps: int = 10) -> dict:
    """지정 횟수만큼 진행 이벤트를 소화하고 현재 상태 리턴"""
    bursts = []
    if _BUILDER["done"]:
        return {"done": True, "bursts": bursts, "index": _BUILDER["index"]}

    gen = _BUILDER.get("gen")
    if gen is None:
        # 안전장치: start가 안된 경우
        start_index_builder(**_BUILDER.get("params", {}))
        gen = _BUILDER["gen"]

    try:
        for _ in range(max(1, steps)):
            ev = next(gen)
            pct, msg = int(ev.get("pct", _BUILDER["last_pct"])), ev.get("msg")
            _BUILDER["last_pct"], _BUILDER["last_msg"] = pct, (msg or _BUILDER["last_msg"])
            bursts.append({"pct": pct, "msg": msg})
    except StopIteration as e:
        _BUILDER["index"] = getattr(e, "value", None)
        _BUILDER["done"] = True
    except CancelledError as e:
        _BUILDER["done"] = True
        _BUILDER["error"] = str(e)
        bursts.append({"pct": _BUILDER["last_pct"], "msg": f"사용자 취소: {e}"})
    except Exception as e:
        _BUILDER["done"] = True
        _BUILDER["error"] = str(e)
        bursts.append({"pct": _BUILDER["last_pct"], "msg": f"오류: {e}"})

    _BUILDER["bursts"] = bursts
    return {"done": _BUILDER["done"], "bursts": bursts, "index": _BUILDER.get("index")}

def cancel_index_builder() -> None:
    """외부 취소 플래그"""
    _BUILDER["cancel"] = True

def get_index_progress() -> dict:
    """마지막 pct/msg/오류 조회"""
    return {
        "pct": _BUILDER.get("last_pct", 0),
        "msg": _BUILDER.get("last_msg", ""),
        "done": _BUILDER.get("done", False),
        "error": _BUILDER.get("error"),
    }

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
