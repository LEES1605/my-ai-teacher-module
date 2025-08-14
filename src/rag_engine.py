# src/rag_engine.py — RAG 유틸(임베딩 1회 + LLM 2개) + 취소/재개 + Google Docs Export 지원
#  - tqdm 콘솔 진행바 억제(TQDM_DISABLE)
#  - Google Docs/Sheets/Slides는 Drive "export"로 텍스트/CSV 변환 후 인덱싱
#  - 첫 실행/깨진 저장소 자동 복구

from __future__ import annotations
import os, json, shutil, re, hashlib, io
from typing import Callable, Any, Mapping, Iterable

# 🔇 tqdm(콘솔 진행바) 억제 — Streamlit Cloud 로그 스팸/워커부하 완화
os.environ.setdefault("TQDM_DISABLE", "1")

import streamlit as st
from src.config import settings

# --- [NEW] 출처 파일명 추출 유틸 -------------------------------
def _source_names_from_nodes(nodes):
    """
    LlamaIndex 응답의 source_nodes에서 안전하게 파일명을 뽑아낸다.
    1) metadata의 다양한 키 시도
    2) file_id만 있을 경우, 로컬 Drive 매니페스트(settings.MANIFEST_PATH)를 활용
    """
    import json, os
    names = set()

    # 1) 로컬 매니페스트 로드(있으면)
    manifest = {}
    try:
        if os.path.exists(settings.MANIFEST_PATH):
            with open(settings.MANIFEST_PATH, "r", encoding="utf-8") as fp:
                manifest = json.load(fp)
    except Exception:
        manifest = {}

    # 2) 노드 메타 파싱
    for n in (nodes or []):
        meta = getattr(n, "metadata", {}) or {}
        # 흔히 쓰이는 후보 키들을 차례로 확인
        for k in ("file_name", "filename", "file", "source", "file_path", "document"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                names.add(v.strip())
                break
        else:
            # 위 키들에 실패했다면 file_id류로 매니페스트 매핑
            fid = meta.get("file_id") or meta.get("id") or meta.get("drive_file_id")
            if isinstance(fid, str) and fid in manifest:
                v = manifest[fid].get("name")
                if isinstance(v, str) and v.strip():
                    names.add(v.strip())

    return ", ".join(sorted(names)) if names else "출처 정보 없음"

# ---------------------------------------------------------------

# (선택) llama_index 로그 억제 — 과도한 디버그 출력 방지
import logging
logging.getLogger("llama_index").setLevel(logging.WARNING)

# Google API
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

# llama_index
from llama_index.core import Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import load_index_from_storage, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

# 취소 신호용 예외
class CancelledError(Exception):
    """사용자가 '취소' 버튼을 눌러 실행을 중단했음을 나타냅니다."""
    pass

# =============================================================================
# 0) 공통: 서비스계정 JSON 정규화 + Drive 서비스
# =============================================================================

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
    scopes = [
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/drive.file",  # export에도 필요
    ]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

# =============================================================================
# 1) 임베딩/LLM 초기화 (임베딩 1회, LLM은 필요 수만큼)
# =============================================================================

def set_embed_provider(provider: str, api_key: str, embed_model: str):
    """임베딩 공급자만 지정해 Settings.embed_model 설정"""
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

# =============================================================================
# 2) Drive 테스트/미리보기
# =============================================================================

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

# =============================================================================
# 3) 매니페스트/체크포인트 + 인덱싱(Resume 지원) + Google Docs Export
# =============================================================================

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

def _manifest_hash(m: dict) -> str:
    """원격 스냅샷을 해시하여 체크포인트 타깃 ID로 사용"""
    s = json.dumps(m, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

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

def _ckpt_path(persist_dir: str) -> str:
    return os.path.join(persist_dir, "build_checkpoint.json")

def _load_ckpt(persist_dir: str) -> dict | None:
    path = _ckpt_path(persist_dir)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            return None
    return None

def _save_ckpt(persist_dir: str, data: dict) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    with open(_ckpt_path(persist_dir), "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2, sort_keys=True)

def _clear_ckpt(persist_dir: str) -> None:
    path = _ckpt_path(persist_dir)
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

@st.cache_resource(show_spinner=False)
def _load_index_from_disk(persist_dir: str):
    """저장된 인덱스를 로드. 실패 시 깨끗한 저장소로 자동 초기화."""
    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)
    except Exception:
        return _ensure_index_initialized(persist_dir)

def _ensure_index_initialized(persist_dir: str):
    """빈 인덱스를 메모리에 만들고, 지정 경로로 최초 persist(필수 파일 생성)."""
    os.makedirs(persist_dir, exist_ok=True)
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents([], storage_context=storage_context)
    index.storage_context.persist(persist_dir=persist_dir)
    return index

def _iter_drive_file_ids(manifest: dict) -> Iterable[str]:
    items = []
    for fid, meta in manifest.items():
        items.append((meta.get("modifiedTime") or "", fid))
    items.sort(reverse=True)
    for _, fid in items:
        yield fid

# === Google Docs/Sheets/Slides Export → 텍스트/CSV =================================

_GOOGLE_APPS = "application/vnd.google-apps."

def _export_text_via_drive(svc, file_id: str, mime_type: str) -> tuple[str | None, str]:
    """
    Google Docs/Sheets/Slides를 텍스트/CSV로 export 후 문자열 반환.
    리턴: (text_or_none, used_mime)
    """
    export_map = {
        _GOOGLE_APPS + "document": "text/plain",      # Docs → txt
        _GOOGLE_APPS + "spreadsheet": "text/csv",     # Sheets → csv
        _GOOGLE_APPS + "presentation": "text/plain",  # Slides → txt (가능한 경우)
    }
    target = export_map.get(mime_type)
    if not target:
        return (None, "")

    try:
        req = svc.files().export_media(fileId=file_id, mimeType=target)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        content = buf.getvalue().decode("utf-8", errors="ignore")
        return (content, target)
    except HttpError as e:
        # Slides에서 text/plain 미지원일 수 있음 → 타겟을 조금 바꿔 시도 (최후 수단)
        if mime_type.endswith("presentation"):
            try:
                alt = "text/csv"
                req = svc.files().export_media(fileId=file_id, mimeType=alt)
                buf = io.BytesIO()
                downloader = MediaIoBaseDownload(buf, req)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                content = buf.getvalue().decode("utf-8", errors="ignore")
                return (content, alt)
            except Exception:
                pass
        # 실패하면 None 반환
        return (None, "")
    except Exception:
        return (None, "")

def _build_or_resume_with_progress(update_pct: Callable[[int, str | None], None],
                                   update_msg: Callable[[str], None],
                                   gdrive_folder_id: str,
                                   gcp_creds: Mapping[str, Any],
                                   persist_dir: str,
                                   manifest: dict,
                                   max_docs: int | None = None,
                                   is_cancelled: Callable[[], bool] | None = None):
    """
    ▶ Resume 지원 빌드
      - 체크포인트에 기록된 파일은 건너뛰고, 나머지 파일만 계속 인덱싱
      - 각 파일 처리 후 persist + 체크포인트 갱신
      - Google Docs/Sheets/Slides는 export로 텍스트/CSV 추출
    """
    from llama_index.readers.google import GoogleDriveReader

    # 0) 리더/서비스 초기화
    update_pct(10, "Drive 리더 초기화")
    try:
        try:
            loader = GoogleDriveReader(service_account_key=gcp_creds)   # 신형
        except TypeError:
            loader = GoogleDriveReader(gcp_creds_dict=gcp_creds)        # 구형
        svc = _build_drive_service(gcp_creds)
    except Exception as e:
        st.error("Google Drive 리더 초기화 실패")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

    # 1) 인덱스 초기화/로드 (없으면 생성 후 persist)
    try:
        if os.path.exists(persist_dir):
            index = _load_index_from_disk(persist_dir)
        else:
            index = _ensure_index_initialized(persist_dir)
    except Exception as e:
        st.error("인덱스 저장소 초기화 실패")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

    # 2) 체크포인트 준비
    target_hash = _manifest_hash(manifest)
    ckpt = _load_ckpt(persist_dir) or {}
    done_ids: set[str] = set(ckpt.get("done_ids", [])) if ckpt.get("target_hash") == target_hash else set()
    if ckpt.get("target_hash") != target_hash:
        ckpt = {"target_hash": target_hash, "done_ids": []}
        _save_ckpt(persist_dir, ckpt)

    # 3) 처리할 ID 목록
    all_ids = list(_iter_drive_file_ids(manifest))
    if max_docs:
        all_ids = all_ids[:max_docs]
    todo_ids = [fid for fid in all_ids if fid not in done_ids]
    total = len(all_ids)

    update_msg(f"체크포인트 로드: 완료 {len(done_ids)}/{total}개 — 재개 준비")

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)

    # 4) 문서별 처리 루프
    for fid in todo_ids:
        if is_cancelled and is_cancelled():
            raise CancelledError("사용자 취소(문서 처리 중)")

        meta = manifest.get(fid, {})
        fname = meta.get("name", fid)
        mime  = meta.get("mimeType", "")
        update_msg(f"문서 처리 중: {fname}")

        docs: list[Document] = []

        # (A) Google Docs/Sheets/Slides → export 로 텍스트/CSV 추출
        if mime.startswith(_GOOGLE_APPS):
            text, used = _export_text_via_drive(svc, fid, mime)
            if text:
                docs = [Document(text=text, metadata={"file_name": fname, "file_id": fid, "mimeType": mime, "exported_as": used})]
            else:
                st.warning(f"Export 실패(건너뜀): {fname} ({mime})")
        # (B) 그 외 바이너리/일반 파일 → LlamaIndex 리더로 시도
        else:
            try:
                docs = loader.load_data(file_ids=[fid])
            except Exception as e:
                st.warning(f"다운로드 실패(건너뜀): {fname} — {e}")

        if not docs:
            continue

        # 노드화 → 인덱스 삽입 → persist → 체크포인트 갱신
        try:
            nodes = splitter.get_nodes_from_documents(docs)
            index.insert_nodes(nodes)
            index.storage_context.persist(persist_dir=persist_dir)  # 부분 저장
            done_ids.add(fid)
            _save_ckpt(persist_dir, {"target_hash": target_hash, "done_ids": sorted(done_ids)})
        except Exception as e:
            st.warning(f"인덱싱 실패({fname}): {e}")
            continue

        # 진행률(30~90 구간 매핑)
        cur_done = len(done_ids)
        pct = 30 + int(60 * (cur_done / total)) if total > 0 else 90
        update_pct(pct, f"진행 {cur_done}/{total} — {fname}")

    # 5) 완료 정리
    update_pct(95, "정리/검증…")
    try:
        index.storage_context.persist(persist_dir=persist_dir)
    except Exception as e:
        st.warning(f"최종 저장 경고: {e}")

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
    """변경 없으면 로딩, 변경 있으면 ▶ 'Resume 체크포인트' 우선 시도 → 없으면 새 빌드"""
    gcp_creds = _normalize_sa(raw_sa)
    if not gcp_creds:
        st.error("서비스계정 JSON 파싱에 실패했습니다.")
        st.stop()

    update_pct(5, "드라이브 변경 확인 중…")
    remote = _fetch_drive_manifest(gcp_creds, gdrive_folder_id, is_cancelled=is_cancelled)
    local = _load_local_manifest(manifest_path)
    target_hash = _manifest_hash(remote)

    # 1) 변경 없으면 빠른 로딩
    if os.path.exists(persist_dir) and not _manifests_differ(local, remote):
        update_pct(25, "변경 없음 → 저장된 두뇌 로딩")
        idx = _load_index_from_disk(persist_dir)
        update_pct(100, "완료!")
        return idx

    # 2) 변경이 있지만, 같은 스냅샷(target_hash)으로 진행 중 체크포인트가 있으면 '재개'
    ckpt = _load_ckpt(persist_dir)
    if ckpt and ckpt.get("target_hash") == target_hash and os.path.exists(persist_dir):
        update_pct(20, "변경 감지 → 미완료 체크포인트 발견 → 재개")
        idx = _build_or_resume_with_progress(
            update_pct, update_msg, gdrive_folder_id, gcp_creds, persist_dir, remote,
            max_docs=max_docs, is_cancelled=is_cancelled
        )
        _save_local_manifest(manifest_path, remote)
        update_pct(100, "완료!")
        return idx

    # 3) 새로 빌드(기존 저장소/체크포인트 제거)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    _clear_ckpt(persist_dir)

    update_pct(25, "변경 감지 → 새 인덱스 생성")
    idx = _build_or_resume_with_progress(
        update_pct, update_msg, gdrive_folder_id, gcp_creds, persist_dir, remote,
        max_docs=max_docs, is_cancelled=is_cancelled
    )

    _save_local_manifest(manifest_path, remote)
    update_pct(100, "완료!")
    return idx

# =============================================================================
# 4) QA 유틸
# =============================================================================

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

        # [개선] 다양한 메타 키 + Drive 매니페스트를 활용해 파일명 복구
        nodes = getattr(response, "source_nodes", []) or []
        source_files = _source_names_from_nodes(nodes)

        return f"{answer_text}\n\n---\n*참고 자료: {source_files}*"

    except Exception as e:
        return f"텍스트 답변 생성 중 오류 발생: {e}"

# === LLM 무검색(직접 완성) 유틸 ======================================
def llm_complete(llm, prompt: str, temperature: float = 0.0) -> str:
    """
    RAG 검색 없이 순수 LLM으로만 결과를 생성.
    llama-index LLM 래퍼 호환 (complete().text 또는 predict())
    """
    try:
        resp = llm.complete(prompt)
        # CompletionResponse(text=...) 형태
        return getattr(resp, "text", str(resp))
    except AttributeError:
        # 일부 구현체는 predict만 제공할 수 있음
        return llm.predict(prompt)
