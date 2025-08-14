# src/rag_engine.py — RAG 유틸(임베딩 1회 + LLM 2개) + 취소 지원 + 재개(Resume) 체크포인트
#                     콘솔 진행바 억제(TQDM_DISABLE), 부분 persist, 변경 감지(매니페스트)

from __future__ import annotations
import os, json, shutil, re, hashlib
from typing import Callable, Any, Mapping, Iterable

# 🔇 tqdm(콘솔 진행바) 억제 — Streamlit Cloud 로그 스팸/워커부하 완화
os.environ.setdefault("TQDM_DISABLE", "1")

import streamlit as st
from src.config import settings

# (선택) llama_index 로그 억제 — 과도한 디버그 출력 방지
import logging
logging.getLogger("llama_index").setLevel(logging.WARNING)

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
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

# =============================================================================
# 1) 임베딩/LLM 초기화 (임베딩 1회, LLM은 필요 수만큼)
# =============================================================================

def set_embed_provider(provider: str, api_key: str, embed_model: str):
    """임베딩 공급자만 지정해 Settings.embed_model 설정"""
    from llama_index.core import Settings
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
# 3) 매니페스트/체크포인트 + 인덱싱(Resume 지원)
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
    from llama_index.core import StorageContext, load_index_from_storage
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    return load_index_from_storage(storage_context)

def _iter_drive_file_ids(manifest: dict) -> Iterable[str]:
    """파일 ID를 최신순으로 정렬해서 반환(큰 파일/최근 파일부터 처리 효과)"""
    items = []
    for fid, meta in manifest.items():
        items.append((meta.get("modifiedTime") or "", fid))
    items.sort(reverse=True)
    for _, fid in items:
        yield fid

def _load_one_document_by_id(loader, file_id: str) -> list:
    """GoogleDriveReader에서 특정 파일만 로드 (리턴: Document 리스트)"""
    # API를 여러 번 치더라도 재개를 위해 '한 파일씩' 안전하게 처리
    return loader.load_data(file_ids=[file_id])

def _ensure_index_initialized(persist_dir: str):
    """저장소가 없으면 빈 인덱스를 초기화해 둡니다 (이후 insert_nodes 사용)"""
    from llama_index.core import StorageContext, VectorStoreIndex
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    # 빈 인덱스를 일단 만들어 구조를 초기화
    return VectorStoreIndex.from_documents([], storage_context=storage_context)

def _build_or_resume_with_progress(update_pct: Callable[[int, str | None], None],
                                   update_msg: Callable[[str], None],
                                   gdrive_folder_id: str,
                                   gcp_creds: Mapping[str, Any],
                                   persist_dir: str,
                                   manifest: dict,
                                   max_docs: int | None = None,
                                   is_cancelled: Callable[[], bool] | None = None):
    """
    ▶ 핵심: Resume 지원 빌드
      - 체크포인트에 기록된 파일은 건너뛰고, 나머지 파일만 계속 인덱싱
      - 각 파일 처리 후 persist + 체크포인트 갱신
    """
    from llama_index.readers.google import GoogleDriveReader
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core import VectorStoreIndex

    # 0) 리더 초기화
    update_pct(10, "Drive 리더 초기화")
    try:
        try:
            loader = GoogleDriveReader(service_account_key=gcp_creds)   # 신형
        except TypeError:
            loader = GoogleDriveReader(gcp_creds_dict=gcp_creds)        # 구형
    except Exception as e:
        st.error("Google Drive 리더 초기화 실패")
        with st.expander("자세한 오류 보기", expanded=True):
            st.exception(e)
        st.stop()

    # 1) 인덱스 초기화/로드
    #    - 저장소가 있다면 로드, 없으면 빈 인덱스 생성
    try:
        if os.path.exists(persist_dir):
            try:
                index = _load_index_from_disk(persist_dir)
            except Exception:
                index = _ensure_index_initialized(persist_dir)
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
        # 다른 스냅샷이면 새로운 타깃으로 초기화
        ckpt = {"target_hash": target_hash, "done_ids": []}
        _save_ckpt(persist_dir, ckpt)

    # 3) 처리할 ID 목록
    all_ids = list(_iter_drive_file_ids(manifest))
    if max_docs:
        all_ids = all_ids[:max_docs]
    todo_ids = [fid for fid in all_ids if fid not in done_ids]
    total = len(all_ids)
    done = len(done_ids)

    update_msg(f"체크포인트 로드: 완료 {done}/{total}개 — 재개 준비")
    # 진행률은 30→90 구간을 문서 처리 비율로 매핑
    def _progress_for(i_done: int) -> int:
        if total == 0:
            return 90
        frac = min(1.0, max(0.0, i_done / total))
        return 30 + int(60 * frac)

    # 4) 문서별 처리 루프
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
    for i, fid in enumerate(todo_ids, start=1):
        if is_cancelled and is_cancelled():
            raise CancelledError("사용자 취소(문서 처리 중)")

        meta = manifest.get(fid, {})
        fname = meta.get("name", fid)
        update_msg(f"문서 처리 중: {fname}")

        try:
            docs = _load_one_document_by_id(loader, fid)
        except Exception as e:
            st.warning(f"문서 로드 실패({fname}): {e}")
            # 실패해도 다음 문서 진행
            continue

        # 노드화 → 인덱스 삽입 → persist → 체크포인트 갱신
        try:
            nodes = splitter.get_nodes_from_documents(docs)
            index.insert_nodes(nodes)
            # 부분 persist (중단돼도 지금까지는 저장됨)
            index.storage_context.persist(persist_dir=persist_dir)
            # 체크포인트 갱신
            done_ids.add(fid)
            _save_ckpt(persist_dir, {"target_hash": target_hash, "done_ids": sorted(done_ids)})
        except Exception as e:
            st.warning(f"인덱싱 실패({fname}): {e}")
            continue

        # 진행률 업데이트(단조 증가)
        cur_done = len(done_ids)
        update_pct(_progress_for(cur_done), f"진행 {cur_done}/{total} — {fname}")

    # 5) 완료 정리
    update_pct(95, "정리/검증…")
    # 마지막 한 번 더 persist
    try:
        index.storage_context.persist(persist_dir=persist_dir)
    except Exception as e:
        st.warning(f"최종 저장 경고: {e}")

    # 체크포인트는 유지해도 되지만, 성공적으로 끝났다면 지워도 OK
    # (선택) 깨끗한 상태를 원하면 주석 해제:
    # _clear_ckpt(persist_dir)

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

    # 2) 변경이 있지만, 같은 스냅샷(target_hash)으로 진행 중이던 체크포인트가 있으면 '재개'
    ckpt = _load_ckpt(persist_dir)
    if ckpt and ckpt.get("target_hash") == target_hash and os.path.exists(persist_dir):
        update_pct(20, "변경 감지 → 미완료 체크포인트 발견 → 재개")
        idx = _build_or_resume_with_progress(
            update_pct, update_msg, gdrive_folder_id, gcp_creds, persist_dir, remote,
            max_docs=max_docs, is_cancelled=is_cancelled
        )
        # 원격 스냅샷을 로컬 매니페스트로 저장(완료 시점)
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
        try:
            files = [n.metadata.get("file_name", "알 수 없음")
                     for n in getattr(response, "source_nodes", [])]
            source_files = ", ".join(sorted(set(files))) if files else "출처 정보 없음"
        except Exception:
            source_files = "출처 정보 없음"

        return f"{answer_text}\n\n---\n*참고 자료: {source_files}*"
    except Exception as e:
        return f"텍스트 답변 생성 중 오류 발생: {e}"
