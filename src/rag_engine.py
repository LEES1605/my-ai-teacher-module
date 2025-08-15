# ============================ Step-wise Indexer =============================
# 긴 인덱싱을 여러 번의 짧은 스텝으로 나눠서 실행하기 위한 헬퍼입니다.
# - 각 스텝은 최대 N개 파일만 처리하고 즉시 반환
# - 다음 틱에서 다시 step()을 호출하면 이어서 진행
# - cancel 플래그는 다음 스텝에서 즉시 반영
# - 파일 로딩이 오래 걸릴 때를 대비해 per-file timeout 지원

from queue import Queue
import threading
import time

def _load_with_timeout(reader, fid: str, timeout_s: int = 40):
    """GoogleDriveReader.load_data(file_ids=[fid])를 쓰되, timeout을 건 래퍼."""
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
        # 스레드가 끝나지 않으면 timeout 처리
        return None, TimeoutError(f"load_data timeout after {timeout_s}s")
    if not err.empty():
        raise err.get()
    return q.get(), None

class IndexBuilder:
    """증분(step) 인덱서를 캡슐화한 객체. Streamlit session_state에 그대로 보관해도 됨."""
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

        # 이번 스텝에서 처리할 목록
        pend = self.pending_ids
        if not pend:
            # 마무리
            if self.batch:
                _insert_docs(self.index, self.batch)
                self.batch.clear()
            _persist_index(self.index, self.persist_dir)
            _clear_ckpt(self.persist_dir)
            if on_pct: on_pct(100, "완료")
            return "done"

        work = pend[:max_files]
        for fid in work:
            if is_cancelled and is_cancelled():
                raise CancelledError("사용자 취소(스텝 중)")

            # 진행 메시지/퍼센트
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
                    try: d.metadata["file_name"] = name
                    except Exception: pass
                self.batch.extend(docs)

                if len(self.batch) >= self.BATCH_PERSIST:
                    _insert_docs(self.index, self.batch)
                    _persist_index(self.index, self.persist_dir)
                    self.batch.clear()

                self.done_ids.add(fid)
                _save_ckpt(self.persist_dir, {"done_ids": list(self.done_ids)})

            except Exception as e:
                msg = f"{name} — {e}"
                self.skipped.append({"name": name, "reason": str(e)})
                if on_msg: on_msg("⚠️ 스킵: " + msg)

            # 스텝 사이에도 취소 확인
            if is_cancelled and is_cancelled():
                raise CancelledError("사용자 취소(스텝 사이)")

        # 스텝 종료 시점에 진행률 업데이트
        pct = 8 + int(80 * (self.processed / max(self.total, 1)))
        if on_pct: on_pct(pct, f"로딩 {self.processed}/{self.total}")

        # 아직 남았으면 running
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
