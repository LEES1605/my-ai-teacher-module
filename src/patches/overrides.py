# ===== [P01] PATCH OVERRIDES ==================================================
# 자주 바뀌는 것들은 전부 여기서! (프리셋/도움말/스텝퍼/플래그/훅)

from dataclasses import dataclass

# ---- 기능 플래그 --------------------------------------------------------------
FEATURE_FLAGS = {
    "SHOW_DRIVE_CARD": True,        # 드라이브 폴더/연결 카드 표시
    "AUTO_BACKUP_ON_SUCCESS": True, # 인덱싱 끝나면 자동 백업
}

# ---- response_mode 도움말 -----------------------------------------------------
RESPONSE_MODE_HELP = (
    "compact: 빠름/경제적 · "
    "refine: 초안→보강(정확성↑) · "
    "tree_summarize: 다문서 종합/장문 요약"
)

# ---- 최적화 프리셋 ------------------------------------------------------------
# cs: chunk size, co: overlap, mc: min chars, dd: dedup, slt: skip low text, psu: pre-summarize
PRESETS = {
    "⚡ 속도 우선": dict(cs=1600, co=40,  mc=80,  dd=True, slt=True, psu=False),
    "🔁 균형":     dict(cs=1024, co=80,  mc=120, dd=True, slt=True, psu=False),
    "🔎 품질 우선": dict(cs=800,  co=120, mc=200, dd=True, slt=True, psu=True),
}

# ---- 스텝퍼 라벨/키워드 매핑 ---------------------------------------------------
STEPPER_LABELS = {
    "check": "드라이브 변경 확인",
    "init":  "Drive 리더 초기화",
    "list":  "문서 목록 불러오는 중",
    "index": "인덱스 생성",
    "save":  "두뇌 저장",
}

# 로그 문구 → 어떤 스텝으로 인식할지(국/영 혼합 키워드)
STEPPER_KEYWORDS = {
    "check": ["변경 확인", "change check", "drive change", "check"],
    "init":  ["리더 초기화", "reader init", "initialize", "init", "인증", "credential", "service"],
    "list":  ["목록", "list", "files", "file list", "manifest", "매니페스트", "로드", "불러오"],
    "index": ["인덱스", "index", "chunk", "청크", "embed", "임베", "build", "vector", "persisting"],
    "save":  ["저장", "save", "persist", "write", "백업", "backup", "upload"],
}
STEPPER_DONE = ["완료", "done", "finish", "finished", "success"]

# ---- 상태키(세션 키) ----------------------------------------------------------
@dataclass(frozen=True)
class STATE_KEYS:
    GDRIVE_FOLDER_ID: str = "_gdrive_folder_id"
    BUILD_PAUSED: str = "build_paused"
    STOP_REQUESTED: str = "stop_requested"

# ---- 훅 (필요할 때 오버라이드) ------------------------------------------------
def on_after_backup(file_name: str) -> None:
    """자동 백업 완료 후 후처리 훅(예: 로깅/슬랙 알림 등)."""
    pass

def on_after_finish() -> None:
    """전체 빌드 완료 후 후처리 훅."""
    pass
