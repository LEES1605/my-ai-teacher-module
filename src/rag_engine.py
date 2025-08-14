# src/rag_engine.py
# =============================================================================
# Google Drive 연결 유틸 (Streamlit)
# - 서비스계정 JSON(st.secrets) 로드
# - 스모크 테스트(smoke_test_drive)
# - 폴더 내 파일 프리뷰(preview_drive_files)
#
# 안전장치
# - 모든 설명/가이드는 트리플 따옴표로 작성 (줄바꿈 안전)
# - 다양한 시크릿 키 이름 후보를 순서대로 검사
# - Shared Drives(공유 드라이브) 옵션 켜서 기업/팀 드라이브 대응
# - st.cache_data(ttl=60) 로 1분 캐시 (과다 호출 방지)
# =============================================================================

from __future__ import annotations

import json
import typing as t

import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# ---- 권한 스코프(읽기전용) ----------------------------------------------------
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# ---- 지원하는 서비스계정 JSON 시크릿 키 후보 -----------------------------------
SECRET_KEY_CANDIDATES: list[str] = [
    # 권장 이름부터 우선
    "GDRIVE_SERVICE_ACCOUNT_JSON",
    # 기존에 쓰던 이름들(하위호환)
    "GOOGLE_SERVICE_ACCOUNT_JSON",
    "google_service_account_json",
    "SERVICE_ACCOUNT_JSON",
]

# ---- 내부 유틸 ---------------------------------------------------------------

def _pick_raw_service_account_json() -> t.Union[str, dict]:
    """
    st.secrets 에서 서비스계정 JSON을 찾아 원본(raw)을 돌려준다.
    - 문자열(JSON) 또는 dict 모두 허용
    - 후보 키들을 위에서 아래 순서로 탐색
    """
    for key in SECRET_KEY_CANDIDATES:
        if key in st.secrets and str(st.secrets[key]).strip():
            return st.secrets[key]
    raise KeyError(
        """
        서비스계정 JSON 비밀키를 찾지 못했습니다.

        지원 키 이름(위쪽이 우선):
          - GDRIVE_SERVICE_ACCOUNT_JSON (권장)
          - GOOGLE_SERVICE_ACCOUNT_JSON
          - google_service_account_json
          - SERVICE_ACCOUNT_JSON

        Streamlit Cloud의 [Settings → Secrets] 에 위 키들 중 하나로
        서비스계정 JSON 전체 내용을 넣어주세요.
        """
    )


def _load_creds() -> Credentials:
    """
    서비스계정 JSON(raw)로부터 google.oauth2 Credentials 객체를 생성한다.
    """
    raw = _pick_raw_service_account_json()
    info = json.loads(raw) if isinstance(raw, str) else dict(raw)
    return Credentials.from_service_account_info(info, scopes=SCOPES)


def _get_folder_id(explicit_folder_id: str | None = None) -> str:
    """
    인자로 folder_id를 받으면 그것을, 아니면 st.secrets['GDRIVE_FOLDER_ID'] 를 사용.
    없으면 빈 문자열 반환.
    """
    if explicit_folder_id and str(explicit_folder_id).strip():
        return str(explicit_folder_id).strip()
    return str(st.secrets.get("GDRIVE_FOLDER_ID", "")).strip()


def _build_drive_service(creds: Credentials):
    """
    구글 드라이브 API 서비스 객체 생성.
    cache_discovery 인자는 버전마다 다를 수 있으니 안전하게 생략.
    """
    return build("drive", "v3", credentials=creds)


def _format_rows(files: list[dict]) -> list[dict]:
    """
    files() 응답에서 화면/테이블 표시에 적합한 최소 컬럼만 정리.
    """
    return [
        {
            "name": f.get("name"),
            "mime": f.get("mimeType"),
            "modified": f.get("modifiedTime"),
            "link": f.get("webViewLink"),
        }
        for f in files or []
    ]


# ---- 공개 함수(앱에서 import 해서 사용) --------------------------------------

def smoke_test_drive(folder_id: str | None = None) -> tuple[bool, str]:
    """
    구글 드라이브 연결 스모크 테스트.

    동작:
      1) 서비스계정 JSON이 올바른지 확인
      2) (선택) GDRIVE_FOLDER_ID 또는 인자로 받은 folder_id 아래에서
         파일 1개를 조회해 API 호출이 실제로 되는지 확인

    반환:
      (ok: bool, message: str)
    """
    try:
        creds = _load_creds()
        svc = _build_drive_service(creds)
    except Exception as e:
        return False, f"자격증명/서비스 생성 실패: {e}"

    # 서비스계정 이메일은 Credentials에 노출됨
    email = getattr(creds, "service_account_email", "(알 수 없음)")

    fid = _get_folder_id(folder_id)
    if not fid:
        # 폴더 없이도 최소 생성 테스트는 통과시켜 유효성 제공
        return True, f"서비스계정 키 OK (email: {email}). GDRIVE_FOLDER_ID가 없어 목록 조회는 생략했습니다."

    try:
        resp = svc.files().list(
            q=f"'{fid}' in parents and trashed=false",
            fields="files(id)",
            pageSize=1,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            corpora="allDrives",
        ).execute()
        cnt = len(resp.get("files", []))
        return True, f"연결 OK (email: {email}). 폴더 접근 및 목록 조회 성공(샘플 {cnt}개)."
    except Exception as e:
        return False, f"드라이브 API 호출 실패(폴더 접근 문제 가능): {e}"


def preview_drive_files(
    folder_id: str | None = None,
    max_items: int = 10,
) -> tuple[bool, str, list[dict]]:
    """
    지정 폴더의 최근 파일 프리뷰(최대 max_items개)를 가져온다.

    반환:
      (ok: bool, message: str, rows: list[dict])

    rows 예시 원소:
      {"name": "...", "mime": "...", "modified": "ISO8601", "link": "..."}
    """
    try:
        # 폴더 ID 준비
        fid = _get_folder_id(folder_id)
        if not fid:
            return False, "Secrets에 GDRIVE_FOLDER_ID가 없습니다.", []

        # 캐시 키에 서비스계정 이메일을 섞어, 키 교체시 자동 무효화
        creds = _load_creds()
        email = getattr(creds, "service_account_email", "unknown@serviceaccount")

        @st.cache_data(ttl=60)
        def _list_files_cached(fid: str, max_items: int, svc_email: str) -> list[dict]:
            """
            60초 캐시된 파일 목록 조회.
            캐시 함수 내부에서 service를 매번 생성해도 비용이 작고 안전함.
            """
            service = _build_drive_service(_load_creds())
            resp = service.files().list(
                q=f"'{fid}' in parents and trashed=false",
                orderBy="modifiedTime desc",
                fields="files(id,name,mimeType,modifiedTime,webViewLink)",
                pageSize=max_items,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                corpora="allDrives",
            ).execute()
            return resp.get("files", [])

        files = _list_files_cached(fid, max_items, email)
        rows = _format_rows(files)

        if not rows:
            msg = (
                "폴더에 파일이 없거나 접근 권한이 없습니다.\n"
                "→ 1) 폴더에 테스트 파일을 하나 넣어보세요.\n"
                "→ 2) 해당 폴더를 서비스계정 이메일과 공유했는지 확인하세요.\n"
                "     (Google Drive에서 '공유' → 사용자 초대에 서비스계정 이메일 추가)\n"
                "→ 3) 공유 드라이브(팀 드라이브)라면, 접근 권한과 조직 정책을 확인하세요."
            )
            return True, msg, rows

        return True, f"최근 파일 {len(rows)}건.", rows

    except Exception as e:
        return False, f"목록 조회 중 오류: {e}", []


# ---- (옵션) 간단 데모용 진입점 -------------------------------------------------
if __name__ == "__main__":
    """
    로컬에서 간단 실행해 상태를 점검할 때:
      $ python -m src.rag_engine
    (단, 이 방법은 Streamlit st.secrets가 없어 실패할 수 있습니다.)
    """
    ok, msg = smoke_test_drive()
    print("[smoke_test_drive]", ok, msg)
    ok, msg, rows = preview_drive_files(max_items=5)
    print("[preview_drive_files]", ok, msg)
    for r in rows:
        print(" -", r["modified"], r["name"], r["mime"])
# ===== END OF FILE ============================================================
