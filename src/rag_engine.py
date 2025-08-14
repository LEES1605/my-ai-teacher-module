# src/rag_engine.py
import json
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def _load_creds():
    """
    st.secrets 안에서 서비스계정 JSON을 찾아 자격증명 객체를 만든다.
    가능한 키 이름을 폭넓게 지원한다.
    """
    candidates = [
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "google_service_account_json",
        "SERVICE_ACCOUNT_JSON",
    ]
    raw = None
    for k in candidates:
        if k in st.secrets and str(st.secrets[k]).strip():
            raw = st.secrets[k]
            break
    if raw is None:
        raise KeyError(
            "서비스계정 JSON 비밀키가 없습니다. "
            "예: st.secrets['GOOGLE_SERVICE_ACCOUNT_JSON']"
        )

    info = json.loads(raw) if isinstance(raw, str) else dict(raw)
    return Credentials.from_service_account_info(info, scopes=SCOPES)

def smoke_test_drive():
    """
    서비스계정 JSON이 정상인지와 서비스계정 이메일을 알려준다.
    """
    try:
        creds = _load_creds()
        # 간단 생성만 해도 형식 검증 됨
        _ = build("drive", "v3", credentials=creds)
        return True, f"서비스 계정 키 형식 OK (email: {creds.service_account_email})"
    except Exception as e:
        return False, f"서비스 계정 키 오류: {e}"

def preview_drive_files(max_items=10):
    """
    GDRIVE_FOLDER_ID 아래 파일을 최대 max_items개 나열.
    - 공유 드라이브도 지원(ALL DRIVES)
    - 오류를 친절한 한국어 메시지로 반환
    """
    try:
        creds = _load_creds()
        service = build("drive", "v3", credentials=creds)

        @st.cache_data(ttl=60)
        def _list_files_cached(fid, max_items, _cache_key):
            # _cache_key: 서비스계정 JSON(문자열)의 존재 자체를 캐시 키에 반영하기 위함
            return service.files().list(
                q=f"'{fid}' in parents and trashed=false",
                fields="files(id,name,mimeType,modifiedTime,webViewLink)",
                pageSize=max_items,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                corpora="allDrives",
            ).execute().get("files", [])


        files = res.get("files", [])
        rows = [
            {
                "name": f.get("name"),
                "mime": f.get("mimeType"),
                "modified": f.get("modifiedTime"),
                "link": f.get("webViewLink"),  # ← 추가
            }
            for f in files
        ]

        if not rows:
            # 비어있거나, 권한이 없으면 둘 다 0건일 수 있음
            msg = (
                "폴더에 파일이 없거나 접근 권한이 없습니다. "
                "→ 1) 폴더에 테스트 파일을 하나 넣어보고\n"
                "→ 2) 해당 폴더를 서비스계정(email 위 메시지 참고)에 '보기(Reader)'로 공유하세요.\n"
                "공유 드라이브라면 드라이브 멤버로 서비스계정을 추가하는 것도 권장합니다."
            )
            return True, msg, []

        return True, "OK", rows

    except Exception as e:
        # 대표적인 에러 해석
        text = str(e)
        if "File not found" in text or "notFound" in text:
            hint = "폴더 ID가 틀렸습니다. Drive 주소의 .../folders/뒤 문자열만 넣으세요."
        elif "insufficient" in text or "permissions" in text:
            hint = "권한 부족입니다. 폴더를 서비스계정 이메일에 '보기'로 공유하세요."
        else:
            hint = "알 수 없는 오류. 메시지를 복사해 알려주세요."
        return False, f"Drive API 오류: {e}\n힌트: {hint}", []
