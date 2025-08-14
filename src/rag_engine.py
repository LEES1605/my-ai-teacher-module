# src/rag_engine.py
from __future__ import annotations
from typing import List, Dict, Any
from google.oauth2 import service_account
from googleapiclient.discovery import build

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def _build_drive_service(creds_dict: Dict[str, Any]):
    """Service Account dict → Drive v3 service"""
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=DRIVE_SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def smoke_test_drive(creds_dict: Dict[str, Any], folder_id: str, limit: int = 10) -> List[Dict[str, str]]:
    """
    최소 연결 테스트: 폴더 안 파일 일부를 나열.
    반환: [{id, name, mimeType, modifiedTime}] 리스트
    """
    svc = _build_drive_service(creds_dict)
    resp = svc.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id,name,mimeType,modifiedTime)",
        pageSize=limit,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
        orderBy="modifiedTime desc"
    ).execute()
    return resp.get("files", [])
