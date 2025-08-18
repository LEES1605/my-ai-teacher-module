# ===== [F01] DRIVE CARD =======================================================
import streamlit as st
from src.config import settings
from src.rag_engine import _normalize_sa, _validate_sa
from src.patches.overrides import STATE_KEYS

def get_effective_gdrive_folder_id() -> str:
    return st.session_state.get(STATE_KEYS.GDRIVE_FOLDER_ID) or getattr(settings, "GDRIVE_FOLDER_ID", "")

def set_effective_gdrive_folder_id(fid: str) -> None:
    st.session_state[STATE_KEYS.GDRIVE_FOLDER_ID] = (fid or "").strip()

def render_drive_check_card() -> None:
    st.subheader("🔌 드라이브 연결 / 폴더")
    col1, col2, col3 = st.columns([0.5, 0.25, 0.25])

    with col1:
        cur = get_effective_gdrive_folder_id()
        new_fid = st.text_input("폴더 ID", value=cur, placeholder="drive 폴더 URL 마지막 ID")
        if new_fid != cur:
            set_effective_gdrive_folder_id(new_fid)
        if new_fid:
            st.markdown(f"[폴더 열기](https://drive.google.com/drive/folders/{new_fid})")

    with col2:
        ok_sa = True
        sa_email = "—"
        try:
            creds = _validate_sa(_normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON))
            sa_email = getattr(creds, "service_account_email", "service account")
        except Exception:
            ok_sa = False
        st.metric("서비스 계정", "정상" if ok_sa else "오류", delta=sa_email)

        if st.button("연결 테스트", key="btn_test_drive"):
            if not ok_sa:
                st.error("서비스계정 JSON을 확인하세요.")
            elif not get_effective_gdrive_folder_id():
                st.warning("폴더 ID를 입력하세요.")
            else:
                st.success("자격증명 구문 확인 OK (폴더 권한은 실제 작업 시 검증됩니다)")

    with col3:
        if st.button("저장", type="primary", key="btn_save_drive"):
            st.success("드라이브 폴더 설정을 저장했습니다. 이후 작업부터 적용됩니다.")
        st.caption("※ 폴더에 서비스계정 이메일을 **편집자**로 초대해야 업/다운로드 가능해요.")
