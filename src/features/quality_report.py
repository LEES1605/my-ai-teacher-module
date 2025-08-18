# ===== [F03] QUALITY REPORT VIEW ============================================
import os, json, streamlit as st
from src.config import QUALITY_REPORT_PATH

def render_quality_report_view():
    st.subheader("📊 최적화 품질 리포트", divider="gray")
    if not os.path.exists(QUALITY_REPORT_PATH):
        st.info("아직 리포트가 없습니다. 인덱싱 후 자동 생성됩니다.")
        return
    try:
        with open(QUALITY_REPORT_PATH, "r", encoding="utf-8") as f:
            rep = json.load(f)
    except Exception as e:
        st.error("리포트 파일을 읽는 중 오류.")
        with st.expander("오류 보기"): st.exception(e)
        return

    s = rep.get("summary", {}) or {}
    st.write(
        f"- 전체 문서: **{s.get('total_docs', 0)}**개  "
        f"- 처리 파일: **{s.get('processed_docs', 0)}**개  "
        f"- 채택(kept): **{s.get('kept_docs', 0)}**개  "
        f"- 스킵(저품질): **{s.get('skipped_low_text', 0)}**개  "
        f"- 스킵(중복): **{s.get('skipped_dup', 0)}**개  "
        f"- 총 본문 문자수: **{s.get('total_chars', 0):,}**"
    )

    rows = []
    for fid, info in (rep.get("files", {}) or {}).items():
        rows.append([
            info.get("name", fid),
            info.get("kept", 0),
            info.get("skipped_low_text", 0),
            info.get("skipped_dup", 0),
            info.get("total_chars", 0),
            info.get("modifiedTime", ""),
        ])
    if rows:
        st.dataframe(
            rows, hide_index=True, use_container_width=True,
            column_config={0:"파일명",1:"채택",2:"저품질 스킵",3:"중복 스킵",4:"문자수",5:"수정시각"}
        )
    else:
        st.caption("아직 수집된 파일 통계가 없습니다.")
