# app.py — 스텝 인덱싱(중간취소/재개) + 두뇌준비 안정화
#        + 인덱싱 보고서(스킵 표시)
#        + Drive 대화로그 저장(❶ OAuth: Markdown / ❷ 서비스계정: JSONL, chat_log/)
#        + 페르소나: 🤖Gemini(친절/꼼꼼), 🤖ChatGPT(유머러스/보완)

from __future__ import annotations
import os, time, uuid, re, json
import pandas as pd
import streamlit as st

# ============= 0) 페이지 설정 (반드시 최상단) ====================================
st.set_page_config(page_title="나의 AI 영어 교사", layout="wide", initial_sidebar_state="collapsed")

# ============= 1) 부트 가드 & 런타임 안정화 ===================================
ss = st.session_state
ss.setdefault("_boot_log", [])
ss.setdefault("_oauth_checked", False)

def _boot(msg: str): ss["_boot_log"].append(msg)

with st.sidebar:
    st.caption("🛠 Boot log (임시)")
    _boot_box = st.empty()

def _flush_boot():
    try:
        _boot_box.write("\n".join(ss["_boot_log"]) or "(empty)")
    except Exception:
        pass

_boot("A: page_config set"); _flush_boot()

# 런타임 튜닝(불필요한 리소스/경합 방지)
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION"] = "false"

# ============= 2) 세션 키 초기화 ==============================================
ss.setdefault("session_id", uuid.uuid4().hex[:12])
ss.setdefault("messages", [])
ss.setdefault("auto_save_chatlog", True)    # OAuth Markdown 저장(내 드라이브)
ss.setdefault("save_logs", False)           # SA JSONL 저장(공유드라이브 writer 필요)
ss.setdefault("prep_both_running", False)
ss.setdefault("prep_both_done", ("qe_google" in ss) or ("qe_openai" in ss))
ss.setdefault("prep_cancel_requested", False)
ss.setdefault("session_terminated", False)
ss.setdefault("index_job", None)

# (진단) 하트비트
st.caption(f"heartbeat ✅ keys={list(ss.keys())[:8]}")

# ============= 3) 기본 UI 헤더/스타일 =========================================
from src.ui import load_css, render_header
load_css()
render_header()
st.info("✅ 변경이 있을 때만 인덱싱합니다. 저장된 두뇌가 있으면 즉시 로드합니다. (중간 취소/재개 지원)")

# ============= 4) OAuth 리다이렉트 처리(최종화 1회만) ==========================
#  - 이전 코드에 있었던 이중 호출을 제거하여 깔끔하게 유지
try:
    from src.google_oauth import finish_oauth_if_redirected
    if not st.secrets.get("OAUTH_DISABLE_FINISH"):
        if not ss.get("_oauth_finalized", False):
            finalized = finish_oauth_if_redirected()
            if finalized:
                ss["_oauth_finalized"] = True
                try:
                    st.query_params.clear()
                except Exception:
                    st.experimental_set_query_params()
                st.rerun()
except Exception as e:
    st.warning(f"OAuth finalize skipped: {e}")

# ============= 5) 사이드바: OAuth 로그인/로그아웃, 저장 옵션 ==================
from src.google_oauth import start_oauth, is_signed_in, build_drive_service, get_user_email, sign_out
with st.sidebar:
    ss.auto_save_chatlog = st.toggle(
        "대화 자동 저장 (OAuth/내 드라이브, Markdown)",
        value=ss.auto_save_chatlog
    )
    ss.save_logs = st.toggle(
        "대화 JSONL 저장 (서비스계정/chat_log/)",
        value=ss.save_logs,
        help="공유드라이브 Writer 권한 필요. 쿼터 문제 시 끄기 권장."
    )
    st.markdown("---")
    st.markdown("### Google 로그인 (내 드라이브 저장)")
    if not is_signed_in():
        if st.button("🔐 Google로 로그인"):
            url = start_oauth()
            st.markdown(f"[여기를 눌러 로그인하세요]({url})")
    else:
        st.success(f"로그인됨: {get_user_email() or '알 수 없음'}")
        if st.button("로그아웃"):
            sign_out()
            st.rerun()

# ============= 6) Google Drive 연결 테스트(안정화 버전) ========================
st.markdown("## 🔗 Google Drive 연결 테스트")
st.caption("서비스계정 저장은 공유드라이브 Writer 권한이 필요. 인덱싱은 Readonly면 충분합니다.")

from src.config import settings
from src.rag_engine import smoke_test_drive, preview_drive_files, drive_diagnostics

try:
    ok_sa, head_sa, details_sa = drive_diagnostics(settings.GDRIVE_FOLDER_ID)  # (ok, 헤드라인, 상세 리스트[str])
    if ok_sa:
        st.success(head_sa)
    else:
        st.warning(head_sa)
    with st.expander("서비스계정 JSON 진단 상세", expanded=not ok_sa):
        st.code("\n".join(details_sa), language="text")
except Exception as e:
    st.warning("진단 함수 예외:")
    st.code(
        f"{type(e).__name__}: {e}\n"
        f"타입={type(settings.GDRIVE_SERVICE_ACCOUNT_JSON).__name__}\n"
        f"프리뷰={str(settings.GDRIVE_SERVICE_ACCOUNT_JSON)[:200]}...",
        language="text"
    )

colL, colR = st.columns([0.65, 0.35], vertical_alignment="top")

with colL:
    if st.button("폴더 파일 미리보기 (최신 10개)", use_container_width=True):
        ok, msg, rows = preview_drive_files(max_items=10)
        if ok and rows:
            df = pd.DataFrame(rows)
            # 보기 친화적으로 가공
            df["type"] = df["mime"].str.replace("application/vnd.google-apps.", "", regex=False)
            df = df.rename(columns={"modified": "modified_at"})[["name","link","type","modified_at"]]
            st.dataframe(
                df,
                use_container_width=True,
                height=360,
                column_config={
                    "name": st.column_config.TextColumn("파일명"),
                    "link": st.column_config.LinkColumn("open", display_text="열기"),
                    "type": st.column_config.TextColumn("유형"),
                    "modified_at": st.column_config.TextColumn("수정시각"),
                },
                hide_index=True
            )
            st.success(f"총 {len(rows)}개 항목 표시 (최신 10개 기준).")
        elif ok:
            st.info("폴더에 파일이 없거나 접근할 수 없습니다.")
        else:
            st.error(msg)

with colR:
    ok, msg = smoke_test_drive()
    if ok:
        st.success(msg)     # ⚠️ 반환값을 st.write 등으로 다시 찍지 않음
    else:
        st.warning(msg)

# ============= 6.5) 📤 관리자: 자료 업로드 (원본 → prepared 저장) ===============
with st.expander("📤 관리자: 자료 업로드 (원본→prepared 저장)", expanded=False):
    st.caption("여러 PDF를 한 번에 업로드합니다. 원본은 prepared 폴더에 저장되며, 텍스트 추출물은 인덱스 캐시에만 저장됩니다.")
    files = st.file_uploader(
        "PDF 파일을 선택하세요 (여러 개 가능)",
        type=["pdf"],                 # ← 필요 시 ["pdf","docx","md","txt","pptx"]로 확장 가능
        accept_multiple_files=True
    )

    # 진행 상태 표시용 위젯
    prog = st.progress(0, text="대기 중…")
    status_area = st.empty()
    result_area = st.empty()

    if files and st.button("업로드 → prepared", type="primary"):
        try:
            import io, re, time
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaIoBaseUpload
            from src.rag_engine import _normalize_sa
            from src.config import settings

            # 0) 서비스계정으로 Drive 클라이언트 생성
            creds = _normalize_sa(settings.GDRIVE_SERVICE_ACCOUNT_JSON)
            drive = build("drive", "v3", credentials=creds)

            # 1) 업로드 루프
            total = len(files)
            rows = []       # 결과표
            uploaded = 0

            for i, f in enumerate(files, start=1):
                # (1) 파일 읽기
                data = f.read()
                buf = io.BytesIO(data)

                # (2) 파일명: 타임스탬프__원본이름 (정렬/중복 방지)
                ts = time.strftime("%Y%m%d_%H%M%S")
                base = re.sub(r"[^\w\-. ]", "_", f.name)
                name = f"{ts}__{base}"

                # (3) 업로드
                prog.progress(int(i / total * 100), text=f"업로드 중… ({i}/{total})")
                status_area.info(f"업로드 중: {name}")

                media = MediaIoBaseUpload(buf, mimetype="application/pdf", resumable=False)
                meta = {"name": name, "parents": [settings.GDRIVE_FOLDER_ID]}

                res = drive.files().create(body=meta, media_body=media, fields="id,webViewLink").execute()
                rows.append({"name": name, "open": res.get("webViewLink")})
                uploaded += 1

                # 드라이브 API rate-limit 완화(가벼운 텀)
                time.sleep(0.1)

            # 2) 요약 출력
            prog.progress(100, text="완료")
            status_area.success(f"{uploaded}개 업로드 완료 (prepared)")

            if rows:
                df = pd.DataFrame(rows)
                result_area.dataframe(
                    df, use_container_width=True, hide_index=True,
                    column_config={"name": st.column_config.TextColumn("파일명"),
                                   "open": st.column_config.LinkColumn("열기", display_text="열기")}
                )
            st.toast("업로드 완료 — 변경 사항은 인덱싱 시 반영됩니다.", icon="✅")

            # 3) 인덱싱을 다시 돌릴 수 있게 준비 버튼 재활성화
            ss.prep_both_done = False

        except Exception as e:
            prog.progress(0, text="오류")
            status_area.error(f"업로드 실패: {e}")

# ==============================================================================

# ============= 7) 인덱싱 보고서(스킵된 파일 포함) ===============================



# ============= 7) 인덱싱 보고서(스킵된 파일 포함) ===============================
rep = ss.get("indexing_report")
if rep:
    with st.expander("🧾 인덱싱 보고서 (스킵된 파일 보기)", expanded=False):
        st.write(
            f"총 파일(매니페스트): {rep.get('total_manifest')}, "
            f"로딩된 문서 수: {rep.get('loaded_docs')}, "
            f"스킵: {rep.get('skipped_count')}"
        )
        skipped = rep.get("skipped", [])
        if skipped:
            st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)
        else:
            st.caption("스킵된 파일이 없습니다 🎉")

# ============= 8) 두뇌 준비(증분 인덱싱 · 중간취소/재개) =======================
st.markdown("---")
st.subheader("🧠 두뇌 준비 — 저장본 로드 ↔ 변경 시 증분 인덱싱 (중간 취소/재개)")

c_g, c_o = st.columns(2)
with c_g: st.caption("Gemini 진행"); g_bar = st.empty(); g_msg = st.empty()
with c_o: st.caption("ChatGPT 진행"); o_bar = st.empty(); o_msg = st.empty()

def _render_progress(slot_bar, slot_msg, pct: int, msg: str | None = None):
    p = max(0, min(100, int(pct)))
    slot_bar.markdown(
        f"<div class='gp-wrap'><div class='gp-fill' style='width:{p}%'></div>"
        f"<div class='gp-label'>{p}%</div></div>",
        unsafe_allow_html=True
    )
    if msg is not None:
        slot_msg.markdown(f"<div class='gp-msg'>{msg}</div>", unsafe_allow_html=True)

def _is_cancelled() -> bool:
    return bool(ss.get("prep_cancel_requested", False))

from src.rag_engine import (
    set_embed_provider, make_llm, CancelledError,
    start_index_builder, resume_index_builder, cancel_index_builder,
    get_text_answer, llm_complete,   # 대화 단계에서 재사용
)

def run_prepare_both_step():
    # 1) 임베딩 설정 (OpenAI 우선, 없으면 Google)
    embed_provider = "openai"
    embed_api = (getattr(settings, "OPENAI_API_KEY", None).get_secret_value()
                 if getattr(settings, "OPENAI_API_KEY", None) else "")
    embed_model = getattr(settings, "OPENAI_EMBED_MODEL", "text-embedding-3-small")
    if not embed_api:
        embed_provider = "google"
        embed_api = settings.GEMINI_API_KEY.get_secret_value()
        embed_model = getattr(settings, "EMBED_MODEL", "text-embedding-004")
    try:
        _render_progress(g_bar, g_msg, 3, f"임베딩 설정({embed_provider})")
        _render_progress(o_bar, o_msg, 3, f"임베딩 설정({embed_provider})")
        set_embed_provider(embed_provider, embed_api, embed_model)
    except Exception as e:
        _render_progress(g_bar, g_msg, 100, f"임베딩 실패: {e}")
        _render_progress(o_bar, o_msg, 100, f"임베딩 실패: {e}")
        ss.prep_both_running = False
        return

    # 2) 인덱스 스텝 진행
    def upd(p, m=None):
        _render_progress(g_bar, g_msg, p, m)
        _render_progress(o_bar, o_msg, p, m)

    def umsg(m):
        _render_progress(g_bar, g_msg, ss.get("p_shared", 0), m)
        _render_progress(o_bar, o_msg, ss.get("p_shared", 0), m)

    persist_dir = f"{getattr(settings,'PERSIST_DIR','/tmp/my_ai_teacher/storage_gdrive')}_shared"
    job = ss.get("index_job")

    try:
        if job is None:
            res = start_index_builder(
                update_pct=upd,
                update_msg=umsg,
                gdrive_folder_id=settings.GDRIVE_FOLDER_ID,
                raw_sa=settings.GDRIVE_SERVICE_ACCOUNT_JSON,
                persist_dir=persist_dir,
                manifest_path=getattr(settings, "MANIFEST_PATH", "/tmp/my_ai_teacher/drive_manifest.json"),
                max_docs=None,     # 전체 인덱싱
                is_cancelled=_is_cancelled,
            )
        else:
            res = resume_index_builder(
                job=job,
                update_pct=upd,
                update_msg=umsg,
                is_cancelled=_is_cancelled,
                batch_size=6
            )

        status = res.get("status")
        if status == "running":
            ss.index_job = res["job"]
            _render_progress(g_bar, g_msg, res.get("pct", 8), res.get("msg", "진행 중…"))
            _render_progress(o_bar, o_msg, res.get("pct", 8), res.get("msg", "진행 중…"))
            time.sleep(0.15)
            st.rerun()
            return

        if status == "cancelled":
            ss.prep_both_running = False
            ss.prep_cancel_requested = False
            ss.index_job = None
            _render_progress(g_bar, g_msg, res.get("pct", 0), "사용자 취소")
            _render_progress(o_bar, o_msg, res.get("pct", 0), "사용자 취소")
            return

        if status != "done":
            _render_progress(g_bar, g_msg, 100, "인덱싱 실패")
            _render_progress(o_bar, o_msg, 100, "인덱싱 실패")
            ss.prep_both_running = False
            return

        index = res["index"]
        ss.index_job = None

    except Exception as e:
        ss.prep_both_running = False
        ss.index_job = None
        _render_progress(g_bar, g_msg, 100, f"에러: {e}")
        _render_progress(o_bar, o_msg, 100, f"에러: {e}")
        return

    # 3) QueryEngine 생성
    try:
        g_llm = make_llm(
            "google",
            settings.GEMINI_API_KEY.get_secret_value(),
            getattr(settings, "LLM_MODEL", "gemini-1.5-pro"),
            float(ss.get("temperature", 0.0))
        )
        ss["llm_google"] = g_llm
        ss["qe_google"] = index.as_query_engine(
            llm=g_llm,
            response_mode=ss.get("response_mode", getattr(settings, "RESPONSE_MODE", "compact")),
            similarity_top_k=int(ss.get("similarity_top_k", getattr(settings, "SIMILARITY_TOP_K", 5))),
        )
        _render_progress(g_bar, g_msg, 100, "완료!")
    except Exception as e:
        _render_progress(g_bar, g_msg, 100, f"Gemini 준비 실패: {e}")

    try:
        if getattr(settings, "OPENAI_API_KEY", None) and settings.OPENAI_API_KEY.get_secret_value():
            o_llm = make_llm(
                "openai",
                settings.OPENAI_API_KEY.get_secret_value(),
                getattr(settings, "OPENAI_LLM_MODEL", "gpt-4o-mini"),
                float(ss.get("temperature", 0.0))
            )
            ss["llm_openai"] = o_llm
            ss["qe_openai"] = index.as_query_engine(
                llm=o_llm,
                response_mode=ss.get("response_mode", getattr(settings, "RESPONSE_MODE", "compact")),
                similarity_top_k=int(ss.get("similarity_top_k", getattr(settings, "SIMILARITY_TOP_K", 5))),
            )
            _render_progress(o_bar, o_msg, 100, "완료!")
        else:
            _render_progress(o_bar, o_msg, 100, "키 누락 — OPENAI_API_KEY 필요")
    except Exception as e:
        _render_progress(o_bar, o_msg, 100, f"ChatGPT 준비 실패: {e}")

    ss.prep_both_running = False
    ss.prep_both_done = True
    time.sleep(0.2)
    st.rerun()

# 실행/취소 버튼 줄
left, right = st.columns([0.7, 0.3])
with left:
    clicked = st.button(
        "🚀 한 번에 준비하기",
        key="prepare_both",
        use_container_width=True,
        disabled=ss.prep_both_running or ss.prep_both_done
    )
with right:
    cancel_clicked = st.button(
        "⛔ 준비 취소",
        key="cancel_prepare",
        use_container_width=True,
        type="secondary",
        disabled=not ss.prep_both_running
    )

from src.rag_engine import cancel_index_builder  # 재노출(명시)

if cancel_clicked and ss.prep_both_running:
    ss.prep_cancel_requested = True
    if ss.get("index_job"):
        cancel_index_builder(ss.index_job)
    st.rerun()

if clicked and not (ss.prep_both_running or ss.prep_both_done):
    ss.prep_cancel_requested = False
    ss.prep_both_running = True
    ss.index_job = None
    st.rerun()

if ss.prep_both_running:
    run_prepare_both_step()

st.caption("준비 버튼을 다시 활성화하려면 아래 재설정 버튼을 누르세요.")
if st.button("🔧 재설정(버튼 다시 활성화)", disabled=not ss.prep_both_done):
    ss.prep_both_done = False
    st.rerun()

# ============= 9) 대화 UI (그룹토론) ===========================================
st.markdown("---")
st.subheader("💬 그룹토론 — 학생 ↔ 🤖Gemini(친절/꼼꼼) ↔ 🤖ChatGPT(유머러스/보완)")

ready_google = "qe_google" in ss
ready_openai = "qe_openai" in ss

if ss.session_terminated:
    st.warning("세션이 종료된 상태입니다. 새로고침으로 다시 시작하세요.")
    st.stop()

if not ready_google:
    st.info("먼저 **[🚀 한 번에 준비하기]**로 두뇌를 준비하세요. (OpenAI 키 없으면 Gemini만 응답)")
    st.stop()

# 과거 메시지 렌더
for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 맥락/도우미
def _strip_sources(text: str) -> str:
    return re.sub(r"\n+---\n\*참고 자료:.*$", "", text, flags=re.DOTALL)

def _build_context(messages, limit_pairs=2, max_chars=2000) -> str:
    pairs, buf_user = [], None
    for m in reversed(messages):
        role, content = m.get("role"), str(m.get("content","")).strip()
        if role == "assistant":
            content = re.sub(r"^\*\*🤖 .*?\*\*\s*\n+", "", content).strip()
            if buf_user is not None:
                pairs.append((buf_user, content))
                buf_user = None
                if len(pairs) >= limit_pairs:
                    break
        elif role == "user" and buf_user is None:
            buf_user = content
    pairs = list(reversed(pairs))
    blocks = [f"[학생]\n{u}\n\n[교사]\n{a}" for u,a in pairs]
    ctx = "\n\n---\n\n".join(blocks).strip()
    return ctx[-max_chars:] if len(ctx) > max_chars else ctx

from src.prompts import EXPLAINER_PROMPT, ANALYST_PROMPT, READER_PROMPT
def _persona():
    mode = ss.get("mode_select", "💬 이유문법 설명")
    base = EXPLAINER_PROMPT if mode=="💬 이유문법 설명" else (ANALYST_PROMPT if mode=="🔎 구문 분석" else READER_PROMPT)
    common = "역할: 학생의 영어 실력을 돕는 AI 교사.\n규칙: 근거가 불충분하면 그 사실을 명확히 밝힌다. 예시는 짧고 점진적으로."
    return base + "\n" + common

GEMINI_STYLE = "당신은 착하고 똑똑한 친구 같은 교사입니다. 칭찬과 격려, 정확한 설명."
CHATGPT_STYLE = (
    "당신은 유머러스하지만 정확한 동료 교사입니다. 동료(Gemini)의 답을 읽고 "
    "빠진 부분을 보완/교정하고 마지막에 <최종 정리>로 요약하세요. 과한 농담 금지."
)

mode = st.radio(
    "학습 모드",
    ["💬 이유문법 설명", "🔎 구문 분석", "📚 독해 및 요약"],
    horizontal=True, key="mode_select"
)

# 서비스계정 JSONL 저장 (chat_log/)
from src import chat_store
from src.drive_log import get_chatlog_folder_id, save_chatlog_markdown_oauth

def _jsonl_log(items):
    if not ss.save_logs:
        return
    try:
        parent_id = (getattr(settings, "CHATLOG_FOLDER_ID", None) or settings.GDRIVE_FOLDER_ID)
        sa = settings.GDRIVE_SERVICE_ACCOUNT_JSON
        if isinstance(sa, str):
            try:
                sa = json.loads(sa)
            except Exception:
                pass
        sub_id = get_chatlog_folder_id(parent_folder_id=parent_id, sa_json=sa)
        chat_store.append_jsonl(folder_id=sub_id, sa_json=sa, items=items)
        st.toast("대화 JSONL 저장 완료", icon="💾")
    except Exception as e:
        st.warning(f"대화 JSONL 저장 실패: {e}")

# 입력 & 응답
user_input = st.chat_input("질문을 입력하거나, 분석/요약할 문장이나 글을 붙여넣으세요.")
if user_input:
    ss.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    _jsonl_log([chat_store.make_entry(
        ss.session_id, "user", "user", user_input, mode, model="user"
    )])

    # Gemini 1차
    with st.spinner("🤖 Gemini 선생님이 먼저 답합니다…"):
        prev_ctx = _build_context(ss.messages[:-1], limit_pairs=2, max_chars=2000)
        gemini_query = (f"[이전 대화]\n{prev_ctx}\n\n" if prev_ctx else "") + f"[학생 질문]\n{user_input}"
        ans_g = get_text_answer(ss["qe_google"], gemini_query, _persona() + "\n" + GEMINI_STYLE)

    content_g = f"**🤖 Gemini**\n\n{ans_g}"
    ss.messages.append({"role":"assistant","content":content_g})
    with st.chat_message("assistant"):
        st.markdown(content_g)

    _jsonl_log([chat_store.make_entry(
        ss.session_id, "assistant", "Gemini", content_g, mode, model=getattr(settings,"LLM_MODEL","gemini")
    )])

    # ChatGPT 보완(키가 있을 때)
    if ready_openai:
        review_directive = (
            "역할: 동료 AI 영어교사\n"
            "목표: [이전 대화], [학생 질문], [동료의 1차 답변]을 읽고 사실오류/빠진점/모호함을 보완.\n"
            "지침: 1)핵심 간결 재정리 2)틀린 부분 근거와 교정 3)예문 2~3개 4)<최종 정리>로 요약. 외부검색 금지."
        )
        prev_all = _build_context(ss.messages, limit_pairs=2, max_chars=2000)
        augmented = (
            (f"[이전 대화]\n{prev_all}\n\n" if prev_all else "") +
            f"[학생 질문]\n{user_input}\n\n"
            f"[동료의 1차 답변(Gemini)]\n{_strip_sources(ans_g)}\n\n"
            "[당신의 작업]\n위 기준으로만 보완/검증."
        )
        with st.spinner("🤝 ChatGPT 선생님이 보완/검증 중…"):
            ans_o = llm_complete(
                ss.get("llm_openai"),
                _persona() + "\n" + CHATGPT_STYLE + "\n\n" + review_directive + "\n\n" + augmented
            )
        content_o = f"**🤖 ChatGPT**\n\n{ans_o}"
        ss.messages.append({"role":"assistant","content":content_o})
        with st.chat_message("assistant"):
            st.markdown(content_o)
    else:
        with st.chat_message("assistant"):
            st.info("ChatGPT 키가 없어 Gemini만 응답했습니다. OPENAI_API_KEY를 추가하면 보완/검증이 활성화됩니다.")

    # OAuth Markdown 저장(내 드라이브)
    if ss.auto_save_chatlog and ss.messages:
        try:
            if is_signed_in():
                svc = build_drive_service()
                parent_id = (st.secrets.get("OAUTH_CHAT_PARENT_ID") or "").strip() or None
                _fid = save_chatlog_markdown_oauth(ss.session_id, ss.messages, svc, parent_id)
                st.toast("내 드라이브에 대화 저장 완료 ✅", icon="💾")
            else:
                st.info("구글 계정으로 로그인하면 대화가 **내 드라이브**에 저장됩니다.")
        except Exception as e:
            st.warning(f"OAuth 저장 실패: {e}")

# EOF
