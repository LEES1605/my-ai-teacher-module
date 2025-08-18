# src/routing.py
# -----------------------------------------------------------------------------
# 질문 내용에 따라 응답 모드를 자동으로 선택하는 라우팅 모듈
# - 규칙기반으로 빠르게 동작
# - 나중에 ML 분류기로 교체할 때도 이 파일만 바꾸면 됨
# -----------------------------------------------------------------------------
from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


class ResponseMode(str, Enum):
    """앱 내부에서 통일해서 쓸 응답 모드 정의"""
    EXPLAINER = "explainer"   # 개념/원리 설명형
    ANALYST   = "analyst"      # 분석/비교/근거 제시형
    COACH     = "coach"        # 단계별 코칭/실습 안내형
    SUMMARIZER= "summarizer"   # 요약/정리형
    CODER     = "coder"        # 코드 작성/수정형
    RAG       = "rag"          # 자료검색/인용형(RAG)
    UNKNOWN   = "unknown"      # 기본값 (분류 실패 시)


@dataclass
class RoutingContext:
    """필요 시 대화 이력, 첨부 등 메타데이터를 넣어 확장할 수 있음"""
    history: Optional[List[str]] = None
    has_files: bool = False
    locale: str = "ko"


_CODE_HINTS = re.compile(r"(```|def |class |import |pip install|npm install|streamlit|python|자바|C\+\+|코드|오류|에러)", re.I)
_SUMMARY_HINTS = re.compile(r"(요약|정리|한줄|bullet|핵심|요점)", re.I)
_EXPLAIN_HINTS = re.compile(r"(설명해|개념|원리|왜|어떻게 동작|정의)", re.I)
_ANALYST_HINTS = re.compile(r"(분석|비교|장단점|추천 이유|근거|데이터)", re.I)
_COACH_HINTS = re.compile(r"(단계|스텝|따라|가이드|실습|튜토리얼|step-by-step|step by step)", re.I)
_RAG_HINTS = re.compile(r"(출처|인용|참고문헌|자료 찾아|검색해|링크|파일 내용|드라이브|RAG|지식)", re.I)

def _rough_lang_detect(text: str) -> str:
    # 초간단 감지(ko/en) – 필요시 fasttext 등으로 교체
    if re.search(r"[가-힣]", text):
        return "ko"
    return "en"

def choose_response_mode(query: str, ctx: Optional[RoutingContext] = None) -> ResponseMode:
    """
    규칙 기반 자동 라우팅:
    - 코드/오류: CODER
    - 요약 요청: SUMMARIZER
    - 개념/원리/이유 설명: EXPLAINER
    - 분석/비교/근거: ANALYST
    - 단계별 가이드/실습: COACH
    - 출처/검색/파일연동: RAG (또는 has_files=True)
    - 그 외: EXPLAINER(기본)
    """
    text = (query or "").strip()
    if not text:
        return ResponseMode.EXPLAINER

    locale = _rough_lang_detect(text)
    if ctx is None:
        ctx = RoutingContext(locale=locale)

    # 파일이 붙어오거나, "출처/인용/검색" 류 키워드면 RAG 우선
    if ctx.has_files or _RAG_HINTS.search(text):
        return ResponseMode.RAG

    if _CODE_HINTS.search(text):
        return ResponseMode.CODER

    if _SUMMARY_HINTS.search(text):
        return ResponseMode.SUMMARIZER

    if _COACH_HINTS.search(text):
        return ResponseMode.COACH

    if _ANALYST_HINTS.search(text):
        return ResponseMode.ANALYST

    if _EXPLAIN_HINTS.search(text):
        return ResponseMode.EXPLAINER

    # 너무 짧은 질문은 요약보단 설명으로 유도
    if len(text) < 20:
        return ResponseMode.EXPLAINER

    # 기본값
    return ResponseMode.EXPLAINER
