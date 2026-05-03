"""
Query Expander — two modes depending on question complexity.

Simple queries  → paraphrase mode: N synonymous phrasings for higher recall.
Complex queries → decompose mode: atomic sub-questions, one per retrieval pass.

Paraphrase mode improves recall when a query might miss relevant chunks due
to vocabulary mismatch.  Decompose mode ensures every distinct aspect of a
multi-part question gets its own independent retrieval pass, so context for
sub-question 4 is not silently dropped in favour of sub-question 1.

Both modes fall back gracefully — any failure returns an empty result and the
pipeline continues with single-query retrieval.
"""
import logging
import re
from typing import List, NamedTuple

from backend.services import get_service

logger = logging.getLogger(__name__)


class ExpansionResult(NamedTuple):
    queries: List[str]  # paraphrases or sub-questions (excluding the original)
    is_decomposed: bool  # True when decompose mode was used


# ---------------------------------------------------------------------------
# Paraphrase prompt — for simple, single-intent queries
# ---------------------------------------------------------------------------

_PARAPHRASE_SYSTEM = """\
You are a search query optimizer for a document retrieval system.
Given a user question, generate {n} alternative search queries.

Rules:
- Same information need, different vocabulary and phrasing
- Each query must be concise and self-contained
- Use the SAME language as the input question
- Output ONLY the queries, one per line — no numbering, no explanation, no blank lines"""

_PARAPHRASE_USER = "Question: {question}\n\nAlternative queries:"


# ---------------------------------------------------------------------------
# Decompose prompt — for complex multi-part questions
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM = """\
You are a query decomposer for a document retrieval system.
The user asked a complex multi-part question. Break it into atomic sub-questions
that can each be answered independently from a single document section.

Rules:
- Each sub-question must be fully self-contained (no pronouns referencing other sub-questions)
- Preserve specific entity names, figures, and terms exactly as written
- Cover every distinct aspect of the original question — do not omit any part
- Keep each sub-question concise (one sentence)
- Use the SAME language as the input question
- Output ONLY the sub-questions, one per line — no numbering, no explanation, no blank lines"""

_DECOMPOSE_USER = "Complex question: {question}\n\nAtomic sub-questions:"


# ---------------------------------------------------------------------------
# Classify prompt — fast routing: paraphrase vs decompose
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """\
Classify the user query into exactly one category.

simple       — single intent, one thing asked
comparative  — compares two or more distinct items, versions, or time periods
multi-aspect — asks about multiple separate aspects, steps, conditions, or components

Output ONLY the label, nothing else."""

_CLASSIFY_USER = "Query: {question}"


# ---------------------------------------------------------------------------
# Multi-part detection heuristics — regex fallback for _classify_query
# ---------------------------------------------------------------------------

_NUMBERED_LIST = re.compile(r"(?:^|\n)\s*(?:[1-9][.):]|[①-⑨])", re.M)
_LIST_INTRO = re.compile(r"\b(?:including:|such as:|specifically:|as follows:)", re.I)
_BEFORE_AFTER = re.compile(r"\b(?:before and after|both\b.{1,40}\band\b|as well as)\b", re.I)
_MULTI_AND = re.compile(r"(?:\band\b.*){2,}", re.I)  # "and" appears ≥2 times


def _is_multi_part(question: str) -> bool:
    """Return True when the question contains signals of multiple distinct sub-questions."""
    if question.count("?") > 1:
        return True
    if _NUMBERED_LIST.search(question):
        return True
    if _LIST_INTRO.search(question):
        return True
    if _BEFORE_AFTER.search(question):
        return True
    if _MULTI_AND.search(question):
        return True
    return False


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class QueryExpanderService:
    """
    Expands a query via paraphrase (simple) or decomposition (complex).

    Returns an ExpansionResult so the pipeline can adapt top_k and reranker
    thresholds to the query complexity.
    """

    async def expand(self, question: str, n: int = 2) -> ExpansionResult:
        """
        Return expanded queries plus a flag indicating the mode used.

        Routing logic:
          - Short query (< 5 words)          → skip, return empty
          - Long query (≥ 8 words)           → LLM classifier → decompose or paraphrase
          - Otherwise                        → paraphrase mode

        LLM classifier returns 'simple' | 'comparative' | 'multi-aspect'.
        comparative / multi-aspect → decompose; simple → paraphrase.
        On classifier failure, regex heuristics (_is_multi_part) act as fallback.

        On any LLM failure:
          - Decompose failure falls back to paraphrase
          - Paraphrase failure returns empty (single-query fallback)
        """
        from backend.config import settings

        if len(question.split()) < 5:
            logger.debug("QueryExpander: skipped (query too short)")
            return ExpansionResult(queries=[], is_decomposed=False)

        use_decompose = False
        if settings.DECOMPOSE_ENABLED and len(question.split()) >= settings.DECOMPOSE_MIN_WORDS:
            label = await self._classify_query(question)
            use_decompose = label in ("multi-aspect", "comparative")
            logger.debug("QueryExpander [classify]: '%s' → %s", question[:60], label)

        if use_decompose:
            queries = await self._decompose(question)
            if queries:
                logger.info(
                    "QueryExpander [decompose]: %d sub-questions — %s",
                    len(queries),
                    " | ".join(f"[{q[:60]}]" for q in queries),
                )
                return ExpansionResult(queries=queries, is_decomposed=True)
            logger.info("QueryExpander [decompose]: LLM failed — falling back to paraphrase")

        queries = await self._paraphrase(question, n)
        if queries:
            logger.info(
                "QueryExpander [paraphrase]: %d variants — %s",
                len(queries),
                " | ".join(f"[{q[:50]}]" for q in queries),
            )
        return ExpansionResult(queries=queries, is_decomposed=False)

    async def _classify_query(self, question: str) -> str:
        """Classify a query as 'simple', 'comparative', or 'multi-aspect'.

        Falls back to regex heuristics (_is_multi_part) on any LLM failure.
        """
        try:
            llm_service = get_service("llm")
            if llm_service is None:
                return "multi-aspect" if _is_multi_part(question) else "simple"
            client, provider = llm_service._get_active_client()
            model = llm_service._get_model_name(provider)
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _CLASSIFY_SYSTEM},
                    {"role": "user", "content": _CLASSIFY_USER.format(question=question)},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            label = (response.choices[0].message.content or "").strip().lower()
            if label in ("simple", "comparative", "multi-aspect"):
                return label
            return "multi-aspect" if _is_multi_part(question) else "simple"
        except Exception as exc:
            logger.warning("QueryExpander (classify) failed (%s) — regex fallback", exc)
            return "multi-aspect" if _is_multi_part(question) else "simple"

    async def _paraphrase(self, question: str, n: int) -> List[str]:
        try:
            llm_service = get_service("llm")
            if llm_service is None:
                return []
            client, provider = llm_service._get_active_client()
            model = llm_service._get_model_name(provider)
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _PARAPHRASE_SYSTEM.format(n=n)},
                    {"role": "user", "content": _PARAPHRASE_USER.format(question=question)},
                ],
                temperature=0.3,
                max_tokens=150,
            )
            return self._parse_lines(
                response.choices[0].message.content or "", question, max_items=n
            )
        except Exception as exc:
            logger.warning("QueryExpander (paraphrase) failed (%s) — single query", exc)
            return []

    async def _decompose(self, question: str) -> List[str]:
        try:
            llm_service = get_service("llm")
            if llm_service is None:
                return []
            client, provider = llm_service._get_active_client()
            model = llm_service._get_model_name(provider)
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _DECOMPOSE_SYSTEM},
                    {"role": "user", "content": _DECOMPOSE_USER.format(question=question)},
                ],
                temperature=0.0,
                max_tokens=400,
            )
            # No max_items cap — keep all sub-questions the LLM identified
            return self._parse_lines(response.choices[0].message.content or "", question)
        except Exception as exc:
            logger.warning("QueryExpander (decompose) failed (%s)", exc)
            return []

    @staticmethod
    def _parse_lines(raw: str, original: str, max_items: int = 0) -> List[str]:
        queries: List[str] = []
        for line in raw.splitlines():
            q = line.strip().lstrip("-•0123456789.) ").strip('"').strip("'").strip()
            if q and q.lower() != original.lower():
                queries.append(q)
        return queries[:max_items] if max_items else queries
