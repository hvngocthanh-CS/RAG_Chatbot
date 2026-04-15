"""
Conversational Query Rewriter Service.

Converts follow-up questions into fully self-contained, standalone queries
so that the vector retrieval step works correctly even in multi-turn
conversations.
"""

import logging
from typing import List, Dict, Optional

from backend.services import get_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt for the rewrite step
# ---------------------------------------------------------------------------
_REWRITE_SYSTEM = (
    "You rewrite a user's follow-up question into ONE clear, self-contained "
    "search query for a document retrieval system. The rewrite will be used "
    "only for retrieval — not shown to the user.\n\n"
    "RULES (in priority order):\n"
    "1. TOPIC SHIFT: If the follow-up is clearly on a NEW topic (different "
    "   entity, domain, or intent than recent turns — e.g. 'actually, forget "
    "   that', 'what about X instead', or simply an unrelated question), "
    "   output the follow-up UNCHANGED. Do NOT merge in prior-turn entities.\n"
    "2. COREFERENCE: If the follow-up references the prior turn via pronouns "
    "   or ellipsis ('it', 'they', 'that', 'those', 'the second one', "
    "   'nó', 'cái đó', bare 'and for X?', 'why?'), resolve every reference "
    "   to the specific entity/topic from history.\n"
    "3. PRESERVATION: Keep every specific name, number, date, and qualifier "
    "   from the follow-up verbatim. Never drop detail.\n"
    "4. NO EXPANSION: Do NOT add facts the user did not ask for. Do NOT invent "
    "   entity names not present in the history or follow-up.\n"
    "5. LANGUAGE: Keep the rewrite in the SAME language as the follow-up.\n"
    "6. FORMAT: Output ONLY the rewritten question as a single line — no "
    "   quotes, no prefix ('Rewritten:', 'Sure,'), no explanation.\n"
    "7. If the follow-up is already self-contained, output it unchanged."
)

_REWRITE_USER_TMPL = (
    "Conversation history (most recent last):\n"
    "{history}\n\n"
    "Follow-up question: {question}\n\n"
    "Standalone question:"
)

# Heuristic: skip LLM rewrite when the question is clearly self-contained.
# Triggers rewrite only when follow-up has pronouns/ellipsis or is short.
_PRONOUN_MARKERS = (
    # English
    " it ", " it?", " it.", " its ", " they ", " them ", " their ",
    " that ", " that?", " that.", " this ", " those ", " these ",
    " he ", " she ", " him ", " her ", " his ",
    " one ", " ones ", " the same ", " such ",
    # Vietnamese
    " nó ", " nó?", " nó.", " họ ", " chúng ",
    " cái đó", " cái này", " điều đó", " việc đó", " vậy",
)
_ELLIPSIS_STARTS = (
    "and ", "but ", "so ", "then ", "also ", "what about", "how about",
    "why", "why?", "when?", "where?", "who?", "how?",
    "và ", "còn ", "thế ", "vậy ", "tại sao", "khi nào", "ai ",
)


class QueryRewriterService:
    """
    Rewrites follow-up questions into standalone queries.

    Uses the same LLM backend as the chat service (via service registry)
    with a lightweight call (low max_tokens, temperature=0) to minimise
    latency.
    """

    async def rewrite(
        self,
        question: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Return a standalone version of *question* given *conversation_history*.

        Falls back to the original question on any error so retrieval
        is never blocked.
        """
        if not conversation_history:
            return question

        prior_turns = [t for t in conversation_history if t.get("content")]
        if not prior_turns:
            return question

        # Skip LLM call when the question looks fully self-contained.
        if self._is_self_contained(question):
            logger.debug("QueryRewriter: skipped (self-contained) [%s]", question[:80])
            return question

        try:
            rewritten = await self._call_llm(question, prior_turns)
            rewritten = self._sanitize(rewritten)
            if rewritten and self._looks_valid(rewritten, question):
                logger.info(
                    "QueryRewriter: [%s] -> [%s]",
                    question[:80],
                    rewritten[:80],
                )
                return rewritten
            logger.debug("QueryRewriter: output rejected, keeping original")
        except Exception as exc:
            logger.warning("QueryRewriter failed (%s), using original query", exc)

        return question

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------
    @staticmethod
    def _is_self_contained(question: str) -> bool:
        """True when the question needs no history to be understood."""
        q = question.strip()
        if len(q) < 15:
            return False
        lower = f" {q.lower()} "
        if any(m in lower for m in _PRONOUN_MARKERS):
            return False
        if any(lower.lstrip().startswith(s) for s in _ELLIPSIS_STARTS):
            return False
        # Has its own subject noun-ish signal: >= 6 tokens and a capitalised
        # token or a content verb. Cheap heuristic — treat long questions as
        # self-contained.
        return len(q.split()) >= 8

    @staticmethod
    def _sanitize(text: str) -> str:
        """Strip common LLM wrappers from the rewrite output."""
        t = (text or "").strip()
        # Take first non-empty line only.
        for line in t.splitlines():
            line = line.strip().strip('"').strip("'").strip("`")
            for prefix in (
                "rewritten:", "standalone question:", "standalone:",
                "query:", "question:", "sure,", "sure:", "here is",
            ):
                if line.lower().startswith(prefix):
                    line = line[len(prefix):].strip(" :-—")
            if line:
                return line
        return ""

    @staticmethod
    def _looks_valid(rewritten: str, original: str) -> bool:
        """Reject clearly-bad rewrites so we fall back to original."""
        if not rewritten or len(rewritten) > 400:
            return False
        # Reject rewrites that are a suspicious multi-line explanation.
        if rewritten.count("\n") > 0:
            return False
        return True

    def _format_history(self, turns: List[Dict]) -> str:
        """Format the last N turns for the prompt."""
        recent = turns[-6:]
        lines = []
        for t in recent:
            role = t.get("role", "user").capitalize()
            content = t.get("content", "")[:400]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    async def _call_llm(self, question: str, history: List[Dict]) -> str:
        """Call the LLM with a lightweight rewrite prompt."""
        llm_service = get_service("llm")
        if llm_service is None:
            return question

        client, provider = llm_service._get_active_client()
        model = llm_service._get_model_name(provider)

        prompt_user = _REWRITE_USER_TMPL.format(
            history=self._format_history(history),
            question=question,
        )

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": prompt_user},
            ],
            temperature=0.0,
            max_tokens=120,
        )

        return response.choices[0].message.content or question
