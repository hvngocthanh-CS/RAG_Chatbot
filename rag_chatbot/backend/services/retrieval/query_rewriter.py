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
    "You are a query rewriter for an enterprise document Q&A system. "
    "Your ONLY job is to rewrite the user's follow-up question into one "
    "clear, self-contained English question that can be understood with NO "
    "prior conversation context. "
    "Rules:\n"
    "1. Resolve all pronouns ('it', 'they', 'those', 'that', etc.) using "
    "   the conversation history.\n"
    "2. Preserve every specific detail, number, or name from the follow-up.\n"
    "3. Output ONLY the rewritten question — no explanation, no preamble.\n"
    "4. If the question is already self-contained, output it unchanged."
)

_REWRITE_USER_TMPL = (
    "Conversation history (most recent last):\n"
    "{history}\n\n"
    "Follow-up question: {question}\n\n"
    "Rewritten standalone question:"
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

        try:
            rewritten = await self._call_llm(question, prior_turns)
            rewritten = rewritten.strip().strip('"').strip("'")
            if rewritten:
                logger.info(
                    "QueryRewriter: [%s] -> [%s]",
                    question[:80],
                    rewritten[:80],
                )
                return rewritten
        except Exception as exc:
            logger.warning("QueryRewriter failed (%s), using original query", exc)

        return question

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
