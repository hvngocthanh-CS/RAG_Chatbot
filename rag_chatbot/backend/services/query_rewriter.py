"""
Conversational Query Rewriter Service.

Converts follow-up questions into fully self-contained, standalone queries
so that the vector retrieval step works correctly even in multi-turn
conversations.

Industry pattern used by:
  - Cohere RAG pipeline (Grounding with rewrite)
  - LlamaIndex CondenseQuestionChatEngine
  - LangChain ConversationalRetrievalChain (we replicate the intent without
    the framework)

Example
-------
history  : [Q: "Tell me about the leave policy.", A: "..."]
follow-up: "How many days are allowed?"
rewritten: "How many leave days are employees allowed per year according to
            the company leave policy?"
"""

import logging
from typing import List, Dict, Optional

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

    Uses the same LLM backend as the chat service but with a lightweight
    call (low max_tokens, temperature=0) to minimise latency.

    Designed to be a no-op when there is no prior conversation history,
    so it is safe to use on every request.
    """

    def __init__(self):
        self._llm_client = None  # Lazy – set on first use via _get_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def rewrite(
        self,
        question: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Return a standalone version of *question* given *conversation_history*.

        Args:
            question:             The current user question (may be a follow-up).
            conversation_history: List of {"role": ..., "content": ...} dicts,
                                  oldest first.

        Returns:
            A rewritten, self-contained question string.  Falls back to the
            original question on any error so retrieval is never blocked.
        """
        # No history → nothing to resolve → return as-is
        if not conversation_history:
            return question

        # Only rewrite when there are actual prior turns
        prior_turns = [t for t in conversation_history if t.get("content")]
        if not prior_turns:
            return question

        # Quick heuristic: skip rewrite if no anaphora/reference words
        if not self._needs_rewrite(question):
            logger.debug("QueryRewriter: no anaphora detected, skipping rewrite")
            return question

        try:
            rewritten = await self._call_llm(question, prior_turns)
            rewritten = rewritten.strip().strip('"').strip("'")
            if rewritten:
                logger.info(
                    "QueryRewriter: [%s] → [%s]",
                    question[:80],
                    rewritten[:80],
                )
                return rewritten
        except Exception as exc:
            # Rewriter errors must NEVER break retrieval
            logger.warning("QueryRewriter failed (%s), using original query", exc)

        return question

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _ANAPHORA = {
        "it", "its", "they", "them", "their", "those", "these",
        "that", "this", "he", "she", "his", "her", "there",
        "which", "such", "the same", "the above", "aforementioned",
        "the previous", "the latter", "the former",
        # domain-specific shortcuts
        "3 people", "2 people", "those people", "those authors",
        "those documents", "that policy", "that procedure",
    }

    def _needs_rewrite(self, question: str) -> bool:
        """Cheap heuristic: does the question contain any anaphoric token?"""
        lower = question.lower()
        tokens = set(lower.split())
        # word-level overlap
        if tokens & self._ANAPHORA:
            return True
        # phrase-level
        return any(phrase in lower for phrase in self._ANAPHORA if " " in phrase)

    def _format_history(self, turns: List[Dict]) -> str:
        """Format the last N turns for the prompt."""
        # Keep last 6 turns (3 exchanges) to stay under token budget
        recent = turns[-6:]
        lines = []
        for t in recent:
            role = t.get("role", "user").capitalize()
            content = t.get("content", "")[:400]  # truncate long turns
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    async def _call_llm(self, question: str, history: List[Dict]) -> str:
        """Call the LLM with a lightweight rewrite prompt."""
        from backend.config import settings
        from openai import AsyncOpenAI

        client = await self._get_client(settings)

        prompt_user = _REWRITE_USER_TMPL.format(
            history=self._format_history(history),
            question=question,
        )

        model = (
            settings.VLLM_MODEL_NAME
            if settings.LLM_PROVIDER == "vllm"
            else settings.OLLAMA_MODEL
        )

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": prompt_user},
            ],
            temperature=0.0,   # deterministic
            max_tokens=120,    # short output – just the rewritten question
        )

        return response.choices[0].message.content or question

    async def _get_client(self, settings) -> "AsyncOpenAI":  # type: ignore[name-defined]
        """Lazy-initialise a thin AsyncOpenAI client (reuses provider config)."""
        if self._llm_client is not None:
            return self._llm_client

        from openai import AsyncOpenAI

        if settings.LLM_PROVIDER == "vllm":
            self._llm_client = AsyncOpenAI(
                base_url=settings.VLLM_BASE_URL,
                api_key=settings.VLLM_API_KEY,
                timeout=10,  # rewrite must be fast
            )
        else:
            self._llm_client = AsyncOpenAI(
                base_url=settings.OLLAMA_BASE_URL,
                api_key="ollama",
                timeout=10,
            )

        return self._llm_client
