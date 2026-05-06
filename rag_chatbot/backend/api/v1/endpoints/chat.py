"""
Chat endpoints with streaming support.
"""
import json
import logging
from typing import List
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.config import settings
from backend.services import get_service
from backend.api.v1.schemas.chat import ChatRequest, ChatResponse, SourceChunk
from backend.core.metrics import (
    RETRIEVAL_DURATION, LLM_DURATION, RETRIEVED_CHUNKS,
    CACHE_HITS, CACHE_MISSES, ACTIVE_CONVERSATIONS,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _build_source_chunks(chunks: List[dict]) -> List[dict]:
    """Build source metadata from retrieved chunks."""
    return [
        {
            "content": chunk["content"][:500],
            "document_id": chunk["metadata"].get("document_id", ""),
            "document_name": chunk["metadata"].get("filename", ""),
            "page_number": chunk["metadata"].get("page_number"),
            "chunk_type": chunk["metadata"].get("chunk_type", "text"),
            "relevance_score": chunk.get("score", 0.0),
        }
        for chunk in chunks
    ]


@router.post("/chat")
async def chat(request: ChatRequest):
    """Send a question and receive an answer based on indexed documents."""
    start_time = datetime.now(timezone.utc)

    retrieval_service = get_service("retrieval")
    llm_service = get_service("llm")
    cache_service = get_service("cache")

    # Check cache
    if cache_service:
        cached = await cache_service.get_cached_response(request.question, request.filters)
        if cached:
            logger.info("Cache hit for: %s", request.question[:50])
            CACHE_HITS.inc()
            return cached
    CACHE_MISSES.inc()

    # Conversation
    conversation_manager = get_service("conversation")
    conversation_id = request.conversation_id or await conversation_manager.create_conversation()
    conversation_history = await conversation_manager.get_history(conversation_id) or []
    ACTIVE_CONVERSATIONS.set(len(conversation_manager._conversations))

    # Retrieve
    with RETRIEVAL_DURATION.time():
        retrieval_result = await retrieval_service.retrieve(
            query=request.question,
            filters=request.filters,
            conversation_history=conversation_history,
        )
    retrieved_chunks = retrieval_result["chunks"]
    sub_questions = retrieval_result["sub_questions"]
    RETRIEVED_CHUNKS.observe(len(retrieved_chunks))

    if not retrieved_chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found. Please upload documents first.",
        )

    context = _format_context(retrieved_chunks)
    # When the query was decomposed, pass the sub-question checklist to the
    # LLM so it addresses every aspect explicitly rather than blending partial
    # context across sub-questions and hallucinating the missing pieces.
    effective_question = _build_effective_question(request.question, sub_questions)

    # Streaming response
    if request.stream:
        return StreamingResponse(
            _stream_response(
                llm_service=llm_service,
                question=effective_question,
                context=context,
                conversation_history=conversation_history,
                conversation_id=conversation_id,
                retrieved_chunks=retrieved_chunks,
                conversation_manager=conversation_manager,
                original_question=request.question,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response
    with LLM_DURATION.time():
        answer = await llm_service.generate(
            question=effective_question,
            context=context,
            conversation_history=conversation_history,
        )

    await conversation_manager.add_message(conversation_id, "user", request.question)  # store original
    await conversation_manager.add_message(conversation_id, "assistant", answer)

    processing_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

    sources = [SourceChunk(**s) for s in _build_source_chunks(retrieved_chunks)]

    response = ChatResponse(
        answer=answer,
        conversation_id=conversation_id,
        sources=sources,
        processing_time_ms=processing_time,
    )

    if cache_service:
        await cache_service.cache_response(
            request.question, request.filters, response.model_dump(),
        )

    return response


@router.get("/chat/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    conversation_manager = get_service("conversation")
    history = await conversation_manager.get_history(conversation_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "messages": history}


@router.delete("/chat/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    conversation_manager = get_service("conversation")
    if not await conversation_manager.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "success", "message": "Conversation deleted"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _stream_response(
    llm_service,
    question: str,
    context: str,
    conversation_history: List[dict],
    conversation_id: str,
    retrieved_chunks: List[dict],
    conversation_manager,
    original_question: str = None,
):
    """SSE generator for streaming response."""
    sources = _build_source_chunks(retrieved_chunks)
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'conversation_id': conversation_id})}\n\n"

    full_answer = ""
    error_message = None
    sent_token = False

    try:
        async for token in llm_service.generate_stream(
            question=question,
            context=context,
            conversation_history=conversation_history,
        ):
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            sent_token = True
    except Exception as e:
        logger.error("LLM streaming failed: %s", e)
        error_message = str(e)
    finally:
        if not sent_token:
            fallback = error_message or "No relevant answer found in the provided documents."
            yield f"data: {json.dumps({'type': 'token', 'content': fallback})}\n\n"

        stored_question = original_question if original_question is not None else question
        await conversation_manager.add_message(conversation_id, "user", stored_question)
        await conversation_manager.add_message(
            conversation_id, "assistant",
            full_answer if full_answer else (error_message or ""),
        )

        done_payload = {"type": "done"}
        if error_message:
            done_payload["error"] = error_message
        yield f"data: {json.dumps(done_payload)}\n\n"


def _build_effective_question(question: str, sub_questions: List[str]) -> str:
    """
    For decomposed queries, append an explicit sub-question checklist so the
    LLM addresses every aspect rather than blending partial context and
    hallucinating the missing pieces.
    """
    if not sub_questions:
        return question
    checklist = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(sub_questions))
    return (
        f"{question}\n\n"
        f"[Address each of the following aspects explicitly:\n{checklist}]"
    )


def _format_context(chunks: List[dict]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    n = len(chunks)
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source = meta.get("filename", "Unknown")
        page = meta.get("page_number", "N/A")
        chunk_type = meta.get("chunk_type", "text")

        annotations = []
        if meta.get("department"):
            annotations.append(f"Dept: {meta['department']}")
        if meta.get("category"):
            annotations.append(f"Category: {meta['category']}")
        if meta.get("version"):
            annotations.append(f"Version: {meta['version']}")
        annotation_str = (" | " + " | ".join(annotations)) if annotations else ""

        type_label = (
            ", Type: Table"
            if chunk_type in ("table", "table_rows", "table_summary")
            else ""
        )
        header = f"[Source {i}: {source}, Page {page}{type_label}{annotation_str}]"
        parts.append(f"{header}\n{chunk['content']}\n")

    # Explicit source-range header so the LLM cannot hallucinate source numbers
    # outside the valid range (e.g. citing Source 17 when only 5 were retrieved).
    range_header = (
        f"[{n} source{'s' if n != 1 else ''} retrieved | "
        f"valid citations: [Source 1] through [Source {n}] ONLY]\n\n"
    )
    return range_header + "\n---\n".join(parts)
