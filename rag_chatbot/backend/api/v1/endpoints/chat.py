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

logger = logging.getLogger(__name__)
router = APIRouter()


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
            return cached

    # Conversation
    conversation_manager = get_service("conversation")
    conversation_id = request.conversation_id or conversation_manager.create_conversation()
    conversation_history = conversation_manager.get_history(conversation_id) or []

    # Retrieve
    retrieved_chunks = await retrieval_service.retrieve(
        query=request.question,
        filters=request.filters,
        conversation_history=conversation_history,
    )

    if not retrieved_chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found. Please upload documents first.",
        )

    context = _format_context(retrieved_chunks)

    # Streaming response
    if request.stream:
        return StreamingResponse(
            _stream_response(
                llm_service=llm_service,
                question=request.question,
                context=context,
                conversation_history=conversation_history,
                conversation_id=conversation_id,
                retrieved_chunks=retrieved_chunks,
                conversation_manager=conversation_manager,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response
    answer = await llm_service.generate(
        question=request.question,
        context=context,
        conversation_history=conversation_history,
    )

    conversation_manager.add_message(conversation_id, "user", request.question)
    conversation_manager.add_message(conversation_id, "assistant", answer)

    processing_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

    sources = [
        SourceChunk(
            content=chunk["content"][:500],
            document_id=chunk["metadata"].get("document_id", ""),
            document_name=chunk["metadata"].get("filename", ""),
            page_number=chunk["metadata"].get("page_number"),
            chunk_type=chunk["metadata"].get("chunk_type", "text"),
            relevance_score=chunk.get("score", 0.0),
        )
        for chunk in retrieved_chunks
    ]

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
    history = conversation_manager.get_history(conversation_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation_id": conversation_id, "messages": history}


@router.delete("/chat/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    conversation_manager = get_service("conversation")
    if not conversation_manager.delete_conversation(conversation_id):
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
):
    """SSE generator for streaming response."""
    sources = [
        {
            "content": chunk["content"][:500],
            "document_id": chunk["metadata"].get("document_id", ""),
            "document_name": chunk["metadata"].get("filename", ""),
            "page_number": chunk["metadata"].get("page_number"),
            "chunk_type": chunk["metadata"].get("chunk_type", "text"),
            "relevance_score": chunk.get("score", 0.0),
        }
        for chunk in retrieved_chunks
    ]
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

        conversation_manager.add_message(conversation_id, "user", question)
        conversation_manager.add_message(
            conversation_id, "assistant",
            full_answer if full_answer else (error_message or ""),
        )

        done_payload = {"type": "done"}
        if error_message:
            done_payload["error"] = error_message
        yield f"data: {json.dumps(done_payload)}\n\n"


def _format_context(chunks: List[dict]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
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

        type_label = ", Type: Table" if chunk_type in ("table", "table_rows") else ""
        header = f"[Source {i}: {source}, Page {page}{type_label}{annotation_str}]"
        parts.append(f"{header}\n{chunk['content']}\n")

    return "\n---\n".join(parts)
