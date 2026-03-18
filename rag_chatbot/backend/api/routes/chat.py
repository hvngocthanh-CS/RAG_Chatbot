"""
Chat endpoints with streaming support.
"""
import json
import logging
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config import settings
from backend.services.retrieval import RetrievalService
from backend.services.llm import LLMService
from backend.services.cache import CacheService
from backend.services.conversation import ConversationManager

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Chat request model."""
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    stream: bool = True
    # Metadata filters passed straight to the vector store.
    # Supported keys: department, category, author, version, doc_date,
    #                 document_id, tags (substring match).
    # Example: {"department": "HR", "category": ["Policy", "SOP"]}
    filters: Optional[dict] = None


class SourceChunk(BaseModel):
    """Retrieved source chunk model."""
    content: str
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    chunk_type: str  # "text" or "table"
    relevance_score: float


class ChatResponse(BaseModel):
    """Chat response model (non-streaming)."""
    answer: str
    conversation_id: str
    sources: List[SourceChunk]
    processing_time_ms: int


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Send a question and receive an answer based on indexed documents.
    
    Supports both streaming and non-streaming responses.
    """
    start_time = datetime.utcnow()
    
    # Check cache first (if enabled)
    cache_service = CacheService()
    if settings.USE_CACHE:
        cached_response = await cache_service.get_cached_response(
            request.question, 
            request.filters
        )
        if cached_response:
            logger.info(f"Cache hit for question: {request.question[:50]}...")
            return cached_response
    
    # Get or create conversation
    conversation_manager = ConversationManager()
    conversation_id = request.conversation_id or conversation_manager.create_conversation()
    conversation_history = conversation_manager.get_history(conversation_id)
    
    # Retrieve relevant documents
    retrieval_service = RetrievalService()
    retrieved_chunks = await retrieval_service.retrieve(
        query=request.question,
        filters=request.filters,
        conversation_history=conversation_history
    )
    
    if not retrieved_chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found. Please upload documents first."
        )
    
    # Prepare context from retrieved chunks
    context = format_context(retrieved_chunks)
    
    # Generate response using LLM
    llm_service = LLMService()
    
    if request.stream:
        # Return streaming response
        return StreamingResponse(
            stream_response(
                llm_service=llm_service,
                question=request.question,
                context=context,
                conversation_history=conversation_history,
                conversation_id=conversation_id,
                retrieved_chunks=retrieved_chunks,
                conversation_manager=conversation_manager
            ),
            media_type="text/event-stream"
        )
    else:
        # Generate full response
        answer = await llm_service.generate(
            question=request.question,
            context=context,
            conversation_history=conversation_history
        )
        
        # Update conversation history
        conversation_manager.add_message(conversation_id, "user", request.question)
        conversation_manager.add_message(conversation_id, "assistant", answer)
        
        # Calculate processing time
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Format sources
        sources = [
            SourceChunk(
                content=chunk["content"][:500],
                document_id=chunk["metadata"]["document_id"],
                document_name=chunk["metadata"]["filename"],
                page_number=chunk["metadata"].get("page_number"),
                chunk_type=chunk["metadata"].get("chunk_type", "text"),
                relevance_score=chunk["score"]
            )
            for chunk in retrieved_chunks
        ]
        
        response = ChatResponse(
            answer=answer,
            conversation_id=conversation_id,
            sources=sources,
            processing_time_ms=processing_time
        )
        
        # Cache the response
        if settings.USE_CACHE:
            await cache_service.cache_response(
                request.question,
                request.filters,
                response.model_dump()
            )
        
        return response


async def stream_response(
    llm_service: LLMService,
    question: str,
    context: str,
    conversation_history: List[dict],
    conversation_id: str,
    retrieved_chunks: List[dict],
    conversation_manager: ConversationManager
):
    """Generator for streaming response."""
    
    # First, send sources
    sources = [
        {
            "content": chunk["content"][:500],
            "document_id": chunk["metadata"]["document_id"],
            "document_name": chunk["metadata"]["filename"],
            "page_number": chunk["metadata"].get("page_number"),
            "chunk_type": chunk["metadata"].get("chunk_type", "text"),
            "relevance_score": chunk["score"]
        }
        for chunk in retrieved_chunks
    ]
    
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'conversation_id': conversation_id})}\n\n"
    
    # Stream the answer with error handling
    full_answer = ""
    error_message = None
    sent_token = False
    try:
        logger.info(f"[DEBUG] LLM context for question: {question[:50]}...\nContext: {context[:200]}...")
        async for token in llm_service.generate_stream(
            question=question,
            context=context,
            conversation_history=conversation_history
        ):
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            sent_token = True
    except Exception as e:
        logger.error(f"LLM streaming failed: {e}")
        error_message = str(e)
    finally:
        # If no token was sent, send a fallback token (for FE display)
        if not sent_token:
            fallback_content = error_message or "Không có câu trả lời phù hợp."
            yield f"data: {json.dumps({'type': 'token', 'content': fallback_content})}\n\n"
        # Update conversation history
        conversation_manager.add_message(conversation_id, "user", question)
        conversation_manager.add_message(conversation_id, "assistant", full_answer if full_answer else (error_message or ""))
        # Always send completion signal, include error if any
        if error_message:
            yield f"data: {json.dumps({'type': 'done', 'error': error_message})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"


def format_context(chunks: List[dict]) -> str:
    """
    Format retrieved chunks into a context block sent to the LLM.

    Each chunk header contains enough metadata for the model to produce
    accurate citations (filename, page, category, department, version).
    """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        chunk_type = meta.get("chunk_type", "text")
        source = meta.get("filename", "Unknown")
        page = meta.get("page_number", "N/A")

        # Build optional annotation parts
        annotations = []
        if meta.get("department"):
            annotations.append(f"Dept: {meta['department']}")
        if meta.get("category"):
            annotations.append(f"Category: {meta['category']}")
        if meta.get("version"):
            annotations.append(f"Version: {meta['version']}")
        if meta.get("doc_date"):
            annotations.append(f"Date: {meta['doc_date']}")
        annotation_str = (" | " + " | ".join(annotations)) if annotations else ""

        type_label = ", Type: Table" if chunk_type in ("table", "table_rows") else ""
        header = f"[Source {i}: {source}, Page {page}{type_label}{annotation_str}]"

        context_parts.append(f"{header}\n{chunk['content']}\n")

    return "\n---\n".join(context_parts)


@router.get("/chat/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    conversation_manager = ConversationManager()
    history = conversation_manager.get_history(conversation_id)
    
    if history is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"conversation_id": conversation_id, "messages": history}


@router.delete("/chat/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    conversation_manager = ConversationManager()
    success = conversation_manager.delete_conversation(conversation_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"status": "success", "message": "Conversation deleted"}
