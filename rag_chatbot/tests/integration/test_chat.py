"""Integration tests — chat and conversation endpoints."""
import json
import pytest


# ---------------------------------------------------------------------------
# Non-streaming chat
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_returns_answer(client):
    response = await client.post(
        "/api/v1/chat",
        json={"question": "What is the onboarding policy?", "stream": False},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The answer is based on the uploaded documents."
    assert "conversation_id" in body
    assert len(body["sources"]) == 1
    assert "processing_time_ms" in body


@pytest.mark.asyncio
async def test_chat_caches_response(client, mock_cache):
    await client.post(
        "/api/v1/chat",
        json={"question": "Cache this response", "stream": False},
    )
    mock_cache.cache_response.assert_called_once()


@pytest.mark.asyncio
async def test_chat_returns_cached_response(client, mock_cache):
    cached = {
        "answer": "Cached answer",
        "conversation_id": "conv-cached",
        "sources": [],
        "processing_time_ms": 10,
    }
    mock_cache.get_cached_response.return_value = cached

    response = await client.post(
        "/api/v1/chat",
        json={"question": "Anything", "stream": False},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "Cached answer"


@pytest.mark.asyncio
async def test_chat_no_documents_returns_404(client, mock_retrieval):
    mock_retrieval.retrieve.return_value = {"chunks": [], "sub_questions": []}
    response = await client.post(
        "/api/v1/chat",
        json={"question": "Any question", "stream": False},
    )
    assert response.status_code == 404
    assert "No relevant documents" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Streaming chat (SSE)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_streaming_sse_events(client):
    events = []
    async with client.stream(
        "POST",
        "/api/v1/chat",
        json={"question": "Streaming question?", "stream": True},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    event_types = [e["type"] for e in events]
    assert "sources" in event_types
    assert "token" in event_types
    assert event_types[-1] == "done"


@pytest.mark.asyncio
async def test_chat_streaming_sources_contain_conversation_id(client):
    events = []
    async with client.stream(
        "POST",
        "/api/v1/chat",
        json={"question": "Who am I?", "stream": True},
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    sources_event = next(e for e in events if e["type"] == "sources")
    assert "conversation_id" in sources_event
    assert len(sources_event["sources"]) == 1


@pytest.mark.asyncio
async def test_chat_streaming_tokens_compose_full_answer(client):
    events = []
    async with client.stream(
        "POST",
        "/api/v1/chat",
        json={"question": "Compose tokens", "stream": True},
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    full = "".join(e["content"] for e in events if e["type"] == "token")
    assert full == "The answer is here."


# ---------------------------------------------------------------------------
# Conversation persistence (within a test session)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_creates_new_conversation(client):
    r1 = await client.post(
        "/api/v1/chat",
        json={"question": "First question", "stream": False},
    )
    r2 = await client.post(
        "/api/v1/chat",
        json={"question": "Second question", "stream": False},
    )
    assert r1.json()["conversation_id"] != r2.json()["conversation_id"]


@pytest.mark.asyncio
async def test_chat_continues_existing_conversation(client):
    r1 = await client.post(
        "/api/v1/chat",
        json={"question": "First question", "stream": False},
    )
    conv_id = r1.json()["conversation_id"]

    r2 = await client.post(
        "/api/v1/chat",
        json={"question": "Follow-up question", "conversation_id": conv_id, "stream": False},
    )
    assert r2.json()["conversation_id"] == conv_id


@pytest.mark.asyncio
async def test_get_conversation_history(client):
    chat_resp = await client.post(
        "/api/v1/chat",
        json={"question": "What is the policy?", "stream": False},
    )
    conv_id = chat_resp.json()["conversation_id"]

    history_resp = await client.get(f"/api/v1/chat/conversations/{conv_id}")
    assert history_resp.status_code == 200
    body = history_resp.json()
    assert body["conversation_id"] == conv_id
    assert len(body["messages"]) == 2  # user + assistant
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_get_conversation_not_found(client):
    response = await client.get("/api/v1/chat/conversations/no-such-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_conversation(client):
    chat_resp = await client.post(
        "/api/v1/chat",
        json={"question": "Delete me", "stream": False},
    )
    conv_id = chat_resp.json()["conversation_id"]

    del_resp = await client.delete(f"/api/v1/chat/conversations/{conv_id}")
    assert del_resp.status_code == 200

    get_resp = await client.get(f"/api/v1/chat/conversations/{conv_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_conversation(client):
    response = await client.delete("/api/v1/chat/conversations/ghost-conv")
    assert response.status_code == 404
