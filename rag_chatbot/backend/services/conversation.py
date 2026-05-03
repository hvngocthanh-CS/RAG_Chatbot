"""
Conversation Manager.
Handles multi-turn conversation memory (in-memory backend).
"""
import uuid
import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    In-memory conversation manager for multi-turn interactions.

    Drop-in replacement when no DATABASE_URL is configured.
    All public methods are async to share the same interface as
    PostgreSQLConversationManager, making them interchangeable at runtime.
    """

    # Class-level storage shared across instances
    _conversations: OrderedDict = OrderedDict()
    _max_conversations: int = 1000
    _max_messages_per_conversation: int = 50

    async def create_conversation(self) -> str:
        conversation_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._conversations[conversation_id] = {
            "id": conversation_id,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self._cleanup_old_conversations()
        logger.debug("Created conversation: %s", conversation_id)
        return conversation_id

    async def get_history(self, conversation_id: str) -> Optional[List[Dict]]:
        conversation = self._conversations.get(conversation_id)
        return conversation["messages"] if conversation else None

    async def add_message(self, conversation_id: str, role: str, content: str):
        now = datetime.now(timezone.utc).isoformat()
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = {
                "id": conversation_id,
                "created_at": now,
                "updated_at": now,
                "messages": [],
            }

        conversation = self._conversations[conversation_id]
        conversation["messages"].append({"role": role, "content": content, "timestamp": now})
        conversation["updated_at"] = now

        if len(conversation["messages"]) > self._max_messages_per_conversation:
            conversation["messages"] = conversation["messages"][-self._max_messages_per_conversation:]

        self._conversations.move_to_end(conversation_id)

    async def delete_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            logger.debug("Deleted conversation: %s", conversation_id)
            return True
        return False

    async def get_conversation_summary(self, conversation_id: str) -> Optional[Dict]:
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            return None
        return {
            "id": conversation["id"],
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "message_count": len(conversation["messages"]),
        }

    async def list_conversations(self, limit: int = 20) -> List[Dict]:
        results = []
        for conv_id in reversed(list(self._conversations.keys())[:limit]):
            summary = await self.get_conversation_summary(conv_id)
            if summary:
                results.append(summary)
        return results

    async def clear_all(self):
        self._conversations.clear()
        logger.info("All conversations cleared")

    def _cleanup_old_conversations(self):
        while len(self._conversations) > self._max_conversations:
            oldest_id = next(iter(self._conversations))
            del self._conversations[oldest_id]
            logger.debug("Cleaned up old conversation: %s", oldest_id)

    @classmethod
    def get_stats(cls) -> Dict:
        return {
            "backend": "in-memory",
            "total_conversations": len(cls._conversations),
            "max_conversations": cls._max_conversations,
            "max_messages_per_conversation": cls._max_messages_per_conversation,
        }
