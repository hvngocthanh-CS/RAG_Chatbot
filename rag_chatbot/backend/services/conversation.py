"""
Conversation Manager.
Handles multi-turn conversation memory.
"""
import uuid
import logging
from typing import List, Dict, Optional
from datetime import datetime
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation history for multi-turn interactions.
    
    Features:
    - In-memory conversation storage
    - Automatic cleanup of old conversations
    - Message limit per conversation
    
    Note: For production, consider using Redis or a database
    for persistent conversation storage.
    """
    
    # Class-level storage (shared across instances)
    _conversations: OrderedDict = OrderedDict()
    _max_conversations: int = 1000
    _max_messages_per_conversation: int = 50
    
    def __init__(self):
        pass
    
    def create_conversation(self) -> str:
        """
        Create a new conversation and return its ID.
        """
        conversation_id = str(uuid.uuid4())
        
        self._conversations[conversation_id] = {
            "id": conversation_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "messages": []
        }
        
        # Cleanup old conversations if limit exceeded
        self._cleanup_old_conversations()
        
        logger.debug(f"Created conversation: {conversation_id}")
        return conversation_id
    
    def get_history(self, conversation_id: str) -> Optional[List[Dict]]:
        """
        Get conversation history by ID.
        
        Returns None if conversation doesn't exist.
        """
        conversation = self._conversations.get(conversation_id)
        
        if conversation:
            return conversation["messages"]
        
        return None
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str
    ):
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            role: "user" or "assistant"
            content: Message content
        """
        if conversation_id not in self._conversations:
            # Create conversation if it doesn't exist
            self._conversations[conversation_id] = {
                "id": conversation_id,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "messages": []
            }
        
        conversation = self._conversations[conversation_id]
        
        # Add message
        conversation["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        conversation["updated_at"] = datetime.utcnow().isoformat()
        
        # Trim old messages if limit exceeded
        if len(conversation["messages"]) > self._max_messages_per_conversation:
            conversation["messages"] = conversation["messages"][-self._max_messages_per_conversation:]
        
        # Move to end of OrderedDict (most recently used)
        self._conversations.move_to_end(conversation_id)
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Returns True if deleted, False if not found.
        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            logger.debug(f"Deleted conversation: {conversation_id}")
            return True
        
        return False
    
    def get_conversation_summary(self, conversation_id: str) -> Optional[Dict]:
        """
        Get a summary of a conversation (without full message content).
        """
        conversation = self._conversations.get(conversation_id)
        
        if conversation:
            return {
                "id": conversation["id"],
                "created_at": conversation["created_at"],
                "updated_at": conversation["updated_at"],
                "message_count": len(conversation["messages"])
            }
        
        return None
    
    def list_conversations(self, limit: int = 20) -> List[Dict]:
        """
        List recent conversations (summaries only).
        """
        conversations = []
        
        # Get most recent conversations
        for conv_id in reversed(list(self._conversations.keys())[:limit]):
            summary = self.get_conversation_summary(conv_id)
            if summary:
                conversations.append(summary)
        
        return conversations
    
    def _cleanup_old_conversations(self):
        """
        Remove oldest conversations if limit exceeded.
        """
        while len(self._conversations) > self._max_conversations:
            # Remove oldest (first item in OrderedDict)
            oldest_id = next(iter(self._conversations))
            del self._conversations[oldest_id]
            logger.debug(f"Cleaned up old conversation: {oldest_id}")
    
    def clear_all(self):
        """
        Clear all conversations.
        """
        self._conversations.clear()
        logger.info("All conversations cleared")
    
    @classmethod
    def get_stats(cls) -> Dict:
        """
        Get conversation manager statistics.
        """
        return {
            "total_conversations": len(cls._conversations),
            "max_conversations": cls._max_conversations,
            "max_messages_per_conversation": cls._max_messages_per_conversation
        }
