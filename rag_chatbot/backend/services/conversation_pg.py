"""
PostgreSQL-backed Conversation Manager using SQLAlchemy async.

Activated automatically when DATABASE_URL is set in settings.
Provides the same async interface as the in-memory ConversationManager,
so both are interchangeable at runtime without touching endpoint code.
"""
import uuid
import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone

from sqlalchemy import String, Text, DateTime, ForeignKey, select, delete, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, selectinload

logger = logging.getLogger(__name__)

UTC = timezone.utc


class _Base(DeclarativeBase):
    pass


class _Conversation(_Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    messages: Mapped[List["_Message"]] = relationship(
        back_populates="conversation",
        order_by="asc(_Message.id)",
        cascade="all, delete-orphan",
    )


class _Message(_Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    conversation: Mapped["_Conversation"] = relationship(back_populates="messages")


class PostgreSQLConversationManager:
    """
    Async PostgreSQL conversation manager.

    Same public interface as ConversationManager (in-memory) — all public
    methods are async and accept the same arguments.
    """

    _max_messages_per_conversation: int = 50

    def __init__(self, database_url: str) -> None:
        self._engine = create_async_engine(database_url, pool_pre_ping=True, pool_size=5)
        self._session = async_sessionmaker(self._engine, expire_on_commit=False)

    async def initialize(self) -> None:
        """Create tables if they do not exist. Call once at startup."""
        async with self._engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)
        logger.info("PostgreSQL conversation manager initialized")

    async def close(self) -> None:
        await self._engine.dispose()
        logger.info("PostgreSQL connection pool closed")

    async def health_check(self) -> bool:
        try:
            async with self._session() as session:
                await session.execute(select(func.count()).select_from(_Conversation))
            return True
        except Exception as exc:
            logger.warning("PostgreSQL health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Public API (mirrors ConversationManager)
    # ------------------------------------------------------------------

    async def create_conversation(self) -> str:
        conversation_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        async with self._session() as session:
            session.add(_Conversation(id=conversation_id, created_at=now, updated_at=now))
            await session.commit()
        logger.debug("Created conversation: %s", conversation_id)
        return conversation_id

    async def get_history(self, conversation_id: str) -> Optional[List[Dict]]:
        async with self._session() as session:
            result = await session.execute(
                select(_Conversation)
                .where(_Conversation.id == conversation_id)
                .options(selectinload(_Conversation.messages))
            )
            conv = result.scalar_one_or_none()
        if conv is None:
            return None
        return [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()}
            for m in conv.messages
        ]

    async def add_message(self, conversation_id: str, role: str, content: str) -> None:
        now = datetime.now(UTC)
        async with self._session() as session:
            result = await session.execute(
                select(_Conversation).where(_Conversation.id == conversation_id)
            )
            conv = result.scalar_one_or_none()
            if conv is None:
                conv = _Conversation(id=conversation_id, created_at=now, updated_at=now)
                session.add(conv)
            else:
                conv.updated_at = now

            session.add(_Message(
                conversation_id=conversation_id, role=role, content=content, timestamp=now
            ))
            await session.flush()

            # Trim messages beyond the per-conversation cap
            count = await session.scalar(
                select(func.count()).select_from(_Message)
                .where(_Message.conversation_id == conversation_id)
            )
            if count > self._max_messages_per_conversation:
                excess = count - self._max_messages_per_conversation
                oldest_ids = await session.scalars(
                    select(_Message.id)
                    .where(_Message.conversation_id == conversation_id)
                    .order_by(_Message.id)
                    .limit(excess)
                )
                await session.execute(
                    delete(_Message).where(_Message.id.in_(oldest_ids.all()))
                )

            await session.commit()

    async def delete_conversation(self, conversation_id: str) -> bool:
        async with self._session() as session:
            result = await session.execute(
                select(_Conversation).where(_Conversation.id == conversation_id)
            )
            conv = result.scalar_one_or_none()
            if conv is None:
                return False
            await session.delete(conv)
            await session.commit()
        logger.debug("Deleted conversation: %s", conversation_id)
        return True

    async def get_conversation_summary(self, conversation_id: str) -> Optional[Dict]:
        async with self._session() as session:
            result = await session.execute(
                select(_Conversation)
                .where(_Conversation.id == conversation_id)
                .options(selectinload(_Conversation.messages))
            )
            conv = result.scalar_one_or_none()
        if conv is None:
            return None
        return {
            "id": conv.id,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
            "message_count": len(conv.messages),
        }

    async def list_conversations(self, limit: int = 20) -> List[Dict]:
        async with self._session() as session:
            result = await session.execute(
                select(_Conversation)
                .options(selectinload(_Conversation.messages))
                .order_by(_Conversation.updated_at.desc())
                .limit(limit)
            )
            convs = result.scalars().all()
        return [
            {
                "id": c.id,
                "created_at": c.created_at.isoformat(),
                "updated_at": c.updated_at.isoformat(),
                "message_count": len(c.messages),
            }
            for c in convs
        ]

    async def clear_all(self) -> None:
        async with self._session() as session:
            await session.execute(delete(_Conversation))
            await session.commit()
        logger.info("All conversations cleared")

    @staticmethod
    def get_stats() -> Dict:
        return {"backend": "postgresql"}
