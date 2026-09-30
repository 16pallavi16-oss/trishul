"""
Durable, queryable conversation history in Postgres -- two tables, `thread`
and `message`. This is separate from chat.py's Redis-backed history:
Redis stays the fast lookup used to build each prompt (unchanged, same
7-day TTL); Postgres is the permanent record that never expires and can
be queried directly (list threads, audit a conversation, etc.).

Deliberately does NOT import .db, even though it also uses SQLModel --
db.py pulls in .ingest, which pulls in unstructured/pypdfium2/docTR, a
heavy dependency chain with no reason to load just to store chat messages
in a request-serving process.
"""
import os
from datetime import datetime, timezone
from typing import List, Optional

from sqlmodel import SQLModel, Field, Session, create_engine, select
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Create a .env file in the project root "
        "(see .env.example)."
    )

engine = create_engine(DATABASE_URL, echo=False)


class Thread(SQLModel, table=True):
    __tablename__ = "thread"

    id: str = Field(primary_key=True)  # the thread_id the client generates
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Message(SQLModel, table=True):
    __tablename__ = "message"

    id: Optional[int] = Field(default=None, primary_key=True)
    thread_id: str = Field(foreign_key="thread.id", index=True)
    role: str  # "system" | "user" | "assistant"
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Feedback(SQLModel, table=True):
    __tablename__ = "feedback"

    id: Optional[int] = Field(default=None, primary_key=True)
    message_id: int = Field(foreign_key="message.id", index=True)
    rating: str  # "up" | "down"
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def init_thread_db() -> None:
    # tables= scopes this to just these two, regardless of what else might
    # be registered on SQLModel's shared metadata in the same process
    SQLModel.metadata.create_all(
        engine, tables=[Thread.__table__, Message.__table__, Feedback.__table__]
    )


def ensure_thread(thread_id: str, session: Session) -> Thread:
    thread = session.get(Thread, thread_id)
    if thread is None:
        thread = Thread(id=thread_id)
    else:
        thread.updated_at = datetime.now(timezone.utc)
    session.add(thread)
    session.commit()
    return thread


def save_message(thread_id: str, role: str, content: str) -> int:
    """Returns the new message's id, so feedback can be attached to it later."""
    with Session(engine) as session:
        ensure_thread(thread_id, session)
        msg = Message(thread_id=thread_id, role=role, content=content)
        session.add(msg)
        session.commit()
        session.refresh(msg)
        return msg.id


def get_messages(thread_id: str, limit: Optional[int] = None) -> List[dict]:
    with Session(engine) as session:
        if limit:
            statement = (
                select(Message)
                .where(Message.thread_id == thread_id)
                .order_by(Message.id.desc())
                .limit(limit)
            )
            rows = list(session.exec(statement))[::-1]  # back to chronological order
        else:
            statement = select(Message).where(Message.thread_id == thread_id).order_by(Message.id)
            rows = list(session.exec(statement))
    return [{"role": m.role, "content": m.content} for m in rows]


def list_threads(limit: int = 50) -> List[dict]:
    with Session(engine) as session:
        statement = select(Thread).order_by(Thread.updated_at.desc()).limit(limit)
        rows = session.exec(statement).all()
    return [
        {"id": t.id, "created_at": t.created_at.isoformat(), "updated_at": t.updated_at.isoformat()}
        for t in rows
    ]


def clear_thread(thread_id: str) -> int:
    """
    Deletes every message in a thread (and any feedback attached to them),
    but keeps the Thread row itself so the id stays valid for continued use.
    Returns how many messages were removed.
    """
    with Session(engine) as session:
        messages = session.exec(select(Message).where(Message.thread_id == thread_id)).all()
        message_ids = [m.id for m in messages]

        if message_ids:
            # feedback rows reference message.id, so they must go first or
            # the foreign key constraint will reject the message deletes
            feedback = session.exec(
                select(Feedback).where(Feedback.message_id.in_(message_ids))
            ).all()
            for f in feedback:
                session.delete(f)

        for m in messages:
            session.delete(m)
        session.commit()

    return len(message_ids)


def save_feedback(message_id: int, rating: str, comment: Optional[str] = None) -> int:
    if rating not in ("up", "down"):
        raise ValueError(f"rating must be 'up' or 'down', got {rating!r}")
    with Session(engine) as session:
        fb = Feedback(message_id=message_id, rating=rating, comment=comment)
        session.add(fb)
        session.commit()
        session.refresh(fb)
        return fb.id


def get_feedback_summary() -> dict:
    """Counts of up/down across all messages -- a quick quality signal."""
    with Session(engine) as session:
        rows = session.exec(select(Feedback)).all()
    return {
        "up": sum(1 for r in rows if r.rating == "up"),
        "down": sum(1 for r in rows if r.rating == "down"),
        "total": len(rows),
    }