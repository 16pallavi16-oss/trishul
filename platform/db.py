import hashlib
import mimetypes
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from sqlmodel import SQLModel, Field, Session, create_engine, select

from .ingest import ingest_folder, SUPPORTED_EXTS

DATABASE_URL = "postgresql://myuser:mypassword@localhost:5432/mydb"

engine = create_engine(DATABASE_URL, echo=False)

class SourceFile(SQLModel, table=True):
    __tablename__ = "source_file"

    id: Optional[int] = Field(default=None, primary_key=True)
    path: str = Field(index=True, unique=True)
    hash: str = Field(index=True)          # sha256 of file contents
    mime: Optional[str] = None
    page_count: Optional[int] = None
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Chunk(SQLModel, table=True):
    __tablename__ = "chunk"

    id: Optional[int] = Field(default=None, primary_key=True)
    file_id: int = Field(foreign_key="source_file.id", index=True)
    chunk_index: int
    text: str
    page: Optional[int] = None
    bbox: Optional[str] = None   # stored as JSON string: [x0, y0, x1, y1]

def _file_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


def _guess_mime(path: Path) -> Optional[str]:
    mime, _ = mimetypes.guess_type(str(path))
    return mime


def _page_count_from_chunks(file_chunks: List[Dict[str, Any]]) -> Optional[int]:
    pages = [c["page_number"] for c in file_chunks if c.get("page_number") is not None]
    return max(pages) if pages else None


def _extract_bbox(chunk: Dict[str, Any]) -> Optional[str]:
    bbox = chunk.get("bbox")
    return json.dumps(bbox) if bbox else None

def init_db() -> None:
    SQLModel.metadata.create_all(engine)


def persist_file_and_chunks(path: Path, file_chunks: List[Dict[str, Any]], session: Session) -> None:
    file_hash = _file_hash(path)

    existing = session.exec(
        select(SourceFile).where(SourceFile.path == str(path))
    ).first()

    if existing:
        if existing.hash == file_hash:
            print(f"[skip] {path.name}: unchanged since last ingestion")
            return

        old_chunks = session.exec(
            select(Chunk).where(Chunk.file_id == existing.id)
        ).all()
        for c in old_chunks:
            session.delete(c)
        existing.hash = file_hash
        existing.mime = _guess_mime(path)
        existing.page_count = _page_count_from_chunks(file_chunks)
        existing.ingested_at = datetime.now(timezone.utc)
        session.add(existing)
        session.commit()
        session.refresh(existing)
        source_file = existing
    else:
        source_file = SourceFile(
            path=str(path),
            hash=file_hash,
            mime=_guess_mime(path),
            page_count=_page_count_from_chunks(file_chunks),
        )
        session.add(source_file)
        session.commit()
        session.refresh(source_file)  # populates source_file.id

    for c in file_chunks:
        session.add(
            Chunk(
                file_id=source_file.id,
                chunk_index=c["chunk_index"],
                text=c["text"],
                page=c.get("page_number"),
                bbox=_extract_bbox(c),
            )
        )
    session.commit()
    print(f"[ok] {path.name}: saved {len(file_chunks)} chunks (file_id={source_file.id})")


def persist_folder(folder_path: str, recursive: bool = True) -> None:
    root = Path(folder_path)
    all_chunks = ingest_folder(folder_path, recursive=recursive)

    chunks_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in all_chunks:
        chunks_by_file.setdefault(chunk["source"], []).append(chunk)

    with Session(engine) as session:
        for source_path, file_chunks in chunks_by_file.items():
            persist_file_and_chunks(Path(source_path), file_chunks, session)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python db.py <folder_path>")
        sys.exit(1)

    init_db()
    persist_folder(sys.argv[1])