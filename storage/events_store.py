import os
from typing import Iterable, Optional

import dotenv

dotenv.load_dotenv()

import psycopg
from psycopg.types.json import Json

from langchain_core.documents import Document

EVENTS_TABLE_NAME = "events"


def _connection_string() -> str:
    connection = os.environ.get("DATABASE_URL")
    if not connection:
        raise ValueError(
            "Set DATABASE_URL to a Postgres connection string, e.g. "
            "postgresql+psycopg://user:password@localhost:5432/news"
        )
    # DATABASE_URL is a SQLAlchemy-style DSN (for the PGVector-based article
    # store); psycopg.connect() wants a plain postgresql:// DSN.
    return connection.replace("postgresql+psycopg://", "postgresql://", 1)


def _event_id(document: Document) -> str:
    return str(document.metadata.get("event_id"))


class EventsPostgresStore:

    def __init__(self, connection: Optional[str] = None, table_name: str = EVENTS_TABLE_NAME):
        self.table_name = table_name
        self._conn = psycopg.connect(connection or _connection_string(), autocommit=True)
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
                """
            )

    def all_descriptions(self) -> list[str]:
        """The description (content) of every stored event."""
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT content FROM {self.table_name}")
            return [content for (content,) in cur.fetchall()]

    def upsert_events(self, events: Iterable[Document]) -> list[str]:
        events = list(events)
        if not events:
            return []

        rows, ids, seen = [], [], set()
        for event in events:
            event_id = _event_id(event)
            ids.append(event_id)
            if event_id in seen:
                continue
            seen.add(event_id)
            rows.append((event_id, event.page_content, Json(event.metadata)))

        with self._conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self.table_name} (id, content, metadata)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata
                """,
                rows,
            )

        return ids


def get_events_store() -> EventsPostgresStore:
    return EventsPostgresStore()
