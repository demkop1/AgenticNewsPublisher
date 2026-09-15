import os
import uuid
from typing import Iterable, Optional

import dotenv

from agent.config import ARTICLE_ID_NAMESPACE

dotenv.load_dotenv()

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from langchain_text_splitters import TextSplitter


NEWS_COLLECTION_NAME = "news_articles"
PUBLISHED_NEWS_COLLECTION_NAME = "published_news_articles"

def _connection_string() -> str:
    connection = os.environ.get("DATABASE_URL")
    if not connection:
        raise ValueError(
            "Set DATABASE_URL to a Postgres connection string, e.g. "
            "postgresql+psycopg://user:password@localhost:5432/news"
        )
    return connection


def article_id(document: Document) -> str:
    key = document.metadata.get("url") or document.page_content
    return str(uuid.uuid5(ARTICLE_ID_NAMESPACE, key))


class NewsPostgresStore:
    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        connection: Optional[str] = None,
        collection_name: str = NEWS_COLLECTION_NAME,
    ):
        if embeddings is None:
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings()

        self.store = PGVector(
            embeddings=embeddings,
            connection=connection or _connection_string(),
            collection_name=collection_name,
            use_jsonb=True,
        )

    def upsert_articles(
        self, documents: Iterable[Document], splitter: Optional[TextSplitter] = None
    ) -> list[str]:
        documents = list(documents)
        if not documents:
            return []

        if splitter is not None:
            chunks, ids = [], []
            for document in documents:
                article_id_value = document.metadata.get("article_id")
                for index, chunk in enumerate(splitter.split_documents([document])):
                    chunks.append(chunk)
                    ids.append(str(uuid.uuid5(ARTICLE_ID_NAMESPACE, f"{article_id_value}|{index}")))
        else:
            chunks = documents
            ids = [d.metadata.get("article_id") for d in documents]

        existing_ids = {d.id for d in self.store.get_by_ids(ids)}

        new_docs, new_ids, seen = [], [], set()
        for doc, doc_id in zip(chunks, ids):
            if doc_id in existing_ids or doc_id in seen:
                continue
            seen.add(doc_id)
            new_docs.append(doc)
            new_ids.append(doc_id)

        if new_docs:
            self.store.add_documents(new_docs, ids=new_ids)

        return ids

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        return self.store.similarity_search(query, k=k)

def get_news_store(embeddings: Optional[Embeddings] = None) -> NewsPostgresStore:
    _news_store = NewsPostgresStore(embeddings=embeddings, collection_name=NEWS_COLLECTION_NAME)
    return _news_store

def get_published_articles_store() -> NewsPostgresStore:
    _news_store = NewsPostgresStore(embeddings=None, collection_name=PUBLISHED_NEWS_COLLECTION_NAME)
    return _news_store