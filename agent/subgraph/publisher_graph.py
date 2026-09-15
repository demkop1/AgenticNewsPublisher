import os
import datetime
import uuid
from typing import Literal
from agent.config import *

import numpy as np

import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from news_client.newsapi_client import NewsAPIClient
from news_client.current_api import CurrentsAPIClient

import re
from agent.prompts import *
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from agent.structured_outputs import EventSelection

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from agent.state import NewsState
from storage.articles_vectorstore import get_news_store, get_published_articles_store
from storage.events_store import get_events_store, EVENTS_TABLE_NAME

llm = ChatOpenAI(model="gpt-5-mini")
embeddings_model = OpenAIEmbeddings()

currentsapi_client = CurrentsAPIClient()

def _load_events_from_db(limit: int = 100) -> list[Document]:
    store = get_events_store()
    with store._conn.cursor() as cur:
        cur.execute(
            f"SELECT id, content, metadata FROM {EVENTS_TABLE_NAME} ORDER BY metadata ->> 'extracted_at' LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()
    return [Document(content, metadata=metadata, id=row_id) for row_id, content, metadata in rows]

def _format_events(events: list[Document]) -> str:
    if not events:
        return "(none)"
    return "\n\n".join(
        f"- [{e.metadata.get('event_id', e.id)}] {e.page_content} "
        f"(type={e.metadata.get('event_type')}, date={e.metadata.get('event_date')}, "
        f"confidence={e.metadata.get('confidence')})"
        for e in events
    )
def _format_articles(articles: list[Document]) -> str:
    if not articles:
        return "(none)"
    return "\n\n".join(
        f"- {a.page_content}\n  url: {a.metadata.get('url')}" for a in articles
    )


def event_picker(state: NewsState) -> NewsState:
    events = _load_events_from_db()

    if not events:
        return {"events": []}

    structured_llm = llm.with_structured_output(EventSelection)
    messages = [
        EVENT_PICKER_SYSTEM_PROMPT,
        EVENT_PICKER_PROMPT_TEMPLATE.format(
            events=_format_events(events),
            published_articles=_format_articles(state.get("published_articles", [])),
        ),
    ]
    selection: EventSelection = structured_llm.invoke(messages)

    events_by_id = {e.metadata.get("event_id", e.id): e for e in events}
    picked = [events_by_id[event_id] for event_id in selection.event_ids if event_id in events_by_id]

    return {"events": picked}

def fetch_rag(state: NewsState) -> NewsState:
    response = llm.invoke([RAG_SYSTEM_PROMPT_TEMPLATE.format(
        events=_format_articles(state["events"])
    )])
    try:
        search_query = re.findall(STRING_EXTRACTOR, response.content)[-1]
    except IndexError:
        search_query = response

    articles_store = get_news_store(embeddings=embeddings_model)
    documents_retrieved = articles_store.similarity_search(search_query)
    
    return {
        "current_articles": documents_retrieved
    }

# def group_similar_events(state: NewsState) -> NewsState:
#     events: list[Document] = state["events"]
#     if len(events) < 2:
#         return {"event_groups": [events] if events else []}

#     event_texts = [event.page_content for event in events]

#     vectorizer = TfidfVectorizer().fit(event_texts)
#     embedded_events = vectorizer.transform(event_texts)  # Perform TF-IDF embedding

#     similarity_scores = (embedded_events @ embedded_events.T).toarray()  # shape: (N_events, N_events)
#     np.fill_diagonal(similarity_scores, 0.0)  # an event is never "similar" to itself
#     similar_mask = similarity_scores > DUPLICATE_SIMILARITY_THRESHOLD

#     # Union-Find: group events pairwise flagged as similar into clusters,
#     # so that A~B and B~C also merges A and C into the same cluster.
#     n = len(events)
#     parent = list(range(n))

#     def find(i: int) -> int:
#         while parent[i] != i:
#             parent[i] = parent[parent[i]]
#             i = parent[i]
#         return i

#     def union(i: int, j: int) -> None:
#         root_i, root_j = find(i), find(j)
#         if root_i != root_j:
#             parent[root_i] = root_j

#     for i in range(n):
#         for j in range(i + 1, n):
#             if similar_mask[i, j]:
#                 union(i, j)

#     clusters: dict[int, list[int]] = {}
#     for i in range(n):
#         clusters.setdefault(find(i), []).append(i)

#     event_groups = [[events[i] for i in indices] for indices in clusters.values()]

#     return {"event_groups": event_groups}

def generate_article(state: NewsState) -> NewsState:
    messages = [
        ARTICLE_GENERATOR_SYSTEM_PROMPT,
        ARTICLE_GENERATOR_PROMPT_TEMPLATE.format(
            events=_format_events(state.get("events", [])),
            retrieved_articles=_format_articles(state.get("current_articles", [])),
            published_articles=_format_articles(state.get("published_articles", [])),
        ),
    ]
    response = llm.invoke(messages)

    published_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    article_id = str(uuid.uuid5(PUBLISHED_ARTICLE_ID_NAMESPACE, f"{published_at}|{response.content}"))
    generated_article = Document(
        response.content,
        metadata={
            "article_id": article_id,
            "published_at": published_at,
            "event_ids": [e.metadata.get("event_id", e.id) for e in state.get("events", [])],
        },
    )

    # Persist so future runs' criticize/event_picker can see what's already published.
    store = get_published_articles_store()
    store.upsert_articles([generated_article])

    return {"generated_article": response.content}

builder = StateGraph(NewsState)

builder.add_node("event_picker", event_picker)
builder.add_node("fetch_rag", fetch_rag)
# builder.add_node("group_similar_events", group_similar_events)
builder.add_node("generate_article", generate_article)

builder.add_edge(START, "event_picker")
builder.add_edge("event_picker", "fetch_rag")
builder.add_edge("fetch_rag", "generate_article")
# builder.add_edge("group_similar_events", "generate_article")
builder.add_edge("generate_article", END)

publisher_graph = builder.compile()
