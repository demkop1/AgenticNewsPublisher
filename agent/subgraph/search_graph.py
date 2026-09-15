import os
import datetime
import uuid
from typing import Literal
from agent.config import AGENT_NAMES, MAX_CRITIQUE_ATTEMPTS, EVENT_ID_NAMESPACE, DUPLICATE_SIMILARITY_THRESHOLD
import agent.config as config

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from news_client.newsapi_client import NewsAPIClient
from news_client.current_api import CurrentsAPIClient

import re
from agent.prompts import *
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agent.structured_outputs import SchedulerDecision, QueryCritique, Event, EventsExtraction

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from agent.state import NewsState

from langchain_text_splitters import RecursiveCharacterTextSplitter

from storage.articles_vectorstore import get_news_store, get_published_articles_store
from storage.events_store import get_events_store

llm = ChatOpenAI(model="gpt-5-mini")
embeddings_model = OpenAIEmbeddings()

currentsapi_client = CurrentsAPIClient()

def search_worker(state: NewsState) -> NewsState:
    chat_history = state.get("agent_chats", {}).get("search_worker", [])
    messages = [SEARCH_SYSTEM_PROMPT, *[m for m in chat_history]]

    response = llm.invoke(messages)
    try:
        search_query = re.findall(STRING_EXTRACTOR, response.content)[-1]
    except IndexError:
        search_query = response

    return {"search_query": search_query, "agent_chats": {"search_worker": [AIMessage(search_query)]}}

def load_published_articles(state: NewsState) -> NewsState:
    """Pull previously-published articles that are semantically close to the current
    search query, so criticize (this graph) and event_picker (publisher_graph) can
    tell what's already been covered instead of starting every run with no memory.
    """
    store = get_published_articles_store()
    published = store.similarity_search( state["search_query"], k=config.N_FETCHED_PUBLISHED_ARTICLES )

    return {"published_articles": published}

def fetch_articles_worker(state: NewsState) -> NewsState:
    def _article_id(article_dict: dict) -> str:
        key = article_dict.get("url") or article_dict.get("title")
        return str(uuid.uuid5(config.ARTICLE_ID_NAMESPACE, key))
    
    def _article_to_document(**metadata):
        doc_content = metadata['title'] + "\n" + metadata['content']
        metadata.pop('content')

        return Document(doc_content, metadata=metadata)

    search_query = state["search_query"]
    
    articles = [
        _article_to_document(article_id=_article_id(a), **a) for a in currentsapi_client.search(search_query)
    ]

    return {
            "current_articles": articles,
            "articles_fetched": True,
        }

def _format_articles(articles: list[Document]) -> str:
    if not articles:
        return "(none)"
    # return "\n\n".join(
    #     f"- {a.page_content}\n  url: {a.metadata.get('url')}" for a in articles
    # )
    return "\n\n".join(
        f"- {a.page_content[:500]}\n  url: {a.metadata.get('url')}" for a in articles
    )

# Covers our own store's isoformat() timestamps, plain ISO 8601, Currents API's
# space-separated format, and a bare date, in that order.
_PUBLISHED_AT_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S %z",
    "%Y-%m-%d",
)
def _parse_published_at(raw: str | None) -> datetime.datetime:
    text = (raw or "").strip().replace("Z", "+0000")
    for fmt in _PUBLISHED_AT_FORMATS:
        try:
            parsed = datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=datetime.timezone.utc)
    return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

def _most_recent(articles: list[Document], limit: int) -> list[Document]:
    def _sort_key(a: Document) -> datetime.datetime:
        raw = a.metadata.get("publishedAt") or a.metadata.get("published_at")
        return _parse_published_at(raw)

    return sorted(articles, key=_sort_key, reverse=True)[:limit]

def criticize(state: NewsState) -> Command[Literal["search_worker", "create_events"]]:
    DECISION2DESTINATION = {
        "approve": "create_events",
        "revise": "search_worker"
    }

    structured_llm = llm.with_structured_output(QueryCritique)

    articles = state.get("current_articles", [])
    recent_articles = _most_recent(articles, limit=10)
    published_articles = _most_recent(state.get("published_articles", []), limit=5)

    chat_history = state.get("agent_chats", {}).get("criticize", [])
    formatted_prompt = CRITIC_PROMPT_TEMPLATE.format(
        search_query=state.get("search_query"),
        num_fetched=len(articles),
        articles=_format_articles(recent_articles),
        published_articles=_format_articles(published_articles),
    )

    messages = [
        CRITIC_SYSTEM_PROMPT,
        *[m for m in chat_history],
        formatted_prompt,
    ]

    critique: QueryCritique = structured_llm.invoke(messages)
    attempts = state.get("critique_attempts", 0) + 1

    should_revise = (
        (critique.verdict == "revise" or critique.is_redundant)
        and critique.revised_query
    )

    if critique.is_redundant and attempts >= MAX_CRITIQUE_ATTEMPTS:
        return Command(goto=END, update={"critique_attempts": attempts})

    if not should_revise or attempts >= MAX_CRITIQUE_ATTEMPTS:
        return Command(goto=DECISION2DESTINATION["approve"], update={"critique_attempts": attempts})

    return Command(
        goto=DECISION2DESTINATION["revise"],
        update={
            "critique_attempts": attempts,
            "agent_chats": {
                "search_worker": [
                    HumanMessage(
                        f"Critic feedback: {critique.feedback}\n"
                        f"Suggested query: {critique.revised_query}"
                    )
                ],
                "criticize": [
                    formatted_prompt,
                    AIMessage(f"Verdict: {critique.verdict}, Feedback: {critique.feedback}, Suggested Query: {critique.revised_query}")
                ]
            },
        },
    )


def _event_to_document(event: Event, article: Document) -> Document:
    source_url = article.metadata.get("url")
    
    event_id = str(uuid.uuid5(EVENT_ID_NAMESPACE, f"{source_url}|{event.event_description}"))

    return Document(
        event.event_description,
        metadata={
            **event.model_dump(mode="json"),
            "event_id": event_id,
            "article_id": article.metadata.get("id"),
            "source_url": source_url,
            "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
    )

def create_events(state: NewsState) -> NewsState:
    structured_llm = llm.with_structured_output(EventsExtraction)

    event_documents: list[Document] = []
    for article in state.get("current_articles", []):
        messages = [
            EVENTS_EXTRACTION_SYSTEM_PROMPT,
            EVENTS_EXTRACTION_PROMPT_TEMPLATE.format(
                published_at=article.metadata.get("publishedAt", "unknown"),
                article=article.page_content,
            ),
        ]
        extraction: EventsExtraction = structured_llm.invoke(messages)
        event_documents.extend(_event_to_document(event, article) for event in extraction.events)

    return {"events": event_documents}

def deduplicate(state: NewsState) -> NewsState:
    events: list[Document] = state["events"]
    if len(events) < 2:
        return {}

    event_texts = [event.page_content for event in events]

    vectorizer = TfidfVectorizer().fit(event_texts)
    embedded_events = vectorizer.transform( event_texts ) #Perform TF-IDF embedding

    similarity_scores = (embedded_events @ embedded_events.T).toarray() # shape: (N_events, N_events)
    np.fill_diagonal(similarity_scores, 0.0)  # an event is never a "duplicate" of itself
    duplicates_mask = similarity_scores > DUPLICATE_SIMILARITY_THRESHOLD

    # Union-Find: group events pairwise flagged as duplicates into clusters,
    # so that A~B and B~C also merges A and C into the same cluster.
    n = len(events)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(n):
        for j in range(i + 1, n):
            if duplicates_mask[i, j]:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    def confidence(index: int) -> float:
        return events[index].metadata.get("confidence", 0.0)

    # Within each duplicate cluster, the event with the highest confidence prevails.
    survivor_indices = sorted(max(indices, key=confidence) for indices in clusters.values())

    return {"events": [events[i] for i in survivor_indices]}

def store_articles_worker(state: NewsState) -> NewsState:
    articles_store = get_news_store(embeddings=embeddings_model)
    events_store = get_events_store()

    stored_article_ids = articles_store.upsert_articles( state["current_articles"], splitter=config.ARTICLE_SPLITTER )
    stored_event_ids = events_store.upsert_events( state["events"] )

    return {
        "stored_article_ids": stored_article_ids,
        "stored_event_ids": stored_event_ids,
         "articles_stored": True
    }


builder = StateGraph(NewsState)

# builder.add_node(
#     "scheduler",
#     scheduler,
#     destinations=("search_worker", END),
# )

builder.add_node("search_worker", search_worker)
builder.add_node("load_published_articles", load_published_articles)
builder.add_node("fetch_articles_worker", fetch_articles_worker)
builder.add_node("create_events", create_events)
builder.add_node("deduplicate", deduplicate)
builder.add_node("store_articles_worker", store_articles_worker)
builder.add_node("criticize", criticize, destinations=("search_worker", "create_events"))

builder.add_edge(START, "search_worker")
builder.add_edge("search_worker", "load_published_articles")
builder.add_edge("load_published_articles", "fetch_articles_worker")
builder.add_edge("fetch_articles_worker", "criticize")

builder.add_edge("create_events", "deduplicate")
builder.add_edge("deduplicate", "store_articles_worker")
builder.add_edge("store_articles_worker", END)

search_graph = builder.compile()
