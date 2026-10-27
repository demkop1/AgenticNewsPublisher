"""Primary search graph: articles come from Tavily's news search.

Tavily takes a natural-language query (max 400 chars) rather than a Boolean one, so the
search worker and critic use Tavily-specific prompts. Everything downstream of the critic
(events, dedup, storage) mirrors the self-contained CurrentsAPI backup in search_graph_currents.py.
"""
import datetime
import uuid
from functools import partial
from typing import Literal

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

import agent.config as config
from agent.config import (
    RELEVANCE_THRESHOLD,
    EVENT_ID_NAMESPACE,
    DUPLICATE_SIMILARITY_THRESHOLD,
)
from agent.prompts import (
    STRING_EXTRACTOR,
    CRITIC_PROMPT_TEMPLATE,
    build_search_system_prompt,
    build_critic_system_prompt,
    EVENTS_EXTRACTION_SYSTEM_PROMPT,
    EVENTS_EXTRACTION_PROMPT_TEMPLATE,
)
from agent.state import NewsState
from agent.structured_outputs import QueryCritique, Event, EventsExtraction
from storage.articles_vectorstore import get_news_store, get_published_articles_store
from storage.events_store import get_events_store
from news_client.tavily_api import TavilyNewsClient

client = TavilyNewsClient()

def search_worker(state: NewsState) -> NewsState:
    chat_history = state.get("agent_chats", {}).get("search_worker", [])
    messages = [
        build_search_system_prompt(state.get("user_profile") or config.USER_PROFILE),
        *chat_history,
    ]

    response = config.SEARCH_LLM.invoke(messages)
    try:
        search_query = " ".join(re.findall(STRING_EXTRACTOR, response.content)[-1].split())
    except IndexError:
        search_query = response.content.strip().strip("*").strip()

    return {"search_query": search_query, "agent_chats": {"search_worker": [AIMessage(search_query)]}}


def fetch_articles_worker(state: NewsState) -> NewsState:
    def _article_id(article_dict: dict) -> str:
        key = article_dict.get("url") or article_dict.get("title")
        return str(uuid.uuid5(config.ARTICLE_ID_NAMESPACE, key))

    def _article_to_document(**metadata):
        doc_content = metadata["title"] + "\n" + metadata["content"]
        metadata.pop("content")
        return Document(doc_content, metadata=metadata)

    articles = [
        _article_to_document(article_id=_article_id(a), **a)
        for a in client.search(state["search_query"])
    ]

    return {"current_articles": articles, "articles_fetched": True}

def check_relevance(
    state: NewsState, threshold: float = RELEVANCE_THRESHOLD, max_chars: int = 500
) -> Command[Literal["criticize", "create_events"]]:
    """Keep only the fetched articles whose reranker relevance score against the search
    query is greater than `threshold`, then route:
    - some articles left: go straight to event extraction;
    - none left and the critic's retries are used up: end the graph;
    - none left otherwise: go to the critic to propose a new query.

    Bind a different threshold with functools.partial(check_relevance, threshold=...).
    """
    articles = state.get("current_articles", [])
    pairs = [(state["search_query"], a.page_content[:max_chars]) for a in articles]
    scores = [float(x) for x in config.RERANKER_MODEL.predict(pairs)] if pairs else []

    relevant = []
    for article, score in zip(articles, scores):
        article.metadata["relevance_score"] = score
        if score > threshold:
            relevant.append(article)

    if relevant:
        return Command(goto="create_events", update={"current_articles": relevant, "article_scores": scores})

    if state.get("critique_attempts", 0) >= config.MAX_CRITIQUE_ATTEMPTS:
        return Command(goto=END, update={"current_articles": [], "article_scores": scores})

    # Keep the rejected articles in state so the critic can see what the query returned.
    return Command(goto="criticize", update={"article_scores": scores})


def criticize(state: NewsState, n_shown: int = 3) -> Command[Literal["search_worker"]]:
    """Called only when no fetched article was relevant: explain why and suggest a new query."""
    structured_llm = config.SEARCH_LLM.with_structured_output(QueryCritique)

    articles = state.get("current_articles", [])
    closest = sorted(articles, key=lambda a: a.metadata.get("relevance_score", 0.0), reverse=True)[:n_shown]

    chat_history = state.get("agent_chats", {}).get("criticize", [])
    formatted_prompt = CRITIC_PROMPT_TEMPLATE.format(
        search_query=state.get("search_query"),
        num_fetched=len(articles),
        threshold=RELEVANCE_THRESHOLD,
        num_shown=len(closest),
        articles=_format_scored_articles(closest),
    )

    messages = [
        build_critic_system_prompt(state.get("user_profile") or config.USER_PROFILE),
        *chat_history,
        formatted_prompt,
    ]

    critique: QueryCritique = structured_llm.invoke(messages)

    return Command(
        goto="search_worker",
        update={
            "critique_attempts": state.get("critique_attempts", 0) + 1,
            "agent_chats": {
                "search_worker": [
                    HumanMessage(
                        f"Critic feedback: {critique.feedback}\n"
                        f"Suggested query: {critique.revised_query}"
                    )
                ],
                "criticize": [
                    formatted_prompt,
                    AIMessage(f"Feedback: {critique.feedback}, Suggested Query: {critique.revised_query}"),
                ],
            },
        },
    )

def load_published_articles(state: NewsState) -> NewsState:
    """Pull previously-published articles that are semantically close to the current
    search query, so criticize (this graph) and event_picker (publisher_graph) can
    tell what's already been covered instead of starting every run with no memory.
    """
    store = get_published_articles_store()
    published = store.similarity_search( state["search_query"], k=config.N_FETCHED_PUBLISHED_ARTICLES )

    return {"published_articles": published}

def _format_scored_articles(articles: list[Document]) -> str:
    if not articles:
        return "(none)"
    return "\n\n".join(
        f"- [score {a.metadata.get('relevance_score', 0.0):.2f}] {a.page_content[:500]}\n  url: {a.metadata.get('url')}"
        for a in articles
    )

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
    structured_llm = config.SEARCH_LLM.with_structured_output(EventsExtraction)

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

def _drop_stored_duplicates(events: list[Document]) -> list[Document]:
    """Drop the events whose description is highly similar (TF-IDF cosine similarity above
    DUPLICATE_SIMILARITY_THRESHOLD) to an event already stored in the database."""
    stored_texts = get_events_store().all_descriptions()
    if not events or not stored_texts:
        return events

    new_texts = [event.page_content for event in events]
    vectorizer = TfidfVectorizer().fit(stored_texts + new_texts)
    similarity_scores = (vectorizer.transform(new_texts) @ vectorizer.transform(stored_texts).T).toarray()  # (N_new, N_stored)

    return [event for event, row in zip(events, similarity_scores) if row.max() <= DUPLICATE_TF_IDF_SIMILARITY_THRESHOLD]

def deduplicate(state: NewsState) -> NewsState:
    events: list[Document] = state["events"]
    if len(events) < 2:
        return {"events": _drop_stored_duplicates(events)}

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

    # Finally, compare the survivors with the events already in the database.
    return {"events": _drop_stored_duplicates([events[i] for i in survivor_indices])}

def store_articles_worker(state: NewsState) -> NewsState:
    articles_store = get_news_store(embeddings=config.EMBEDDINGS_MODEL)
    events_store = get_events_store()

    stored_article_ids = articles_store.upsert_articles( state["current_articles"], splitter=config.ARTICLE_SPLITTER )
    stored_event_ids = events_store.upsert_events( state["events"] )

    return {
        "stored_article_ids": stored_article_ids,
        "stored_event_ids": stored_event_ids,
         "articles_stored": True
    }


builder = StateGraph(NewsState)

builder.add_node("search_worker", search_worker)
builder.add_node("load_published_articles", load_published_articles)
builder.add_node("fetch_articles_worker", fetch_articles_worker)
builder.add_node(
    "check_relevance",
    partial(check_relevance, threshold=RELEVANCE_THRESHOLD),
    destinations=("criticize", "create_events"),
)
builder.add_node("criticize", criticize, destinations=("search_worker",))
builder.add_node("create_events", create_events)
builder.add_node("deduplicate", deduplicate)
builder.add_node("store_articles_worker", store_articles_worker)

builder.add_edge(START, "search_worker")
builder.add_edge("search_worker", "load_published_articles")
builder.add_edge("load_published_articles", "fetch_articles_worker")
builder.add_edge("fetch_articles_worker", "check_relevance")

builder.add_edge("create_events", "deduplicate")
builder.add_edge("deduplicate", "store_articles_worker")
builder.add_edge("store_articles_worker", END)

search_graph = builder.compile()
