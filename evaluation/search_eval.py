import dotenv
from langgraph.graph import StateGraph, START, END
from langsmith import Client, evaluate

dotenv.load_dotenv()

from functools import partial

from agent.config import RELEVANCE_THRESHOLD
from agent.state import NewsState
from agent.subgraph.search_graph import (
    search_worker,
    load_published_articles,
    fetch_articles_worker,
    check_relevance,
    criticize,
)
from evaluation.criteria import has_articles, relevance

DATASET_NAME = "news-user-preferences"
MAX_CONCURRENCY = 10
N_ARTICLES_TO_JUDGE = 5
ARTICLE_CHARS_FOR_JUDGE = 500

# Same wiring as agent.subgraph.search_graph, but stops once check_relevance finds relevant
# articles (or the critic's retries run out): create_events/deduplicate/store_articles_worker are skipped so concurrent
# runs don't write to the shared vector stores.
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
builder.add_node("create_events", lambda state: {})
builder.add_edge(START, "search_worker")
builder.add_edge("search_worker", "load_published_articles")
builder.add_edge("load_published_articles", "fetch_articles_worker")
builder.add_edge("fetch_articles_worker", "check_relevance")
builder.add_edge("create_events", END)
eval_graph = builder.compile()

def target(inputs: dict) -> dict:
    state = eval_graph.invoke({"user_profile": inputs["user_profile"]})

    return {
        "search_query": state.get("search_query"),
        "num_articles": len(state.get("current_articles", [])),
        "articles": [
            a.page_content[:ARTICLE_CHARS_FOR_JUDGE]
            for a in state.get("current_articles", [])[:N_ARTICLES_TO_JUDGE]
        ],
        "critique_attempts": state.get("critique_attempts", 0),
    }

if __name__ == "__main__":
    client = Client()
    evaluate(
        target,
        data=client.list_examples(dataset_name=DATASET_NAME, limit=10),  # drop limit for the full run
        evaluators=[has_articles, relevance],
        experiment_prefix="search-agent-tavily",
        max_concurrency=MAX_CONCURRENCY,
        client=client,
    )
