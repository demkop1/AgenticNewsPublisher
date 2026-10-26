import dotenv
from langsmith import Client, evaluate

dotenv.load_dotenv()

from agent.subgraph.search_graph import search_graph
from agent.subgraph.publisher_graph import publisher_graph
from evaluation.utils import serialize_state
from evaluation.criteria import has_articles, relevance, events_confidence, crossencoder_score, article_quality

DATASET_NAME = "news-user-preferences"
MAX_CONCURRENCY = 5
N_ARTICLES_TO_JUDGE = 5
ARTICLE_CHARS_FOR_JUDGE = 500


def target(inputs: dict) -> dict:
    # Same two stages as agent.graph (search_graph -> publisher_graph), invoked one after
    # the other so the search stage's articles and events can be scored: the publisher
    # stage overwrites both `current_articles` (RAG results) and `events` (picked events).
    search_state = search_graph.invoke({"user_profile": inputs["user_profile"]})
    # Same condition as agent.graph: the writer only runs when the search produced events.
    publisher_state = publisher_graph.invoke(search_state) if search_state.get("events") else {}

    articles = search_state.get("current_articles", [])
    events = search_state.get("events", [])
    return {
        "search_query": search_state.get("search_query"),
        "num_articles": len(articles),
        "articles": [a.page_content[:ARTICLE_CHARS_FOR_JUDGE] for a in articles[:N_ARTICLES_TO_JUDGE]],
        "article_scores": search_state.get("article_scores", []),
        "articles_url": [a.metadata.get("url") for a in articles],
        "event_confidences": [e.metadata.get("confidence", 0.0) for e in events],
        "generated_article": publisher_state.get("generated_article"),
        "state": {"search": serialize_state(search_state), "publisher": serialize_state(publisher_state)},
    }


if __name__ == "__main__":
    client = Client()
    evaluate(
        target,
        data=client.list_examples(dataset_name=DATASET_NAME, limit=10),  # drop limit for the full run
        evaluators=[has_articles, relevance, events_confidence, crossencoder_score, article_quality],
        experiment_prefix="graph-agent-tavily",
        max_concurrency=MAX_CONCURRENCY,
        client=client,
    )
