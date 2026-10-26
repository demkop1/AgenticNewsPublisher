import dotenv
from langsmith import Client, evaluate

dotenv.load_dotenv()

from agent.config import RERANKER_MODEL
from agent_baseline.chain import chain
from evaluation.utils import serialize_state
from evaluation.criteria import has_articles, relevance, crossencoder_score, article_quality

DATASET_NAME = "news-user-preferences"
MAX_CONCURRENCY = 5
N_ARTICLES_TO_JUDGE = 5
ARTICLE_CHARS_FOR_JUDGE = 500


def target(inputs: dict) -> dict:
    state = chain.invoke({"user_profile": inputs["user_profile"]})

    articles = state.get("articles", [])
    # The baseline has no relevance filter, so score the articles here with the same model
    # and input format as the agent's check_relevance (RERANKER_MODEL), to make
    # crossencoder_score comparable between the two.
    pairs = [(state["search_query"], a.page_content[:ARTICLE_CHARS_FOR_JUDGE]) for a in articles]
    scores = RERANKER_MODEL.predict(pairs) if pairs else []

    return {
        "search_query": state.get("search_query"),
        "num_articles": len(articles),
        "articles": [a.page_content[:ARTICLE_CHARS_FOR_JUDGE] for a in articles[:N_ARTICLES_TO_JUDGE]],
        "article_scores": [float(s) for s in scores],
        "articles_url": [a.metadata.get("url") for a in articles],
        "generated_article": state.get("generated_article"),
        "state": serialize_state(state),
    }


if __name__ == "__main__":
    client = Client()
    evaluate(
        target,
        data=client.list_examples(dataset_name=DATASET_NAME, limit=10),  # drop limit for the full run
        # No events_confidence: the baseline chain does not extract events.
        evaluators=[has_articles, relevance, crossencoder_score, article_quality],
        experiment_prefix="baseline-chain-tavily",
        max_concurrency=MAX_CONCURRENCY,
        client=client,
    )
