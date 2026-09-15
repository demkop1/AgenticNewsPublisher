from typing import Literal
import warnings
import os

from langgraph.graph import END
from langchain_text_splitters import RecursiveCharacterTextSplitter

import dotenv
dotenv.load_dotenv()

import uuid

# Fixed namespace so the same article URL always maps to the same row id,
# across processes and restarts.
ARTICLE_ID_NAMESPACE = uuid.UUID("f0b3d9d0-6e1a-4b8b-9b0a-2f5c7a1e9d3c")
EVENT_ID_NAMESPACE = uuid.UUID("6b3f8a2e-6e0a-4a3e-9f0a-9e0d3b7a1c2f")
PUBLISHED_ARTICLE_ID_NAMESPACE = uuid.UUID("c0cab9e6-0836-4cb7-a5f5-52358e2d7934")

if not os.environ['USER_PROFILE']:
    if os.environ['USER_PROFILE_FILEPATH']:
        with open(os.environ['USER_PROFILE_FILEPATH'], 'r') as f:
            USER_PROFILE = f.read()
    else:
        USER_PROFILE = None
        warnings.warn("Specify either USER_PROFILE or USER_PROFILE_FILEPATH in your environment!")
else:
    USER_PROFILE = os.environ['USER_PROFILE']

AGENT_DESCRIPTIONS: dict[str, str] = {
    "search_worker": "Generates the next API search query from the user's profile and preferences.",
    # "fetch_articles_worker": "Fetches top headlines and matching articles from NewsAPI for the current search query.",
    # "store_articles_worker": "Persists fetched articles as embeddings in Postgres, deduplicated by URL.",
    END: "When everything seems okay and an article is published, the end of execution is called"
}

AGENT_NAMES = list(AGENT_DESCRIPTIONS.keys())

MAX_CRITIQUE_ATTEMPTS = 3
N_FETCHED_PUBLISHED_ARTICLES = 5

DUPLICATE_SIMILARITY_THRESHOLD = 0.3

ARTICLE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
)