import os

import dotenv
dotenv.load_dotenv()

from langchain_openai import ChatOpenAI
from news_client.tavily_api import TavilyNewsClient

# Default profile when none is passed to the chain.
USER_PROFILE = os.environ.get("USER_PROFILE")

# Same models as the agent, so the comparison isolates the architecture (chain vs. graph).
SEARCH_LLM = ChatOpenAI(model="gpt-5-nano")
WRITER_LLM = ChatOpenAI(model="gpt-5-nano")

N_ARTICLES_FOR_WRITER = 10
ARTICLE_CHARS_FOR_WRITER = 1000

# Shared client instances
TAVILY_CLIENT = TavilyNewsClient()
