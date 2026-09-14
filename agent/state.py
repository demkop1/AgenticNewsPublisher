from typing import TypedDict, Literal, Annotated
from agent.config import USER_PROFILE, AGENT_NAMES
import operator

from langchain_core.documents import Document


def merge_agent_prompts(
    existing: dict[str, list[str]], update: dict[str, list[str]]
) -> dict[str, list[str]]:
    merged = {**existing}
    for agent, new_messages in update.items():
        merged[agent] = merged.get(agent, []) + new_messages
    return merged


class NewsState(TypedDict):
    # user_profile: str = USER_PROFILE if USER_PROFILE else None
    search_query: str
    current_articles: list[Document] = []
    stored_article_ids: list[str] = []
    stored_event_ids: list[str] = []
    events: list[Document] = []
    event_groups: list[list[Document]] = []
    generated_article: str = ""

    published_articles: list[Document|str] = []

    agent_chats: Annotated[dict[str, list], merge_agent_prompts] = {}

    #Critic
    critique_attempts: int = 0

    #Orchestrator
    # next_agent: AGENT_NAMES