from typing import TypedDict

from langchain_core.documents import Document


class ChainState(TypedDict, total=False):
    user_profile: str
    search_query: str
    articles: list[Document]
    generated_article: str
