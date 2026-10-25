"""Baseline: a plain LangChain chain instead of the agent's graph.

search query (LLM) -> Tavily search -> news article (LLM). There is no relevance filter,
critic, event extraction, deduplication, storage, or memory of what was already published.
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from agent_baseline import config
from agent_baseline.prompts import SEARCH_PROMPT, WRITER_PROMPT, STRING_EXTRACTOR
from agent_baseline.state import ChainState


def _extract_query(text: str) -> str:
    matches = STRING_EXTRACTOR.findall(text)
    return " ".join((matches[-1] if matches else text.strip().strip("*")).split())


def _to_documents(results: list[dict]) -> list[Document]:
    return [
        Document(r["title"] + "\n" + r["content"], metadata={k: v for k, v in r.items() if k != "content"})
        for r in results
    ]


def _format_articles(articles: list[Document]) -> str:
    if not articles:
        return "(none)"
    return "\n\n".join(
        f"- {a.page_content[:config.ARTICLE_CHARS_FOR_WRITER]}\n  url: {a.metadata.get('url')}"
        for a in articles[:config.N_ARTICLES_FOR_WRITER]
    )


def build_chain(search_llm=config.SEARCH_LLM, writer_llm=config.WRITER_LLM, search_client=config.TAVILY_CLIENT) -> Runnable:
    search_query = SEARCH_PROMPT | search_llm | StrOutputParser() | _extract_query
    fetch_articles = RunnableLambda(lambda state: _to_documents(search_client.search(state["search_query"])))
    write_article = (
        RunnableLambda(lambda state: {"user_profile": state["user_profile"], "articles": _format_articles(state["articles"])})
        | WRITER_PROMPT
        | writer_llm
        | StrOutputParser()
    )

    return (
        RunnablePassthrough.assign(search_query=search_query)
        | RunnablePassthrough.assign(articles=fetch_articles)
        | RunnablePassthrough.assign(generated_article=write_article)
    )


chain = build_chain()
