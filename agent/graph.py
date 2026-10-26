import os
from typing import Literal
from agent.config import AGENT_NAMES

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from news_client.newsapi_client import NewsAPIClient
from news_client.current_api import CurrentsAPIClient

import re
from agent.prompts import *
from langchain_core.messages import SystemMessage, HumanMessage

from agent.subgraph.search_graph import search_graph
from agent.subgraph.publisher_graph import publisher_graph

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from agent.state import NewsState


def search2publisher_conditioner(state: NewsState) -> Command:
    # No events (no relevant articles, or none extracted): nothing to write about, so don't
    # call the writer on an empty list.
    if state.get("events"):
        return Command(goto="publisher_graph")
    return Command(goto=END)

builder = StateGraph(NewsState)

builder.add_node("search_graph", search_graph)
builder.add_node("search2publisher_conditioner", search2publisher_conditioner, destinations=(END, "publisher_graph"))
builder.add_node("publisher_graph", publisher_graph)

builder.add_edge(START, "search_graph")
builder.add_edge("search_graph", "search2publisher_conditioner")
builder.add_edge("publisher_graph", END)

graph = builder.compile()
