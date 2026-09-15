"""
Minimal demo of talking to a running LangGraph API server (the `api` service
in docker-compose.yml) via langgraph_sdk instead of importing the graph
in-process.

Usage:
    docker compose up -d          # make sure redis / news_postgres / api are up
    python demo/run_graph.py

Env vars (optional, see .env):
    LANGGRAPH_API_URL   base URL of the LangGraph API server (default: http://localhost:8000)
    LANGGRAPH_API_KEY   API key to send as `x-api-key`, if the server requires auth
"""
import os
from pathlib import Path
import sys
sys.path.append( os.getcwd() )

from agent.config import USER_PROFILE

import dotenv
from langgraph_sdk import get_client

dotenv.load_dotenv()

# LANGGRAPH_API_URL = os.environ.get("LANGGRAPH_API_URL", "http://localhost:8000")
LANGGRAPH_API_URL = r"http://localhost:8000"
LANGGRAPH_API_KEY = os.environ.get("LANGGRAPH_API_KEY")

# The graph registered under LANGSERVE_GRAPHS in the Dockerfile / langgraph.json.
GRAPH_ID = "graph"


def main() -> None:
    client = get_client(url=LANGGRAPH_API_URL, api_key=LANGGRAPH_API_KEY)

    thread = client.threads.create()
    print(f"Created thread {thread['thread_id']}")

    print("Streaming run updates:")
    for chunk in client.runs.stream(
        thread["thread_id"],
        GRAPH_ID,
        input={
            "user_profile": USER_PROFILE
        },
        stream_mode="values",
    ):
        print(f"[{chunk.event}] {chunk.data}")

    final_state = client.threads.get_state(thread["thread_id"])
    generated_article = final_state["values"].get("generated_article")

    print("\nFinal generated_article:")
    print(generated_article or "(none produced)")


if __name__ == "__main__":
    main()
