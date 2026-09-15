# NewsAgent

An agentic news pipeline built on [LangGraph](https://github.com/langchain-ai/langgraph) that searches for news matching a user's stated interests, extracts and deduplicates discrete events from the coverage, and picks which of those events are worth writing up — as a step toward auto-publishing curated news via API.

## How it works

The graph is composed of two LangGraph subgraphs, each a mix of LLM-driven agents and deterministic steps:

Exported directly from the compiled graph (`agent.graph.graph.get_graph(xray=True).draw_mermaid_png()`), so it always reflects the actual node names and wiring in code rather than a hand-drawn approximation:

![Graph structure](docs/mermaid-diagram.png)

Note: `criticize`'s two dotted edges above (to `search_worker` and `create_events`) are its only *declared* destinations (`add_node(..., destinations=(...))` in `search_graph.py`) — the direct-to-`END` branch it can now also take on persistent redundancy isn't part of the static graph declaration, so LangGraph's own exporter doesn't draw it either.

Everything shares one `NewsState` (`agent/state.py`) as it flows through the graph, and every LLM call is grounded in a single user profile (`user_profile.txt` or `USER_PROFILE` in `agent/config.py`) so the whole pipeline stays personalized to what that user actually wants to read.

### Node reference

**Search Graph** (`agent/subgraph/search_graph.py`) — find, extract, store:

- **Search** (`search_worker`) — LLM drafts a CurrentsAPI boolean query from the user profile (and, on a retry, the critic's feedback).
- **Load Published** (`load_published_articles`) — semantic search over the published-articles store for the most recent articles related to the current query, so redundancy can be judged downstream by both Critic and Pick Events.
- **Fetch Articles** (`fetch_articles_worker`) — calls CurrentsAPI with the query and normalizes the results into `Document`s.
- **Critic** (`criticize`) — LLM judges the fetched articles for relevance, CurrentsAPI fit, and (via an explicit `is_redundant` flag) overlap with the recently published articles; routes back to Search to revise the query, forward to Create Events to proceed, or straight to `END` if coverage is still redundant after the retry cap is used up.
- **Create Events** (`create_events`) — per-article structured-output LLM call that extracts discrete events (description, type, date, actors, confidence).
- **Deduplicate** (`deduplicate`) — TF-IDF cosine-similarity clustering of events extracted in this run; keeps only the highest-confidence event per cluster.
- **Store** (`store_articles_worker`) — chunks and embeds articles into the pgvector store, and upserts events into Postgres.

**Publisher Graph** (`agent/subgraph/publisher_graph.py`) — pick & write:

- **Pick Events** (`event_picker`) — loads events from Postgres and has an LLM choose which are worth publishing, weighing relevance, confidence, and redundancy against the published articles loaded earlier.
- **RAG** (`fetch_rag`) — LLM turns the picked events into a semantic search query and retrieves supporting article excerpts from the pgvector store for grounding.
- **Generate Article** (`generate_article`) — writes a single 1-2 paragraph article grounded strictly in the retrieved excerpts and selected events, explicitly instructed to avoid rehashing the most recently published articles and write about what's new instead; persists the result to the published-articles store.

## Features

- **Query generation & self-critique** — an LLM drafts a boolean search query from the user profile, a second LLM pass critiques it for relevance and redundancy against what's already been published, and the query gets revised (up to a configurable retry cap) before articles are ever fetched.
- **Structured event extraction** — each fetched article is passed through a structured-output LLM call that pulls out discrete events (description, type, date, actors, confidence) rather than treating the whole article as one blob.
- **Similarity-based deduplication** — near-duplicate events (e.g. the same event reported by two sources) are clustered via TF-IDF cosine similarity and collapsed down to the single highest-confidence version. The reason why we chose a keywords-based embedding method like TF-IDF, instead of a semantic one, is because the articles are roughly about the same topic. Therefore the cosine similarity of their semantic embeddings will be almost always high and indistinguishable from duplicates and non-duplicates. 
- **Two storage backends, chosen deliberately per use case**:
  - Articles are chunked (`RecursiveCharacterTextSplitter`) and embedded into Postgres via `pgvector`/`langchain-postgres`, for semantic retrieval later.
  - Events are stored as plain rows (`id`, `content`, `metadata JSONB`) via raw `psycopg`, upserted by a stable id — no embeddings needed for structured event data.
- **Idempotent upserts everywhere** — articles, article chunks, and events are all keyed by deterministic ids, so re-running the pipeline updates existing rows instead of duplicating them.
- **Swappable news sources** — `news_client/` ships both a NewsAPI client and a Currents API client behind the same interface.

## Tech stack

- [LangGraph](https://github.com/langchain-ai/langgraph) for orchestration, [LangChain](https://github.com/langchain-ai/langchain) + OpenAI for the LLM/embedding calls
- [Postgres](https://www.postgresql.org/) with [pgvector](https://github.com/pgvector/pgvector) (via `langchain-postgres` for articles, raw `psycopg` for events)
- [Currents API](https://currentsapi.services/) / [NewsAPI](https://newsapi.org/) as news sources
- scikit-learn (TF-IDF + cosine similarity) for dedup

## Getting started

### Prerequisites

- Python 3.11+
- Docker (to run the application using LangGraph's built Dockerfile, as well as for running PostgreSQL and Redis instances)
- An OpenAI API key, and an API key from [Currents API](https://currentsapi.services/)

### Setup

```bash
git clone <this-repo>
cd agenticnewspublisher
pip install -r requirements.txt
```

Create a `.env` file in the project root:

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | LLM calls and embeddings |
| `CURRENTS_API` | Yes* | Currents API key (currently the active news source) |
| `NEWS_API` | Yes* | NewsAPI key (alternative client, not currently wired into the graph) |
| `DATABASE_URL` | Yes | Postgres connection string, e.g. `postgresql+psycopg://news:news@localhost:5432/news` |
| `USER_PROFILE` or `USER_PROFILE_FILEPATH` | Yes (one of them) | The reader's interests/preferences, in plain text — drives every prompt in the pipeline |
| `LANGSMITH_TRACING`, `LANGSMITH_PROJECT` | No | Optional [LangSmith](https://smith.langchain.com/) tracing |
| `LANGSMITH_API_KEY` | Yes, for Docker | Only needed to run the app via `docker compose`/the built image (see below) — the image is LangGraph Platform's licensed API server, which refuses to start without either this or a `LANGSMITH_API_KEY` from an account with LangGraph Cloud/Platform access |

\* only whichever news client you're actually using needs a valid key.

### Running the application

The app runs as a [LangGraph Platform](https://github.com/langchain-ai/langgraph) API server — the `Dockerfile` builds `agent/graph.py:graph` into that server, and `docker-compose.yml` wires it up together with the Postgres (`news_postgres`, pgvector-enabled, doubling as both the app's own article/event storage and the API server's control-plane DB) and Redis (`redis`) it needs.

1. **Start the stack:**

   ```bash
   docker compose up -d
   ```

   This builds/starts three services: `news_postgres` (port `5432`), `redis` (port `6379`), and `api` (port `8000`, the LangGraph API server). Check everything is healthy:

   ```bash
   docker compose ps
   ```

   All three should show `healthy`. If `api` keeps restarting, check its logs (`docker compose logs api`) — the most common cause is the license check above.

2. **Trigger a run** — the graph runs as a thread on the API server, not as a plain Python function call, so drive it through the API rather than importing `agent.graph` directly:

   ```bash
   python demo/demo.py
   ```

   This connects via `langgraph_sdk`, creates a thread, streams live per-node updates as the pipeline runs (search → critique → extract events → dedupe → store → pick events → RAG → write article), and prints the final `generated_article`. A full run makes real OpenAI/Currents API calls and can take a few minutes.

3. **Inspect the data directly**, if you want to see what got stored:

   ```bash
   docker compose exec news_postgres psql -U news -d news
   ```

   `\dt` lists tables — your own `events` and `langchain_pg_collection`/`langchain_pg_embedding` (articles + published articles, both pgvector-backed) live alongside the LangGraph API's own control-plane tables (`thread`, `run`, `checkpoints`, etc.) in the same database.

4. **Stop the stack** with `docker compose down` (add `-v` to also drop the Postgres/Redis volumes and start clean).

## Project structure

```
agent/
  graph.py              # top-level graph: search_graph -> publisher_graph
  state.py              # shared NewsState
  config.py             # user profile loading, thresholds, id namespaces, splitter config
  prompts.py             # every prompt used by the pipeline
  structured_outputs.py # pydantic schemas for structured LLM outputs
  subgraph/
    search_graph.py     # search -> fetch -> critique -> extract events -> dedupe -> store
    publisher_graph.py  # pick events -> retrieve context (RAG) -> generate article
news_client/
  newsapi_client.py     # NewsAPI client
  current_api.py        # Currents API client (same article shape as NewsAPI's)
storage/
  articles_vectorstore.py # pgvector-backed article store (chunking, dedup, similarity search)
  events_store.py          # plain Postgres/psycopg event store
demo/
  demo.py                  # example client: drives a run via langgraph_sdk against the API server
docker-compose.yml       # news_postgres (pgvector) + redis + the built API server
user_profile.txt         # example/default user profile
```
