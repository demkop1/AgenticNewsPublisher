# NewsAgent

An agentic news pipeline built on [LangGraph](https://github.com/langchain-ai/langgraph) that searches for news matching a user's stated interests, extracts and deduplicates discrete events from the coverage, and reranks those events by relevance to the reader before writing up the best ones — as a step toward auto-publishing curated news via API.

## How it works

The graph is composed of two LangGraph subgraphs, each a mix of LLM-driven agents and deterministic steps:

Hand-drawn with matplotlib (`docs/draw_graph_diagram.py`) rather than exported from LangGraph, so update the script when the nodes or wiring change:

<img src="docs/graph-diagram.png" alt="Graph structure" width="450">

Everything shares one `NewsState` (`agent/state.py`) as it flows through the graph, and every LLM call is grounded in a single user profile (`user_profile.txt` or `USER_PROFILE` in `agent/config.py`) so the whole pipeline stays personalized to what that user actually wants to read. The profile can also be passed per run as `user_profile` in the input state (this is how the evaluations run many profiles); the environment profile is the default. All models and shared clients (LLMs, embeddings, rerankers, news clients) are defined once in `agent/config.py`.

### Node reference

**Search Graph** (`agent/subgraph/search_graph.py`, Tavily; CurrentsAPI backup in `search_graph_currents.py`) — find, extract, store:

- **Search** (`search_worker`) — LLM drafts a natural-language Tavily news query (no boolean operators) from the user profile (and, on a retry, the critic's feedback).
- **Load Published** (`load_published_articles`) — semantic search over the published-articles store for the most recent articles related to the current query, so the writer can avoid rehashing what has already been covered.
- **Fetch Articles** (`fetch_articles_worker`) — calls Tavily's news search (articles from the last 7 days, `news_client/tavily_api.py`) with the query and normalizes the results into `Document`s.
- **Check Relevance** (`check_relevance`) — a reranker model (`RERANKER_MODEL`) scores every fetched article against the search query and keeps those above `RELEVANCE_THRESHOLD`. If at least one article passes, the graph goes straight to Create Events; if none do, it routes to the Critic, or ends the graph once the critic retry cap is used up. Scores are in [0, 1] (the reranker output is passed through a sigmoid). The scores of all fetched articles are kept in state (`article_scores`).
- **Critic** (`criticize`) — only reached when no fetched article passed Check Relevance. It does not decide whether to revise: an LLM looks at the query and the closest articles (with their scores), explains why nothing relevant came back, and proposes a new query, then the graph returns to Search. The number of such retries is capped by `MAX_CRITIQUE_ATTEMPTS`; when it is used up and nothing is relevant, Check Relevance ends the graph instead.
- **Create Events** (`create_events`) — per-article structured-output LLM call that extracts discrete events (description, type, date, actors, confidence). Outputs are validated and normalized (placeholder dates become null, confidence is clamped to 0–1, malformed events are dropped), so one bad event does not fail the extraction.
- **Deduplicate** (`deduplicate`) — TF-IDF cosine-similarity clustering of events extracted in this run; keeps only the highest-confidence event per cluster.
- **Store** (`store_articles_worker`) — chunks and embeds articles into the pgvector store, and upserts events into Postgres.

**Top-level graph** (`agent/graph.py`) runs the search graph, then the publisher graph only if the search produced events; otherwise the run ends without writing anything.

**Publisher Graph** (`agent/subgraph/publisher_graph.py`) — pick & write:

- **Rerank Events** (`events_reranker`) — a reranker model scores this run's events against the user profile and keeps the 5 most relevant.
- **Generate Article** (`generate_article`) — writes a single 1-2 paragraph article grounded strictly in the selected events, explicitly instructed to avoid rehashing the most recently published articles and write about what's new instead; persists the result to the published-articles store.

## Features

- **Query generation, relevance filtering & self-critique** — an LLM drafts a natural-language search query from the user profile, a reranker scores the fetched articles against it and drops the irrelevant ones, and when none are relevant a second LLM pass explains why and proposes a new query (up to a configurable retry cap).
- **Structured event extraction** — each relevant article is passed through a structured-output LLM call that pulls out discrete events (description, type, date, actors, confidence) rather than treating the whole article as one blob. The output schema normalizes unexpected LLM output instead of failing on it.
- **Similarity-based deduplication** — near-duplicate events (e.g. the same event reported by two sources) are clustered via TF-IDF cosine similarity and collapsed down to the single highest-confidence version. The reason why we chose a keywords-based embedding method like TF-IDF, instead of a semantic one, is because the articles are roughly about the same topic. Therefore the cosine similarity of their semantic embeddings will be almost always high and indistinguishable from duplicates and non-duplicates. 
- **Two storage backends, chosen deliberately per use case**:
  - Articles are chunked (`RecursiveCharacterTextSplitter`) and embedded into Postgres via `pgvector`/`langchain-postgres`, for semantic retrieval later.
  - Events are stored as plain rows (`id`, `content`, `metadata JSONB`) via raw `psycopg`, upserted by a stable id — no embeddings needed for structured event data.
- **Idempotent upserts everywhere** — articles, article chunks, and events are all keyed by deterministic ids, so re-running the pipeline updates existing rows instead of duplicating them.
- **Relevance-ranked publishing** — a reranker model scores this run's events against the user profile and the writer works from the 5 most relevant ones (`N_TOP_EVENTS`).
- **Swappable news sources** — Tavily is the primary source; `news_client/` also ships Currents API and NewsAPI clients behind the same article shape, and the Currents version of the search graph is kept as a self-contained backup (`search_graph_currents.py`).
- **Evaluation & baseline** — LangSmith experiments score the search stage, the full graph, and a plain LangChain chain baseline on the same user profiles (see [Evaluation](#evaluation)).

## Tech stack

- [LangGraph](https://github.com/langchain-ai/langgraph) for orchestration, [LangChain](https://github.com/langchain-ai/langchain) + OpenAI for the LLM/embedding calls
- [Postgres](https://www.postgresql.org/) with [pgvector](https://github.com/pgvector/pgvector) (via `langchain-postgres` for articles, raw `psycopg` for events)
- [Tavily](https://tavily.com/) as the news source ([Currents API](https://currentsapi.services/) / [NewsAPI](https://newsapi.org/) as alternatives)
- [sentence-transformers](https://www.sbert.net/) rerankers (`Qwen/Qwen3-Reranker-0.6B`) for article and event relevance
- scikit-learn (TF-IDF + cosine similarity) for dedup
- [LangSmith](https://smith.langchain.com/) + [openevals](https://github.com/langchain-ai/openevals) for evaluation

## Getting started

### Prerequisites

- Python 3.11+
- Docker (to run the application using LangGraph's built Dockerfile, as well as for running PostgreSQL and Redis instances)
- An OpenAI API key and an API key from [Tavily](https://tavily.com/) (plus a [Currents API](https://currentsapi.services/) key, which `agent/config.py` currently requires at import)

### Setup

```bash
git clone <this-repo>
cd agenticnewspublisher
pip install -r requirements.txt
```

The first run downloads the reranker model (about 1 GB) from Hugging Face.

Create a `.env` file in the project root:

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | LLM calls and embeddings |
| `TAVILY_API_KEY` | Yes | Tavily key (the active news source) |
| `CURRENTS_API` | Yes | Currents API key (used by the backup search graph; the client is created in `agent/config.py`) |
| `NEWS_API` | No | NewsAPI key (alternative client, not wired into the graph) |
| `DATABASE_URL` | Yes | Postgres connection string, e.g. `postgresql+psycopg://news:news@localhost:5432/news` |
| `USER_PROFILE` or `USER_PROFILE_FILEPATH` | Yes (one of them) | The reader's interests/preferences, in plain text — drives every prompt in the pipeline |
| `LANGSMITH_TRACING`, `LANGSMITH_PROJECT` | No | Optional [LangSmith](https://smith.langchain.com/) tracing |
| `LANGSMITH_USER_PREFERENCES_DATASET` | No | Name of the LangSmith dataset the evaluations use (default `news-user-preferences`) |
| `LANGSMITH_API_KEY` | Yes, for Docker and evaluation | Needed to run the app via `docker compose`/the built image (see below), and to run the LangSmith evaluations — the image is LangGraph Platform's licensed API server, which refuses to start without either this or a `LANGSMITH_API_KEY` from an account with LangGraph Cloud/Platform access |

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

   This connects via `langgraph_sdk` and re-runs the pipeline on a fresh thread every `--interval` seconds (default 600; `--max-runs N` stops after N checks, otherwise Ctrl-C), printing one progress line per step and each article it writes. A full run makes real OpenAI/Tavily API calls and can take a few minutes.

   For a full-screen dashboard instead of log lines, see [Live dashboard](#live-dashboard-demolive_demopy) below.

3. **Inspect the data directly**, if you want to see what got stored:

   ```bash
   docker compose exec news_postgres psql -U news -d news
   ```

   `\dt` lists tables — your own `events` and `langchain_pg_collection`/`langchain_pg_embedding` (articles + published articles, both pgvector-backed) live alongside the LangGraph API's own control-plane tables (`thread`, `run`, `checkpoints`, etc.) in the same database.

4. **Stop the stack** with `docker compose down` (add `-v` to also drop the Postgres/Redis volumes and start clean).

## Live dashboard (`demo/live_demo.py`)

<p align="center">
  <img src="docs/live_dashboard.png" alt="A screenshot of the live demo's dashboard" width="900">
</p>

`live_demo.py` does what `demo.py` does — it keeps re-running the pipeline against the API server, one fresh thread per check — but renders it as a full-screen terminal dashboard built with [`rich`](https://github.com/Textualize/rich) (included in `requirements.txt`).

```bash
docker compose up -d                       # the api, redis and news_postgres services must be up
python demo/live_demo.py                   # check every 10 minutes until Ctrl-C
python demo/live_demo.py --interval 120    # check every 2 minutes
python demo/live_demo.py --max-runs 3      # stop after 3 checks
```

Run it from the repository root, in a real terminal (it uses the alternate screen). Options:

- `--interval SECONDS` — wait between checks (default `600`, must not be negative).
- `--max-runs N` — stop after N checks (default: run until Ctrl-C).
- `LANGGRAPH_API_URL` / `LANGGRAPH_API_KEY` env vars (or `.env`) — server URL (default `http://localhost:8000`) and optional API key.

The screen shows:

- **Agent pipeline** — every node of the search and publisher graphs: `●` running, `✔` done, `○` pending, `×N` if the node ran several times (e.g. the critic sent the search back for a revised query).
- **Agent activity** — a timestamped feed of what each agent produced (drafted query, articles fetched, critic pass, events extracted, ...).
- **Latest article** and **Front page history** — the newest article rendered as Markdown, and earlier ones below it.
- **Footer** — status or countdown to the next check, plus check / article / failure counters.

A failed check (server restarting, API quota hit) is logged in the feed and retried at the next interval. On exit (Ctrl-C or `--max-runs`), the counts and every article written during the session are printed. Each check makes real OpenAI and Tavily API calls (about 3–4 minutes each), so mind your quotas when choosing a short interval.

## Evaluation

The `evaluation/` folder runs [LangSmith](https://smith.langchain.com/) experiments over a dataset of 100 realistic user profiles (`evaluation/data/user_profiles.txt`), running the agents concurrently across the profiles. Run the scripts from the repository root.

```bash
python -m evaluation.push_user_preferences_dataset   # make the LangSmith dataset match user_profiles.txt
python -m evaluation.search_eval                     # search stage only (search -> fetch -> relevance check -> critic)
python -m evaluation.graph_eval                      # the whole graph (search graph, then publisher graph if there are events)
python -m evaluation.baseline_eval                   # the plain LangChain chain baseline, for comparison
```

The criteria are shared and live in `evaluation/criteria.py`:

- `has_articles` — whether any articles were retrieved.
- `relevance` — LLM-as-a-judge (openevals): the fraction of retrieved articles relevant to the profile. Gives no score when the query returned no articles.
- `crossencoder_score` — mean reranker score (0–1) of the fetched articles against the search query.
- `events_confidence` — mean confidence of the extracted events (graph only; the baseline has no events).
- `article_quality` — LLM-as-a-judge: relevance to the profile and quality of the generated article.

Notes:

- `search_eval.py` skips event storage, so it is safe to run. `graph_eval.py` runs the full graph and **writes articles, events and published articles to the database `DATABASE_URL` points at** — use a scratch database for it. The baseline never touches the database.
- Each target also returns its final `state` (serialized by `evaluation/utils.py`) so runs can be inspected in LangSmith; `graph_eval.py` returns the search and publisher states separately.
- The scripts evaluate only the first few examples by default (`limit=` in the `evaluate(...)` call); remove it for a full run.
- `agent_baseline/` is a plain LangChain chain (LLM writes the search query → Tavily search → LLM writes the article) with no filtering, critic, events, or storage, and no imports from `agent/`.
- Example IDs in the dataset are derived from `_ID_NAMESPACE` in the push script. LangSmith keeps the IDs of deleted examples reserved, so if you delete examples by hand and a re-push fails with a 409, change that namespace UUID.

## Project structure

```
agent/
  graph.py              # top-level graph: search_graph -> publisher_graph (only if events were found)
  state.py              # shared NewsState
  config.py             # user profile loading, thresholds, id namespaces, shared LLMs/rerankers/clients
  prompts.py            # every prompt used by the pipeline
  structured_outputs.py # pydantic schemas for structured LLM outputs (with output normalization)
  subgraph/
    search_graph.py     # search -> fetch (Tavily) -> check relevance (-> critique -> search again) -> extract events -> dedupe -> store
    search_graph_currents.py  # backup: same pipeline fetching from CurrentsAPI (self-contained)
    publisher_graph.py  # rerank events -> generate article
agent_baseline/         # baseline: a plain LangChain chain (search query -> Tavily -> article), no imports from agent/
  chain.py              # the chain
  config.py  prompts.py  state.py
evaluation/
  criteria.py           # shared evaluators (has_articles, relevance, crossencoder_score, events_confidence, article_quality)
  search_eval.py        # LangSmith experiment: search stage
  graph_eval.py         # LangSmith experiment: whole graph
  utils.py              # serialize_state: JSON-safe state for eval outputs
  baseline_eval.py      # LangSmith experiment: baseline chain
  push_user_preferences_dataset.py  # syncs data/user_profiles.txt to the LangSmith dataset
  data/user_profiles.txt  # 100 user profiles for the experiments
news_client/
  tavily_api.py         # Tavily client (primary; news from the last 7 days, same article shape)
  newsapi_client.py     # NewsAPI client
  current_api.py        # Currents API client (same article shape as NewsAPI's)
storage/
  articles_vectorstore.py # pgvector-backed article store (chunking, dedup, similarity search)
  events_store.py          # plain Postgres/psycopg event store
demo/
  demo.py                  # example client: drives a run via langgraph_sdk against the API server
  live_demo.py             # same, as a full-screen rich dashboard
docker-compose.yml       # news_postgres (pgvector) + redis + the built API server
user_profile.txt         # example/default user profile
```
