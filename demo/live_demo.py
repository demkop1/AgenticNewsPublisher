"""
Live newsroom dashboard for the multi-agent news pipeline.

Like demo.py, it talks to a running LangGraph API server (the `api` service in
docker-compose.yml) via langgraph_sdk, but instead of printing log lines it
renders a full-screen terminal dashboard (built with `rich`):

  * a pipeline tracker showing which agent node is working right now,
  * an activity feed narrating what each agent produced,
  * the newest article, plus a running "front page" of everything published,
  * a countdown to the next check.

Every check runs on a fresh thread; the published-articles store in Postgres is
what keeps later checks from repeating earlier articles. Stop with Ctrl-C.

Usage:
    docker compose up -d                       # redis / news_postgres / api
    python demo/live_demo.py                   # check every 10 minutes
    python demo/live_demo.py --interval 120    # every 2 minutes
    python demo/live_demo.py --max-runs 3      # stop after 3 checks

Each check makes real OpenAI and Currents API calls, so mind your quotas.

Env vars (optional): LANGGRAPH_API_URL (default http://localhost:8000),
LANGGRAPH_API_KEY.
"""
import argparse
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

sys.path.append(os.getcwd())

import dotenv
from langgraph_sdk import get_sync_client
from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent.config import USER_PROFILE

dotenv.load_dotenv()

LANGGRAPH_API_URL = os.environ.get("LANGGRAPH_API_URL", "http://localhost:8000")
LANGGRAPH_API_KEY = os.environ.get("LANGGRAPH_API_KEY")
GRAPH_ID = "graph"
DEFAULT_INTERVAL_SECONDS = 600

# (node name, label, graph) in execution order.
PIPELINE = [
    ("search_worker", "Search", "search"),
    ("load_published_articles", "Load Published", "search"),
    ("fetch_articles_worker", "Fetch Articles", "search"),
    ("check_relevance", "Check Relevance", "search"),
    ("criticize", "Critic", "search"),
    ("create_events", "Create Events", "search"),
    ("deduplicate", "Deduplicate", "search"),
    ("store_articles_worker", "Store", "search"),
    ("events_reranker", "Rerank Events", "publish"),
    ("generate_article", "Generate Article", "publish"),
]
NODE_INDEX = {name: i for i, (name, _, _) in enumerate(PIPELINE)}


@dataclass
class Article:
    number: int
    time: str
    text: str


@dataclass
class Dashboard:
    """Shared state written by the worker thread and read by the renderer."""
    interval: float
    max_runs: int | None
    checks: int = 0
    failures: int = 0
    status: str = "starting"               # starting | running | waiting | stopped
    next_check_at: float = 0.0
    done: set = field(default_factory=set)
    current: str | None = None
    passes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)  # event descriptions of this check
    no_relevant: bool = False              # this check ended with no relevant articles
    log: deque = field(default_factory=lambda: deque(maxlen=200))
    articles: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def say(self, message: str, style: str = "white") -> None:
        with self.lock:
            self.log.append((time.strftime("%H:%M:%S"), message, style))

    def reset_run(self) -> None:
        with self.lock:
            self.done, self.current, self.passes, self.events = set(), PIPELINE[0][0], {}, []
            self.no_relevant = False


def short(value, limit: int = 100) -> str:
    value = " ".join(str(value).split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def narrate(dash: Dashboard, node: str, update: dict) -> None:
    """Turn a node's state update into a human-readable activity line."""
    update = update if isinstance(update, dict) else {}
    if node == "search_worker" and update.get("search_query"):
        dash.say(f"Search agent drafted query: {short(update['search_query'])}", "cyan")
    elif node == "load_published_articles":
        dash.say(f"Loaded {len(update.get('published_articles') or [])} previously published article(s) for context", "blue")
    elif node == "fetch_articles_worker":
        dash.say(f"Fetched {len(update.get('current_articles') or [])} article(s) from the news API", "blue")
    elif node == "check_relevance":
        kept = len(update.get("current_articles") or [])
        scores = update.get("article_scores")
        if kept:
            dash.say(f"{kept} of {len(scores or [])} article(s) passed the relevance check", "green")
        elif "current_articles" in update:
            dash.say("No article passed the relevance check and the critic's retries are used up", "bold yellow")
        elif scores is not None:
            dash.say("No article passed the relevance check; asking the critic for a new query", "yellow")
    elif node == "criticize":
        dash.say(f"Critic reviewed coverage (pass {update.get('critique_attempts', '?')})", "yellow")
    elif node == "create_events":
        dash.say(f"Extracted {len(update.get('events') or [])} event(s) from the articles", "magenta")
    elif node == "deduplicate":
        events = update.get("events") or []
        dash.say(f"{len(events)} distinct event(s) after de-duplication", "magenta")
    elif node == "store_articles_worker":
        dash.say(
            f"Stored {len(update.get('stored_article_ids') or [])} article(s) and "
            f"{len(update.get('stored_event_ids') or [])} event(s)", "blue",
        )
    elif node == "events_reranker":
        dash.say("Reranker kept the events most relevant to the profile", "green")
    elif node == "generate_article":
        dash.say("Writer agent finished a new article", "bold green")
    else:
        dash.say(f"{node} finished", "white")


def run_check(client, dash: Dashboard) -> str | None:
    """Run the pipeline once on a fresh thread, updating the dashboard live."""
    dash.reset_run()
    thread = client.threads.create()
    article = None

    for chunk in client.runs.stream(
        thread["thread_id"],
        GRAPH_ID,
        input={"user_profile": USER_PROFILE},
        stream_mode=["updates", "values"],
        stream_subgraphs=True,
    ):
        if chunk.event == "error":
            raise RuntimeError(chunk.data)
        kind = chunk.event.split("|")[0]
        data = chunk.data
        if not isinstance(data, dict):
            continue

        if kind == "updates":
            for node, update in data.items():
                if node not in NODE_INDEX:
                    continue
                with dash.lock:
                    dash.done.add(node)
                    dash.passes[node] = dash.passes.get(node, 0) + 1
                    nxt = NODE_INDEX[node] + 1
                    if node == "check_relevance":
                        update_dict = update if isinstance(update, dict) else {}
                        if update_dict.get("current_articles"):
                            nxt = NODE_INDEX["create_events"]
                        elif "current_articles" in update_dict:
                            # emptied list: no relevant articles and critic attempts exhausted, graph ended
                            dash.no_relevant = True
                            nxt = len(PIPELINE)
                    if node == "criticize" and isinstance(update, dict) and "agent_chats" in update:
                        nxt = NODE_INDEX["search_worker"]  # critic asked for a revised query
                    dash.current = PIPELINE[nxt][0] if nxt < len(PIPELINE) else None
                narrate(dash, node, update)
        elif kind == "values":
            if data.get("generated_article"):
                article = data["generated_article"]
            events = data.get("events") or []
            with dash.lock:
                dash.events = [
                    getattr(e, "page_content", None) or (e.get("page_content") if isinstance(e, dict) else str(e))
                    for e in events
                ]

    with dash.lock:
        # The publisher stage is skipped when the search produced no events.
        if article is None and "events_reranker" not in dash.done:
            dash.no_relevant = True
    if article is None and not dash.no_relevant:
        state = client.threads.get_state(thread["thread_id"])
        article = state["values"].get("generated_article")
    return article


def worker(dash: Dashboard, stop: threading.Event) -> None:
    client = get_sync_client(url=LANGGRAPH_API_URL, api_key=LANGGRAPH_API_KEY)
    while not stop.is_set() and (dash.max_runs is None or dash.checks < dash.max_runs):
        dash.checks += 1
        dash.status = "running"
        dash.say(f"Check #{dash.checks}: scanning the web for news...", "bold cyan")
        try:
            article = run_check(client, dash)
        except Exception as exc:  # server restarting, quota hit, ...: keep monitoring
            dash.failures += 1
            dash.say(f"Check failed ({type(exc).__name__}: {short(exc, 80)}); will retry", "bold red")
        else:
            if article:
                with dash.lock:
                    dash.articles.append(Article(len(dash.articles) + 1, time.strftime("%H:%M"), article.strip()))
                dash.say("New article published to the front page", "bold green")
            elif dash.no_relevant:
                dash.say("No relevant articles or events found; nothing to publish", "bold yellow")
            else:
                dash.say("Nothing new worth publishing this time", "yellow")
        with dash.lock:
            dash.current = None

        if dash.max_runs is not None and dash.checks >= dash.max_runs:
            break
        dash.status = "waiting"
        dash.next_check_at = time.time() + dash.interval
        stop.wait(dash.interval)
    dash.status = "stopped"


# ---------------------------------------------------------------- rendering

def render_header(dash: Dashboard) -> Panel:
    title = Text("AGENTIC NEWS PUBLISHER", style="bold white")
    title.append("  ·  live newsroom", style="dim")
    profile = Text(f"Covering: {short(USER_PROFILE.strip(), 150)}", style="italic cyan")
    return Panel(Group(title, profile), box=box.HEAVY, border_style="bright_blue", padding=(0, 2))


def render_pipeline(dash: Dashboard) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(width=2)
    table.add_column()
    table.add_column(justify="right", style="dim")
    last_graph = None
    with dash.lock:
        done, current, passes = set(dash.done), dash.current, dict(dash.passes)
    running = dash.status == "running"
    for name, label, graph in PIPELINE:
        if graph != last_graph:
            heading = "SEARCH GRAPH" if graph == "search" else "PUBLISHER GRAPH"
            table.add_row("", Text(heading, style="bold dim"), "")
            last_graph = graph
        if running and name == current:
            icon, style = Text("●", style="bold yellow"), "bold yellow"
        elif name in done:
            icon, style = Text("✔", style="green"), "green"
        else:
            icon, style = Text("○", style="dim"), "dim"
        count = f"×{passes[name]}" if passes.get(name, 0) > 1 else ""
        table.add_row(icon, Text(label, style=style), count)
    return Panel(table, title="Agent pipeline", border_style="blue", padding=(1, 2))


def render_activity(dash: Dashboard, height: int) -> Panel:
    with dash.lock:
        entries = list(dash.log)[-max(height - 2, 1):]
    body = Text()
    for stamp, message, style in entries:
        body.append(f"{stamp}  ", style="dim")
        body.append(message + "\n", style=style)
    return Panel(body, title="Agent activity", border_style="blue", padding=(0, 1))


def render_article(dash: Dashboard) -> Panel:
    with dash.lock:
        articles = list(dash.articles)
    if not articles:
        waiting = "Writer agent is on it..." if dash.status == "running" else "No articles yet."
        if dash.no_relevant and dash.status != "running":
            waiting = "No relevant articles found in the last check."
        return Panel(Text(waiting, style="dim italic"), title="Latest article", border_style="green", padding=(1, 2))
    latest = articles[-1]
    return Panel(
        Markdown(latest.text),
        title=f"Latest article · #{latest.number} · {latest.time}",
        border_style="green", padding=(1, 2),
    )


def render_feed(dash: Dashboard) -> Panel:
    with dash.lock:
        articles = list(dash.articles)
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", width=6)
    table.add_column(overflow="ellipsis", no_wrap=True)
    for a in reversed(articles[:-1][-6:]):
        table.add_row(a.time, short(a.text, 90))
    if not articles[:-1]:
        table.add_row("", Text("Earlier articles will appear here.", style="dim italic"))
    return Panel(table, title="Front page history", border_style="green", padding=(0, 1))


def render_footer(dash: Dashboard) -> Panel:
    if dash.status == "running":
        state = Text("● scanning", style="bold yellow")
    elif dash.status == "waiting":
        left = max(0, int(dash.next_check_at - time.time()))
        state = Text(f"⏱ next check in {left // 60:02d}:{left % 60:02d}", style="bold cyan")
    elif dash.status == "stopped":
        state = Text("■ finished", style="bold")
    else:
        state = Text("starting...", style="dim")
    line = Text.assemble(
        state, "   |   ",
        (f"checks {dash.checks}", "white"), "   ",
        (f"articles {len(dash.articles)}", "green"), "   ",
        (f"failures {dash.failures}", "red" if dash.failures else "dim"), "   |   ",
        ("Ctrl-C to quit", "dim"),
    )
    return Panel(line, box=box.ROUNDED, border_style="bright_blue", padding=(0, 2))


def render(dash: Dashboard, console: Console) -> Layout:
    layout = Layout()
    layout.split_column(Layout(name="header", size=4), Layout(name="body"), Layout(name="footer", size=3))
    layout["body"].split_row(Layout(name="left", ratio=2), Layout(name="right", ratio=3))
    layout["left"].split_column(Layout(name="pipeline", size=len(PIPELINE) + 6), Layout(name="activity"))
    layout["right"].split_column(Layout(name="article", ratio=3), Layout(name="feed", size=9))

    layout["header"].update(render_header(dash))
    layout["pipeline"].update(render_pipeline(dash))
    activity_height = max(console.height - 4 - 3 - (len(PIPELINE) + 6), 4)
    layout["activity"].update(render_activity(dash, activity_height))
    layout["article"].update(render_article(dash))
    layout["feed"].update(render_feed(dash))
    layout["footer"].update(render_footer(dash))
    return layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live dashboard for the continuously running news pipeline.")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SECONDS,
                        help=f"seconds to wait between checks (default {DEFAULT_INTERVAL_SECONDS})")
    parser.add_argument("--max-runs", type=int, help="stop after this many checks (default: until Ctrl-C)")
    args = parser.parse_args()
    if args.interval < 0:
        parser.error("--interval must not be negative")
    return args


def main() -> None:
    args = parse_args()
    console = Console()
    dash = Dashboard(interval=args.interval, max_runs=args.max_runs)
    stop = threading.Event()
    thread = threading.Thread(target=worker, args=(dash, stop), daemon=True)
    thread.start()

    try:
        with Live(render(dash, console), console=console, screen=True, refresh_per_second=4) as live:
            while thread.is_alive():
                live.update(render(dash, console))
                time.sleep(0.25)
            live.update(render(dash, console))
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()

    console.print(f"Stopped after {dash.checks} check(s); {len(dash.articles)} article(s) written.")
    for a in dash.articles:
        console.print(Panel(Markdown(a.text), title=f"Article #{a.number} · {a.time}", border_style="green"))


if __name__ == "__main__":
    main()
