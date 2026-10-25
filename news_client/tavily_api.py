import datetime
import os
import dotenv

dotenv.load_dotenv()

from tavily import TavilyClient

MAX_QUERY_CHARS = 400  # Tavily rejects longer queries
DEFAULT_DAYS_BACK = 30


class TavilyNewsClient:
    def __init__(self, api_key: str = None):
        api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("Set TAVILY_API_KEY (env or arg).")
        self._client = TavilyClient(api_key=api_key)

    @staticmethod
    def _normalize_article(a: dict) -> dict:
        """Reshape a Tavily result into the same article shape CurrentsAPIClient
        returns (title, content, url, source.name, publishedAt, ...)."""
        url = a.get("url") or ""
        return {
            "title": a.get("title", ""),
            "content": a.get("content", ""),
            "url": url,
            "source": {"name": url.split("/")[2] if url.count("/") >= 2 else "Tavily"},
            "author": None,
            "publishedAt": a.get("published_date"),
            "urlToImage": None,
            "language": "en",
            "category": None,
            "score": a.get("score"),
        }

    def search(self, query: str, start_date: datetime.date | None = None, max_results: int = 20,
               search_depth: str = "basic") -> list:
        """Search news published on or after `start_date` (default: 7 days ago, UTC) that matches
        a natural-language query. Returns a list of article dicts."""
        start_date = start_date or (
            datetime.datetime.now(datetime.timezone.utc).date() - datetime.timedelta(days=DEFAULT_DAYS_BACK)
        )
        response = self._client.search(
            query=query[:MAX_QUERY_CHARS],
            topic="news",
            start_date=start_date.isoformat(),
            max_results=max_results,
            search_depth=search_depth,
        )
        return [self._normalize_article(a) for a in response.get("results", [])]


if __name__ == "__main__":
    for a in TavilyNewsClient().search("latest developments in the war in Ukraine")[:5]:
        print(f"- {a['title']} ({a['source']['name']}) - {a['publishedAt']}")
