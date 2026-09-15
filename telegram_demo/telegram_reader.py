import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import dotenv

dotenv.load_dotenv()

from telethon import TelegramClient

DEFAULT_CHANNELS_FILE = Path(__file__).with_name("telegram_channels.txt")
SESSION_NAME = os.environ.get("TELEGRAM_SESSION_NAME", "telegram_reader")


class TelegramReader:

    def __init__(self, api_id: str = None, api_hash: str = None, session_name: str = None):
        self.api_id = api_id or os.environ.get("TELEGRAM_API_ID")
        self.api_hash = api_hash or os.environ.get("TELEGRAM_API_HASH")
        if not self.api_id or not self.api_hash:
            raise ValueError(
                "Set TELEGRAM_API_ID and TELEGRAM_API_HASH (env or args). "
                "Get them at https://my.telegram.org/apps"
            )
        session_path = Path(__file__).with_name(session_name or SESSION_NAME)
        print(int(self.api_id), self.api_hash)
        self.client = TelegramClient(str(session_path), int(self.api_id), self.api_hash)

    @staticmethod
    def load_channels(path=DEFAULT_CHANNELS_FILE) -> list[str]:
        """Read channel usernames/links from a txt file, one per line."""
        with open(path, encoding="utf-8") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]

    async def fetch_recent(self, channel: str, limit: int = 5, hours: int = 24) -> list[dict]:
        """Fetch recent text messages from a single channel."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        username = channel.lstrip("@")
        items = []
        async for msg in self.client.iter_messages(channel, limit=limit):
            if msg.date < cutoff:
                break
            if not msg.text:
                continue
            items.append({
                "channel": channel,
                "date": msg.date,
                "text": msg.text,
                "link": f"https://t.me/{username}/{msg.id}",
            })
        return items

    async def fetch_all(self, channels: list[str] = None, limit: int = 5, hours: int = 24) -> list[dict]:
        """Fetch recent messages across all channels, sorted newest first."""
        channels = channels or self.load_channels()
        results = []
        async with self.client:
            for channel in channels:
                try:
                    results.extend(await self.fetch_recent(channel, limit=limit, hours=hours))
                except Exception as e:
                    print(f"Skipping {channel}: {e}")
        results.sort(key=lambda item: item["date"], reverse=True)
        return results


def format_news_list(items: list[dict]) -> str:
    lines = []
    for item in items:
        ts = item["date"].strftime("%Y-%m-%d %H:%M UTC")
        snippet = " ".join(item["text"].split())
        if len(snippet) > 200:
            snippet = snippet[:200].rstrip() + "…"
        lines.append(f"- [{ts}] {item['channel']}: {snippet}\n  {item['link']}")
    return "\n".join(lines)


async def main():
    reader = TelegramReader()
    news = await reader.fetch_all(limit=5, hours=24)
    print(format_news_list(news) if news else "No recent messages found.")

if __name__ == "__main__":
    asyncio.run(main())
