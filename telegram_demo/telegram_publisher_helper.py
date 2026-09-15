import html
import os
import dotenv

dotenv.load_dotenv()

import requests

API_BASE = "https://api.telegram.org"


class TelegramPublisher:
    def __init__(self, token: str = None, channel_id: str = None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.channel_id = channel_id or os.environ.get("TELEGRAM_CHANNEL_ID")
        if not self.token or not self.channel_id:
            raise ValueError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID (env or args).")

    def _call(self, method: str, data: dict = None, files: dict = None) -> dict:
        url = f"{API_BASE}/bot{self.token}/{method}"
        response = requests.post(url, data=data, files=files, timeout=30)
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(f"{method} failed: {payload.get('description')}")
        return payload["result"]

    def post_text(self, text: str, parse_mode: str = "HTML") -> int:
        """Post a text message. Returns the message id."""
        result = self._call("sendMessage", {
            "chat_id": self.channel_id,
            "text": text,
            "parse_mode": parse_mode,
        })
        return result["message_id"]

    def post_photo(self, photo, caption: str = None, parse_mode: str = "HTML") -> int:
        """Post a photo. `photo` can be a local file path or an image URL."""
        data = {"chat_id": self.channel_id, "caption": caption, "parse_mode": parse_mode}
        if isinstance(photo, str) and photo.startswith("http"):
            data["photo"] = photo
            result = self._call("sendPhoto", data)
        else:
            with open(photo, "rb") as f:
                result = self._call("sendPhoto", data, files={"photo": f})
        return result["message_id"]

    def post_news_item(self, title: str, summary: str, url: str,
                        source: str = None, image_url: str = None) -> int:
        """Post a formatted news item (title, summary, source, link)."""
        text = f"<b>{html.escape(title)}</b>\n\n{html.escape(summary)}"
        if source:
            text += f"\n\n<i>Source: {html.escape(source)}</i>"
        text += f'\n<a href="{html.escape(url)}">Read more</a>'

        if image_url:
            return self.post_photo(image_url, caption=text)
        return self.post_text(text)

    def delete_message(self, message_id: int) -> bool:
        self._call("deleteMessage", {"chat_id": self.channel_id, "message_id": message_id})
        return True


if __name__ == "__main__":
    bot = TelegramPublisher()
    msg_id = bot.post_text("Test post from telegram_publisher.py")
    print(f"Posted message {msg_id} to {bot.channel_id}")
