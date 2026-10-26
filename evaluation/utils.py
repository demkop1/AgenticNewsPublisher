import json

from langchain_core.documents import Document


def _default(value):
    if isinstance(value, Document):
        return {"page_content": value.page_content, "metadata": value.metadata}
    return str(value)  # dates, numpy floats, ...


def serialize_state(state: dict) -> dict:
    """JSON-safe copy of a graph/chain state (Documents become {page_content, metadata}),
    so it can be returned from an eval target and inspected in LangSmith."""
    return json.loads(json.dumps(state, default=_default))
