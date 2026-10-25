from pydantic import BaseModel, BeforeValidator, Field, ValidationError
from agent.config import AGENT_NAMES, AGENT_DESCRIPTIONS
from typing import Annotated, Literal
import datetime

_NEXT_AGENT_DESCRIPTION = "The next agent to run. One of:\n" + "\n".join(
    f"- {name}: {description}" for name, description in AGENT_DESCRIPTIONS.items()
)

class SchedulerDecision(BaseModel):
    # """You have to choose the next agent"""
    next_agent: Literal[AGENT_NAMES] = Field(description=_NEXT_AGENT_DESCRIPTION)
    agent_prompt: str = Field(
        default="",
        description=(
            "Optional extra instructions or context to hand to the chosen agent for "
            "this run only (e.g. narrow the search, focus on a specific angle). "
            "Leave empty if the agent's default behavior is sufficient."
        ),
    )

# Normalisers for unexpected LLM output (placeholder dates, "85%", a string instead of a list, ...)
def _text(v) -> str:
    return " ".join(str(v or "").split())

def _description(v) -> str:
    if not (text := _text(v)):
        raise ValueError("empty event_description")  # the event is dropped by EventsExtraction
    return text

def _date(v) -> datetime.date | None:
    try:
        return datetime.date.fromisoformat(str(v).strip()[:10])
    except ValueError:  # "0000-01-01", "unknown", "2026-13-45", None
        return None

def _actors(v) -> list[str]:
    items = [v] if isinstance(v, str) else v if isinstance(v, (list, tuple, set)) else []
    return list(dict.fromkeys(filter(None, map(_text, items))))  # no blanks or duplicates, order kept

def _confidence(v) -> float:
    try:
        n = float(str(v).strip().rstrip("%"))
    except ValueError:
        return 0.0
    return 0.0 if n != n else min(max(n / 100 if 1 < n <= 100 else n, 0.0), 1.0)


class Event(BaseModel):
    event_description: Annotated[str, BeforeValidator(_description)] = Field(
        description="Short description of the event"
    )
    event_type: Annotated[str, BeforeValidator(lambda v: _text(v).lower() or "other")] = Field(
        description="Type of the event"
    )
    event_date: Annotated[datetime.date | None, BeforeValidator(_date)] = Field(
        description="The date of the event as YYYY-MM-DD. Null if the article does not give or imply a date."
    )
    actors: Annotated[list[str], BeforeValidator(_actors)] = Field(
        description="The list of actors in the event. Empty list if none are named."
    )
    confidence: Annotated[float, BeforeValidator(_confidence)] = Field(
        description="The estimated confidence in the accuracy of the event, a number between 0 and 1."
    )


def _valid_events(v) -> list[Event]:
    events = []
    for item in v if isinstance(v, list) else []:
        try:
            events.append(Event.model_validate(item))
        except ValidationError:  # keep the valid events instead of failing the whole extraction
            pass
    return events


class EventsExtraction(BaseModel):
    events: Annotated[list[Event], BeforeValidator(_valid_events)] = Field(
        description="Every discrete event described in the article. Empty list if the article describes no clear events."
    )


class QueryCritique(BaseModel):
    verdict: Literal["approve", "revise"] = Field(
        description=(
            "'approve' if the search query is well-aligned with the user profile and is "
            "turning up enough new, on-topic articles; 'revise' if it should be changed."
        )
    )
    is_redundant: bool = Field(
        description=(
            "True if the 10 most recently fetched articles largely describe the same "
            "underlying idea or event as one of the 5 most recently published articles "
            "(even if the wording, angle, or source differs); False if they cover "
            "genuinely new ground. This is judged independently of verdict: a query can "
            "be well-formed and on-topic (verdict='approve') while still turning up "
            "redundant coverage."
        )
    )
    feedback: str = Field(
        description=(
            "A concise explanation of what is or isn't working about the query, referencing "
            "the user's profile and how the fetched articles compare to what's already been "
            "published."
        )
    )
    revised_query: str = Field(
        default="",
        description=(
            "Only when verdict is 'revise': a corrected CurrentsAPI keyword query to try next, "
            "surrounded by ** the same way the search worker formats it. Leave empty when "
            "verdict is 'approve'."
        ),
    )

#publisher_graph.py
class EventSelection(BaseModel):
    event_ids: list[str] = Field(
        description=(
            "The event_id values of the events worth publishing, chosen from the "
            "candidate list, ordered most newsworthy first. Empty list if none of "
            "the candidates are worth publishing."
        )
    )