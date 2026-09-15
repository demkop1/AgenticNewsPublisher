from pydantic import BaseModel, Field
from agent.config import AGENT_NAMES, AGENT_DESCRIPTIONS
from typing import Literal
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

class Event(BaseModel):
    event_description: str = Field(description="Short description of the event")
    event_type: str = Field(description="Type of the event")
    event_date: datetime.date = Field(description="The date of the event")
    actors: list[str] = Field(description="The list of actors in the event")
    confidence: float = Field(description="The estiamted confidence of the accuracy of the event.")


class EventsExtraction(BaseModel):
    events: list[Event] = Field(
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