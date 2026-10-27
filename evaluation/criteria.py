"""Evaluation criteria (LangSmith evaluators) shared by the eval scripts.

Each evaluator reads the keys the eval targets return: num_articles, articles,
article_scores, event_confidences and generated_article.
"""
from openevals.llm import create_llm_as_judge

from agent.config import JUDGE_LLM, RELEVANCE_THRESHOLD

STRICT_JSON_SCHEMA = {"method": "json_schema", "strict": True}


def has_articles(outputs: dict) -> bool:
    return outputs["num_articles"] > 0

RELEVANCE_PROMPT = """
You are a strict evaluator of news retrieval. The input contains the user profile describing
what a news channel wants to publish. The output contains the retrieved articles. For each
article decide whether it is relevant to the profile: it must be about the topics the profile
cares about and, where the profile specifies one, match its region, audience and angle.
Tangential or merely keyword-matching articles are not relevant.

Score the fraction of retrieved articles that are relevant, as a number between 0 and 1.
In your reasoning, list each article with a relevant / not relevant verdict.

<input>
{inputs}
</input>

<output>
{outputs}
</output>
"""

_judge = create_llm_as_judge(
    prompt=RELEVANCE_PROMPT,
    judge=JUDGE_LLM,
    feedback_key="article_relevance",
    continuous=True,    # Provider-native strict JSON schema: every field (score, reasoning) is guaranteed, unlike
    # the default function-calling mode where the model can omit "reasoning" (-> KeyError).
    structured_output_kwargs=STRICT_JSON_SCHEMA,
)


def relevance(inputs: dict, outputs: dict) -> dict:
    """LLM-as-a-judge: fraction of the retrieved articles relevant to the user profile."""
    if not outputs["articles"]:
        return {"key": "article_relevance", "score": None, "comment": "The query returned no articles; nothing to score."}
    return _judge(
        inputs=inputs["user_profile"],
        outputs="\n\n".join(f"[{i}] {a}" for i, a in enumerate(outputs["articles"])),
    )


def events_confidence(outputs: dict) -> dict:
    """Mean confidence score of the events extracted from this run's articles."""
    confidences = outputs["event_confidences"]
    if not confidences:
        return {"key": "events_confidence", "score": None, "comment": "No events extracted."}
    return {"key": "events_confidence", "score": sum(confidences) / len(confidences)}


def crossencoder_score(outputs: dict) -> dict:
    """Mean cross-encoder score (article vs. search query) over all fetched articles,
    including the ones check_relevance filtered out."""
    scores = outputs["article_scores"]
    if not scores:
        return {"key": "crossencoder_score", "score": None, "comment": "No articles scored."}
    n_kept = sum(s > RELEVANCE_THRESHOLD for s in scores)
    return {
        "key": "crossencoder_score",
        "score": sum(scores) / len(scores),
        "comment": f"{n_kept}/{len(scores)} articles above threshold {RELEVANCE_THRESHOLD}",
    }


ARTICLE_QUALITY_PROMPT = """
You are a strict editor evaluating a news article written for a channel. The input is the
user profile describing what the channel wants to publish. The output is the generated
article. The events the writer was given are listed in the sources, so you can see what
material was available. Judge the article on both of these:

1. Relevance - the article is about the topics the profile cares about, matches its region,
   audience, angle and tone, and respects any exclusions it states.
2. Quality - it opens with a short, specific headline; it tells one coherent story instead of
   a list of unrelated items; it is concise (1-2 short paragraphs), clear and neutral in
   style; it reads like finished news copy; and it contains no fabricated byline, publication
   name or hyperlinks, and no meta-commentary or placeholder text.

Score from 0 to 1: 1 means highly relevant and well written, 0 means off-topic or unusable.
If the article is empty, the score is 0. In your reasoning, give a short verdict for relevance
and for quality before the final score.

<input>
{inputs}
</input>

<sources>
{sources}
</sources>

<output>
{outputs}
</output>
"""

_article_judge = create_llm_as_judge(
    prompt=ARTICLE_QUALITY_PROMPT,
    judge=JUDGE_LLM,
    feedback_key="article_quality",
    continuous=True,    # Provider-native strict JSON schema: every field (score, reasoning) is guaranteed, unlike
    # the default function-calling mode where the model can omit "reasoning" (-> KeyError).
    structured_output_kwargs=STRICT_JSON_SCHEMA,
)


def article_quality(inputs: dict, outputs: dict) -> dict:
    """LLM-as-a-judge: relevance to the user profile and quality of the generated article."""
    article = (outputs.get("generated_article") or "").strip()
    if not article:
        return {"key": "article_quality", "score": None, "comment": "No article generated."}
    return _article_judge(inputs=inputs["user_profile"], outputs=article, sources=_sources(outputs))


def _sources(outputs: dict) -> str:
    """The material the writer was given (events for the graph, articles for the baseline)."""
    return "\n\n".join(f"[{i}] {s}" for i, s in enumerate(outputs.get("sources") or [])) or "(none)"


ARTICLE_FACTUALITY_PROMPT = """
You are a strict fact-checker. The sources are the events the writer was given; the output is
the news article it wrote from them. Check every factual claim in the article (who did what,
when, where, numbers, quotes, causes) against the sources. A claim is supported only if the
sources state or clearly imply it. Details that appear nowhere in the sources - invented
names, figures, dates, quotes, or background presented as fact - are unsupported, even if they
happen to be true. Wording that is vaguer than the sources is fine; wording that contradicts
them is a serious error.

Score the fraction of the article's factual claims that are supported by the sources, as a
number between 0 and 1, lowering it further for contradictions. If the sources are empty
("(none)"), any concrete claim is unsupported. In your reasoning, list the unsupported or
contradicted claims (or say there are none) before the final score.

<input>
{inputs}
</input>

<sources>
{sources}
</sources>

<output>
{outputs}
</output>
"""

_factuality_judge = create_llm_as_judge(
    prompt=ARTICLE_FACTUALITY_PROMPT,
    judge=JUDGE_LLM,
    feedback_key="article_factuality",
    continuous=True,
    structured_output_kwargs=STRICT_JSON_SCHEMA,
)


def article_factuality(inputs: dict, outputs: dict) -> dict:
    """LLM-as-a-judge: fraction of the article's claims supported by the events/articles it was written from."""
    article = (outputs.get("generated_article") or "").strip()
    if not article:
        return {"key": "article_factuality", "score": None, "comment": "No article generated."}
    return _factuality_judge(inputs=inputs["user_profile"], outputs=article, sources=_sources(outputs))
