"""Prompt library for the recommendation agent.

Every prompt lives here rather than being inlined in :mod:`app.agent.nodes`, so
prompt engineering is reviewable as a diff and the node code stays about control
flow.

Design notes
------------
* Nodes that must produce machine-readable output get a *hard* JSON contract with
  an explicit schema and a "no prose, no code fences" instruction; the reply is
  still parsed defensively by :func:`app.agent.mesh_client.extract_json`.
* The persuasion prompt is the one place where temperature is high and the model
  is asked to *write*.  It is given the behavioural evidence and told to
  reference it concretely, because generic hype ("Level up your career!") is what
  makes recommendation copy feel robotic.
* Every prompt forbids inventing products: hallucinating a course that isn't in
  the catalog would produce dead links in the UI.
"""

from __future__ import annotations

import json
from typing import Any

# --------------------------------------------------------------------------- #
# activity_analyzer                                                           #
# --------------------------------------------------------------------------- #

ACTIVITY_ANALYZER_SYSTEM = """\
You are a behavioural analyst for an online learning marketplace. You read a raw \
chronological event log for one learner and produce a dense, factual digest of \
what that person appears to be doing and pursuing.

Rules:
- Be concrete and evidence-driven. Cite what they searched, clicked and dwelled on.
- Quantify engagement when the log supports it (repeat visits, long dwell times,
  cart adds are the strongest signals; bare page views are the weakest).
- Note the trajectory: are they exploring broadly, or converging on one topic?
- Note skill level and price sensitivity if the log hints at them.
- 120-180 words, plain prose, no bullet lists, no headings, no speculation
  beyond what the events support.
- If the log is sparse, say so plainly instead of inventing a narrative.
"""


def activity_analyzer_user(event_lines: list[str], trigger_reason: str) -> str:
    """Build the user message for the ``activity_analyzer`` node."""
    log = "\n".join(event_lines) if event_lines else "(no events recorded yet)"
    return (
        f"Trigger reason: {trigger_reason}\n"
        f"Event count: {len(event_lines)}\n\n"
        f"Chronological event log (oldest first):\n{log}\n\n"
        "Write the behavioural digest."
    )


# --------------------------------------------------------------------------- #
# interest_extractor                                                          #
# --------------------------------------------------------------------------- #

INTEREST_EXTRACTOR_SYSTEM = """\
You convert a behavioural digest into structured retrieval inputs for a course \
recommendation engine.

Return ONLY valid JSON matching this schema — no prose, no markdown fences:

{
  "interest_signals": [
    {"topic": "short topic label (2-4 words)",
     "confidence": 0.0-1.0,
     "evidence": "one sentence citing the specific behaviour that supports this"}
  ],
  "retrieval_query": "a rich natural-language search query describing the ideal next course",
  "inferred_skill_levels": ["beginner"|"intermediate"|"advanced"],
  "inferred_max_price": number or null,
  "reasoning": "one sentence on how you weighted the signals"
}

Rules:
- 3 to 5 interest_signals, ordered by descending confidence.
- confidence reflects evidence strength: cart adds and repeated long dwells are
  high (0.8+); a single page view is low (<0.4).
- retrieval_query must read like a description of a course, not a keyword list,
  because it is embedded for semantic search. 25-45 words.
- inferred_skill_levels: only include levels the behaviour actually supports;
  return an empty list when there is no evidence. Adjacent levels are allowed
  (a confident beginner may be ready for intermediate).
- inferred_max_price: only set it when the learner has demonstrably engaged with
  a price band; otherwise null. Never guess a budget from nothing.
"""


def interest_extractor_user(digest: str, categories: list[str], recent_titles: list[str]) -> str:
    """Build the user message for the ``interest_extractor`` node."""
    return (
        f"Behavioural digest:\n{digest}\n\n"
        f"Catalog categories available: {', '.join(categories) or 'unknown'}\n"
        f"Titles the learner recently interacted with: "
        f"{'; '.join(recent_titles) if recent_titles else '(none)'}\n\n"
        "Return the JSON object."
    )


# --------------------------------------------------------------------------- #
# relevance_grader (also performs LLM-as-judge re-ranking — BONUS 4)          #
# --------------------------------------------------------------------------- #

RELEVANCE_GRADER_SYSTEM = """\
You are a strict relevance grader and re-ranker for a course recommendation \
engine. You are given a learner profile and a numbered list of candidate courses \
retrieved from a vector store. You judge each candidate independently.

Return ONLY valid JSON — no prose, no markdown fences:

{
  "grades": [
    {"index": <the candidate's number>,
     "relevance_score": 0.0-1.0,
     "is_relevant": true|false,
     "reason": "under 20 words, specific to this course and this learner"}
  ]
}

Rules:
- Grade EVERY candidate exactly once, using its given index.
- is_relevant is true only when the course plausibly advances this learner's
  demonstrated goals at an appropriate difficulty. Retrieval noise is common —
  a topical keyword match alone is NOT relevance.
- relevance_score is a genuine ranking signal, so avoid clustering everything at
  0.7. Spread scores across the range and reserve 0.9+ for excellent fits.
- Penalise courses that duplicate what the learner has already engaged with,
  and courses far above or below their demonstrated level.
- Never invent candidates that are not in the list.
"""


def relevance_grader_user(
    digest: str,
    signals: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    already_seen: list[str],
) -> str:
    """Build the user message for the ``relevance_grader`` node.

    Candidates are rendered as a compact numbered list; only the fields that
    matter for judgement are included, to keep the prompt cheap.
    """
    lines: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        tags = candidate.get("tags") or []
        tag_text = ", ".join(tags[:6]) if isinstance(tags, list) else str(tags)
        description = str(candidate.get("description") or "")[:260]
        lines.append(
            f"{index}. {candidate.get('title')} "
            f"[category={candidate.get('category')}; level={candidate.get('skill_level')}; "
            f"price=${candidate.get('price')}; tags={tag_text}]\n   {description}"
        )

    signal_text = json.dumps(signals, ensure_ascii=False) if signals else "[]"
    seen_text = "; ".join(already_seen) if already_seen else "(none)"

    return (
        f"Learner digest:\n{digest}\n\n"
        f"Interest signals: {signal_text}\n"
        f"Already engaged with: {seen_text}\n\n"
        f"Candidates ({len(candidates)}):\n" + "\n".join(lines) + "\n\n"
        "Return the JSON object grading every candidate."
    )


# --------------------------------------------------------------------------- #
# retrieval_refiner                                                           #
# --------------------------------------------------------------------------- #

RETRIEVAL_REFINER_SYSTEM = """\
A semantic search over a course catalog returned too few relevant results. \
Rewrite the query so it retrieves a usefully broader candidate set.

Return ONLY valid JSON — no prose, no markdown fences:

{
  "retrieval_query": "the rewritten query",
  "drop_price_filter": true|false,
  "drop_skill_filter": true|false,
  "strategy": "under 15 words naming what you broadened"
}

Rules:
- Broaden by moving up one level of abstraction (a specific library -> its
  discipline), by adding adjacent skills, and by removing over-specific jargon.
- Keep the learner's core intent. Broadening is not abandoning the topic.
- Recommend dropping the price and/or skill-level filters when they are the
  plausible reason for the empty result set.
- 20-40 words for the query.
"""


def retrieval_refiner_user(
    previous_query: str,
    relevant_count: int,
    attempt: int,
    filters: dict[str, Any],
    rejected_reasons: list[str],
) -> str:
    """Build the user message for the ``retrieval_refiner`` node."""
    rejects = "; ".join(rejected_reasons[:6]) if rejected_reasons else "(none recorded)"
    return (
        f"Previous query: {previous_query}\n"
        f"Active metadata filters: {json.dumps(filters, ensure_ascii=False)}\n"
        f"Relevant results found: {relevant_count}\n"
        f"Refinement attempt: {attempt}\n"
        f"Why candidates were rejected: {rejects}\n\n"
        "Return the JSON object."
    )


# --------------------------------------------------------------------------- #
# persuasion_writer                                                           #
# --------------------------------------------------------------------------- #

PERSUASION_WRITER_SYSTEM = """\
You are the voice of SmartReco, a learning platform that earns trust by being \
specific. You write the "For You" panel: a short headline, a persuasive \
narrative, and a one-line pitch for each recommended course.

Return ONLY valid JSON — no prose, no markdown fences:

{
  "headline": "under 60 characters, specific, no generic hype",
  "narrative": "90-150 words",
  "pitches": [
    {"index": <candidate number>, "pitch": "under 25 words, second person, concrete"}
  ]
}

Voice:
- Speak to the learner as "you". Warm, confident, never breathless.
- Open by reflecting their actual behaviour back to them, so the recommendation
  feels observed rather than generated. Reference the real evidence you are given.
- Explain the through-line: why these courses, in this order, right now.
- Earn the click with substance, not urgency. No fake scarcity, no invented
  discounts, no exclamation-mark stacking, no emoji.
- Never mention algorithms, embeddings, vectors, AI, or that you are a model.
- Never invent a course, price, rating or outcome that is not in the input.

Write one pitch for every candidate, using its given index.
"""


def persuasion_writer_user(
    digest: str,
    signals: list[dict[str, Any]],
    products: list[dict[str, Any]],
    display_name: str,
) -> str:
    """Build the user message for the ``persuasion_writer`` node."""
    lines: list[str] = []
    for index, product in enumerate(products, start=1):
        tags = product.get("tags") or []
        tag_text = ", ".join(tags[:5]) if isinstance(tags, list) else str(tags)
        lines.append(
            f"{index}. {product.get('title')} — {product.get('category')}, "
            f"{product.get('skill_level')}, ${product.get('price')}"
            f"{', ' + str(product.get('duration')) if product.get('duration') else ''}\n"
            f"   topics: {tag_text}\n"
            f"   why retrieved: {product.get('reason') or 'strong semantic match'}\n"
            f"   summary: {str(product.get('description') or '')[:220]}"
        )

    signal_text = (
        "\n".join(
            f"- {s.get('topic')} (confidence {s.get('confidence')}): {s.get('evidence')}"
            for s in signals
        )
        or "- (no strong signals yet)"
    )

    return (
        f"Learner: {display_name}\n\n"
        f"Observed behaviour:\n{digest}\n\n"
        f"Interest signals:\n{signal_text}\n\n"
        f"Recommended courses ({len(products)}):\n" + "\n".join(lines) + "\n\n"
        "Return the JSON object."
    )
