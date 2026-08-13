"""Golden evaluation cases for retrieval quality.

Cases are defined by **title matchers** rather than hard-coded product ids so
they stay valid when the sample catalog is re-seeded with new primary keys.
The runner resolves matchers against the live SQL catalog at eval time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

from sqlalchemy.orm import Session

from app.models.product import Product

SplitName = Literal["train", "test"]


@dataclass(frozen=True)
class RetrievalCase:
    """One labelled retrieval query.

    Attributes:
        id: Stable case id (used by ``EvalParams.case_ids``).
        query: Free-text retrieval query.
        relevant_title_substrings: Case-insensitive substrings; a product is
            relevant if **any** substring appears in its title.
        relevant_categories: Optional category allow-list (union with titles).
        split: ``train`` / ``test`` for optional holdout-style filtering.
        notes: Free-form description of the judgement.
        tags: Extra labels for filtering (e.g. ``agentic``, ``devops``).
    """

    id: str
    query: str
    relevant_title_substrings: tuple[str, ...] = ()
    relevant_categories: tuple[str, ...] = ()
    split: SplitName = "test"
    notes: str = ""
    tags: tuple[str, ...] = ()

    def resolve_relevant_ids(self, products: Sequence[Product]) -> list[int]:
        """Map title/category judgements onto concrete product ids."""
        relevant: list[int] = []
        title_needles = tuple(s.lower() for s in self.relevant_title_substrings)
        categories = {c.lower() for c in self.relevant_categories}
        for product in products:
            title = (product.title or "").lower()
            category = (product.category or "").lower()
            title_hit = any(needle in title for needle in title_needles) if title_needles else False
            category_hit = category in categories if categories else False
            if title_hit or category_hit:
                relevant.append(product.id)
        return relevant


#: Golden set aligned with ``tests.conftest.SAMPLE_PRODUCTS``.
#: train = simple keyword-heavy queries; test = paraphrases / multi-intent.
GOLDEN_RETRIEVAL_CASES: tuple[RetrievalCase, ...] = (
    RetrievalCase(
        id="langgraph-agents",
        query="building agents as state machines with langgraph",
        relevant_title_substrings=("LangGraph",),
        split="train",
        notes="Direct title/tag overlap with the LangGraph course.",
        tags=("agentic", "keyword"),
    ),
    RetrievalCase(
        id="agentic-rag",
        query="grade retrieved documents rewrite queries hybrid dense keyword",
        relevant_title_substrings=("Agentic RAG", "Retrieval That Reasons"),
        split="train",
        notes="RAG-specific vocabulary should surface the RAG course.",
        tags=("agentic", "rag"),
    ),
    RetrievalCase(
        id="multi-agent",
        query="supervisor architectures hierarchical multi-agent coordination",
        relevant_title_substrings=("Multi-Agent",),
        split="train",
        notes="Multi-agent orchestration query.",
        tags=("agentic",),
    ),
    RetrievalCase(
        id="vector-db",
        query="vector database hybrid search reciprocal rank fusion qdrant",
        relevant_title_substrings=("Vector Databases",),
        split="train",
        notes="Dense+sparse vector infra course.",
        tags=("data", "retrieval"),
    ),
    RetrievalCase(
        id="kubernetes",
        query="kubernetes helm deployments probes autoscaling",
        relevant_title_substrings=("Kubernetes",),
        split="train",
        notes="Ops distractor — should not rank agentic courses first.",
        tags=("devops",),
    ),
    RetrievalCase(
        id="tech-writing",
        query="writing design docs and blameless postmortems for engineers",
        relevant_title_substrings=("Writing for Engineers", "Design Docs"),
        split="train",
        notes="Career-skills writing course.",
        tags=("career",),
    ),
    # Paraphrased / harder holdout-style queries
    RetrievalCase(
        id="agent-orchestration-paraphrase",
        query="how do I wire conditional edges and checkpointing in an agent graph",
        relevant_title_substrings=("LangGraph",),
        relevant_categories=("Agentic AI",),
        split="test",
        notes="Paraphrase without the brand name; category still agentic.",
        tags=("agentic", "paraphrase"),
    ),
    RetrievalCase(
        id="semantic-search-paraphrase",
        query="approximate nearest neighbour indexes with metadata filters",
        relevant_title_substrings=("Vector Databases",),
        split="test",
        notes="Paraphrase of vector DB description.",
        tags=("data", "paraphrase"),
    ),
    RetrievalCase(
        id="agentic-category",
        query="production ready agentic systems with refinement loops",
        relevant_categories=("Agentic AI",),
        split="test",
        notes="Any Agentic AI course is relevant.",
        tags=("agentic", "category"),
    ),
)


def load_products(db: Session) -> list[Product]:
    """Active catalog rows ordered by id."""
    from sqlalchemy import select

    return list(
        db.scalars(select(Product).where(Product.is_active.is_(True)).order_by(Product.id))
    )


def filter_cases(
    cases: Sequence[RetrievalCase],
    *,
    split: str = "all",
    case_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    tag: Optional[str] = None,
) -> list[RetrievalCase]:
    """Apply split / id / tag / limit filters used by :class:`EvalParams`."""
    selected = list(cases)
    if split and split != "all":
        selected = [c for c in selected if c.split == split]
    if case_ids:
        allowed = set(case_ids)
        selected = [c for c in selected if c.id in allowed]
    if tag:
        needle = tag.lower()
        selected = [c for c in selected if needle in {t.lower() for t in c.tags}]
    if limit is not None:
        selected = selected[:limit]
    return selected


def resolve_cases(
    db: Session,
    cases: Sequence[RetrievalCase],
) -> list[dict[str, Any]]:
    """Attach resolved relevant product ids to each case.

    Drops cases whose judgement set is empty against the current catalog
    (avoids undefined recall when the seed catalog changes).
    """
    products = load_products(db)
    resolved: list[dict[str, Any]] = []
    for case in cases:
        relevant_ids = case.resolve_relevant_ids(products)
        if not relevant_ids:
            continue
        resolved.append(
            {
                "id": case.id,
                "query": case.query,
                "relevant_ids": relevant_ids,
                "split": case.split,
                "tags": list(case.tags),
                "notes": case.notes,
            }
        )
    return resolved
