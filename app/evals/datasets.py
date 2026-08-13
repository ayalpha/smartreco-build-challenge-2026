"""Golden evaluation cases for retrieval quality.

Cases are defined by **title matchers** rather than hard-coded product ids so
they stay valid when the sample catalog is re-seeded with new primary keys.
The runner resolves matchers against the live SQL catalog at eval time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    RetrievalCase(
        id="rrf-fusion-keywords",
        query="reciprocal rank fusion dense sparse retrieval",
        relevant_title_substrings=("Vector Databases", "Agentic RAG"),
        split="train",
        notes="RRF language appears in both vector DB and RAG course copy.",
        tags=("retrieval", "keyword"),
    ),
    RetrievalCase(
        id="blameless-postmortem",
        query="blameless postmortem action items for engineers",
        relevant_title_substrings=("Writing for Engineers", "Postmortems"),
        split="test",
        notes="Paraphrase of career writing course outcomes.",
        tags=("career", "paraphrase"),
    ),
    RetrievalCase(
        id="helm-rollouts",
        query="helm rollout strategies resource limits and probes",
        relevant_title_substrings=("Kubernetes",),
        split="test",
        notes="Ops paraphrase; should prefer Kubernetes over agentic courses.",
        tags=("devops", "paraphrase"),
    ),
    RetrievalCase(
        id="state-machines-python",
        query="python state machine agent orchestration with retries",
        relevant_title_substrings=("LangGraph",),
        split="train",
        notes="LangGraph description mentions state machines and refinement retries.",
        tags=("agentic", "keyword"),
    ),
    RetrievalCase(
        id="message-passing-agents",
        query="message passing between cooperating agents that plan and critique",
        relevant_title_substrings=("Multi-Agent",),
        split="test",
        notes="Near-copy of multi-agent course description.",
        tags=("agentic", "paraphrase"),
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
    exclude_case_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    tag: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    tag_any: Optional[Sequence[str]] = None,
    shuffle: bool = False,
    seed: int = 0,
) -> list[RetrievalCase]:
    """Apply split / id / tag / limit filters used by :class:`EvalParams`."""
    selected = list(cases)
    if split and split != "all":
        selected = [c for c in selected if c.split == split]
    if case_ids:
        allowed = set(case_ids)
        selected = [c for c in selected if c.id in allowed]
    if exclude_case_ids:
        blocked = set(exclude_case_ids)
        selected = [c for c in selected if c.id not in blocked]
    if tag:
        needle = tag.lower()
        selected = [c for c in selected if needle in {t.lower() for t in c.tags}]
    if tags:
        required = {t.lower() for t in tags}
        selected = [
            c
            for c in selected
            if required.issubset({t.lower() for t in c.tags})
        ]
    if tag_any:
        allowed_tags = {t.lower() for t in tag_any}
        selected = [
            c
            for c in selected
            if allowed_tags.intersection({t.lower() for t in c.tags})
        ]
    if shuffle:
        import random

        rng = random.Random(seed)
        rng.shuffle(selected)
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


@dataclass(frozen=True)
class ClassificationFixture:
    """Hand-labelled y_true / y_pred (or scores) for metric regression tests."""

    id: str
    y_true: tuple[Any, ...]
    y_pred: tuple[Any, ...] = ()
    scores: tuple[float, ...] = ()
    expected: dict[str, float] = field(default_factory=dict)
    notes: str = ""


#: Deterministic label fixtures with expected accuracy/precision/recall/F1.
GOLDEN_CLASSIFICATION_FIXTURES: tuple[ClassificationFixture, ...] = (
    ClassificationFixture(
        id="perfect-binary",
        y_true=(1, 1, 0, 0),
        y_pred=(1, 1, 0, 0),
        expected={"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        notes="All labels correct.",
    ),
    ClassificationFixture(
        id="balanced-errors",
        y_true=(1, 1, 0, 0),
        y_pred=(1, 0, 1, 0),
        expected={"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5},
        notes="TP=FP=TN=FN=1.",
    ),
    ClassificationFixture(
        id="precision-oriented",
        y_true=(1, 1, 0, 0, 0),
        y_pred=(1, 0, 0, 0, 0),
        expected={
            "accuracy": 0.8,
            "precision": 1.0,
            "recall": 0.5,
            "f1": 2.0 / 3.0,
        },
        notes="TP=1 FP=0 FN=1 TN=3.",
    ),
    ClassificationFixture(
        id="recall-oriented",
        y_true=(1, 1, 0, 0),
        y_pred=(1, 1, 1, 0),
        expected={
            "accuracy": 0.75,
            "precision": 2.0 / 3.0,
            "recall": 1.0,
            "f1": 0.8,
        },
        notes="TP=2 FP=1 FN=0 TN=1.",
    ),
    ClassificationFixture(
        id="score-threshold-default",
        y_true=(1, 1, 0, 0),
        scores=(0.9, 0.4, 0.6, 0.1),
        expected={"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5},
        notes="At threshold 0.5 → pred [1,0,1,0].",
    ),
    ClassificationFixture(
        id="all-negative-true",
        y_true=(0, 0, 0, 0),
        y_pred=(0, 0, 1, 0),
        expected={
            "accuracy": 0.75,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
        notes="No true positives possible; FP=1 TN=3.",
    ),
    ClassificationFixture(
        id="high-score-confidence",
        y_true=(1, 1, 1, 0, 0),
        scores=(0.95, 0.91, 0.88, 0.12, 0.05),
        expected={
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
        notes="Well-separated scores at default 0.5 threshold.",
    ),
)


#: Graded-item fixtures for thresholded relevance scoring (grader eval path).
#: ``relevant`` is ground truth; ``relevance_score`` mimics judge/heuristic output.
GOLDEN_GRADER_SCORE_FIXTURES: tuple[dict[str, Any], ...] = (
    {
        "id": "clear-separation",
        "items": (
            {"relevance_score": 0.9, "relevant": True},
            {"relevance_score": 0.85, "relevant": True},
            {"relevance_score": 0.2, "relevant": False},
            {"relevance_score": 0.1, "relevant": False},
        ),
        "threshold": 0.35,
        "expected": {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
    },
    {
        "id": "borderline-heuristic",
        # Agent heuristic threshold is 0.35 — one true positive sits just above.
        "items": (
            {"relevance_score": 0.4, "relevant": True},
            {"relevance_score": 0.3, "relevant": True},  # FN at 0.35
            {"relevance_score": 0.36, "relevant": False},  # FP at 0.35
            {"relevance_score": 0.1, "relevant": False},
        ),
        "threshold": 0.35,
        # pred at 0.35: [1, 0, 1, 0]; true [1,1,0,0] → TP1 FP1 FN1 TN1
        "expected": {
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        },
    },
    {
        "id": "strict-half",
        "items": (
            {"relevance_score": 0.7, "relevant": True},
            {"relevance_score": 0.55, "relevant": True},
            {"relevance_score": 0.45, "relevant": False},
            {"relevance_score": 0.2, "relevant": False},
        ),
        "threshold": 0.5,
        "expected": {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
    },
)


#: Multi-class fixtures: expected keys use ``accuracy`` plus micro/macro F1.
GOLDEN_MULTICLASS_FIXTURES: tuple[dict[str, Any], ...] = (
    {
        "id": "three-class-hand-checked",
        "y_true": ["A", "A", "B", "B", "C"],
        "y_pred": ["A", "B", "A", "B", "C"],
        # correct 3/5 → accuracy 0.6; micro P=R=F1=0.6; macro F1 = 2/3
        "expected": {
            "accuracy": 0.6,
            "micro_f1": 0.6,
            "macro_f1": 2.0 / 3.0,
            "weighted_f1": 0.6,
        },
    },
    {
        "id": "perfect-multiclass",
        "y_true": ["x", "y", "z", "x"],
        "y_pred": ["x", "y", "z", "x"],
        "expected": {
            "accuracy": 1.0,
            "micro_f1": 1.0,
            "macro_f1": 1.0,
            "weighted_f1": 1.0,
        },
    },
)
