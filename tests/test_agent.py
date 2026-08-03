"""Tests for the LangGraph recommendation agent.

These exercise the *real* compiled graph — no mocked nodes — with Mesh absent, so
every graceful-degradation path is covered and the whole pipeline is verified end
to end without network access or credentials.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.graph import build_graph, get_graph, render_mermaid, route_after_grading
from app.agent.mesh_client import MeshTelemetry, extract_json
from app.agent.nodes import (
    NODE_PERSUASION_WRITER,
    NODE_RETRIEVAL_REFINER,
    activity_analyzer,
    interest_extractor,
    relevance_grader,
    retrieval_node,
)
from app.agent.observability import build_run_config, tracing_enabled
from app.agent.runner import maybe_dispatch, run_agent, run_agent_now
from app.agent.state import make_initial_state, summarise_state
from app.agent.triggers import (
    REASON_EVENT_THRESHOLD,
    REASON_FIRST_TIME,
    REASON_NONE,
    evaluate,
)
from app.config import get_settings
from app.models.product import Product
from app.models.recommendation import Recommendation
from app.models.user import User
from app.vector_store.bm25 import BM25Document, build_index
from app.vector_store.embeddings import (
    FALLBACK_MODEL_NAME,
    cosine_similarity,
    embed_documents,
    hashing_embedding,
)
from app.vector_store.qdrant_client import RRF_K, SearchFilters
from app.vector_store.sync import hybrid_retrieve, keyword_search

settings = get_settings()

EXPECTED_NODES = {
    "activity_analyzer",
    "interest_extractor",
    "retrieval_node",
    "relevance_grader",
    "retrieval_refiner",
    "persuasion_writer",
    "recommendation_storer",
}


# --------------------------------------------------------------------------- #
# Graph structure                                                             #
# --------------------------------------------------------------------------- #

class TestGraphStructure:
    """The graph must have exactly the seven named nodes and compile cleanly."""

    def test_all_seven_nodes_registered(self) -> None:
        graph = build_graph()
        assert EXPECTED_NODES.issubset(set(graph.nodes)), (
            f"missing nodes: {EXPECTED_NODES - set(graph.nodes)}"
        )

    def test_graph_compiles_with_checkpointer(self) -> None:
        compiled = get_graph()
        assert compiled is not None
        # A checkpointer is what makes runs replayable/debuggable.
        assert getattr(compiled, "checkpointer", None) is not None

    def test_mermaid_rendering_includes_conditional_edge(self) -> None:
        diagram = render_mermaid()
        assert "relevance_grader" in diagram
        assert "retrieval_refiner" in diagram
        assert "persuasion_writer" in diagram


class TestConditionalRouting:
    """The single conditional edge has three distinct outcomes."""

    def _state(self, relevant: int, retries: int) -> dict[str, Any]:
        graded = [
            {"product": {"id": index}, "relevance_score": 0.9, "is_relevant": True}
            for index in range(relevant)
        ]
        graded.append({"product": {"id": 99}, "relevance_score": 0.1, "is_relevant": False})
        return {"graded_products": graded, "retry_count": retries}

    def test_enough_relevant_goes_to_writer(self) -> None:
        state = self._state(relevant=settings.agent_min_relevant_products, retries=0)
        assert route_after_grading(state) == NODE_PERSUASION_WRITER  # type: ignore[arg-type]

    def test_too_few_with_budget_goes_to_refiner(self) -> None:
        state = self._state(relevant=1, retries=0)
        assert route_after_grading(state) == NODE_RETRIEVAL_REFINER  # type: ignore[arg-type]

    def test_exhausted_budget_writes_anyway(self) -> None:
        state = self._state(relevant=0, retries=settings.agent_max_retrieval_retries)
        assert route_after_grading(state) == NODE_PERSUASION_WRITER  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Individual nodes                                                            #
# --------------------------------------------------------------------------- #

class TestNodes:
    """Each node degrades rather than raising when Mesh is unavailable."""

    def test_activity_analyzer_produces_digest(
        self, user: User, products: list[Product], make_events: Any
    ) -> None:
        make_events(user, products)
        state = make_initial_state(user.id, trigger_reason="manual")

        update = activity_analyzer(state)

        assert update["behavior_digest"], "digest must never be empty"
        assert len(update["raw_events"]) > 0
        assert update["seen_product_ids"]
        # No Mesh key configured, so the heuristic path must have been used.
        assert update["degraded"] is True
        assert update["node_log"][0]["node"] == "activity_analyzer"

    def test_activity_analyzer_handles_cold_start(self, user: User) -> None:
        update = activity_analyzer(make_initial_state(user.id))
        assert update["raw_events"] == []
        assert "no recorded activity" in update["behavior_digest"].lower()

    def test_interest_extractor_weights_cart_adds_highest(
        self, user: User, products: list[Product], make_events: Any
    ) -> None:
        make_events(user, products, include_cart=True)
        state = make_initial_state(user.id)
        state.update(activity_analyzer(state))  # type: ignore[typeddict-item]

        update = interest_extractor(state)
        signals = update["interest_signals"]

        assert 1 <= len(signals) <= 5
        assert all(0.0 <= signal["confidence"] <= 1.0 for signal in signals)
        # Signals must be ordered by descending confidence.
        confidences = [signal["confidence"] for signal in signals]
        assert confidences == sorted(confidences, reverse=True)
        assert update["retrieval_query"]
        # The dominant interest is agentic AI, which the fixture over-weights.
        assert any("agentic" in signal["topic"].lower() for signal in signals)

    def test_interest_extractor_infers_price_ceiling(
        self, user: User, products: list[Product], make_events: Any
    ) -> None:
        make_events(user, products)
        state = make_initial_state(user.id)
        state.update(activity_analyzer(state))  # type: ignore[typeddict-item]

        filters = interest_extractor(state)["retrieval_filters"]

        assert "max_price" in filters
        assert filters["max_price"] > 0
        assert "skill_levels" in filters

    def test_retrieval_node_returns_candidates(
        self, user: User, products: list[Product]
    ) -> None:
        state = make_initial_state(user.id)
        state["retrieval_query"] = "building agents as state machines with langgraph"

        update = retrieval_node(state)
        retrieved = update["retrieved_products"]

        assert retrieved, "retrieval must always return candidates"
        assert all("id" in item and "title" in item for item in retrieved)
        assert all("fused_score" in item for item in retrieved)

    def test_retrieval_node_falls_back_when_filters_are_impossible(
        self, user: User, products: list[Product]
    ) -> None:
        state = make_initial_state(user.id)
        state["retrieval_query"] = "langgraph agents"
        # A price ceiling no course can satisfy.
        state["retrieval_filters"] = {"max_price": 0.01}

        update = retrieval_node(state)

        assert update["retrieved_products"], "must fall back rather than return nothing"

    def test_relevance_grader_grades_every_candidate(
        self, user: User, products: list[Product], make_events: Any
    ) -> None:
        make_events(user, products)
        state = make_initial_state(user.id)
        state.update(activity_analyzer(state))  # type: ignore[typeddict-item]
        state.update(interest_extractor(state))  # type: ignore[typeddict-item]
        state.update(retrieval_node(state))  # type: ignore[typeddict-item]

        update = relevance_grader(state)
        graded = update["graded_products"]

        assert len(graded) == len(state["retrieved_products"])
        assert all(0.0 <= item["relevance_score"] <= 1.0 for item in graded)
        assert all("rerank_score" in item for item in graded)
        # Re-ranking must place relevant items before irrelevant ones.
        flags = [bool(item["is_relevant"]) for item in graded]
        assert flags == sorted(flags, reverse=True)


# --------------------------------------------------------------------------- #
# Full graph execution                                                        #
# --------------------------------------------------------------------------- #

class TestFullGraphRun:
    """End-to-end execution of the compiled graph."""

    def test_run_stores_a_recommendation(
        self, db: Session, user: User, products: list[Product], make_events: Any
    ) -> None:
        make_events(user, products, count=12)

        result = run_agent(user.id, reason="manual", event_count=13)

        assert result.ok, f"agent run failed: {result.error}"
        assert result.recommendation_id is not None
        assert result.duration_ms > 0

        record = db.get(Recommendation, result.recommendation_id)
        assert record is not None
        assert record.user_id == user.id
        assert record.is_active is True
        assert record.narrative, "narrative must never be empty"
        assert record.headline
        assert 1 <= len(record.products) <= settings.agent_final_product_count
        assert record.interest_signals
        assert record.trigger_reason == "manual"
        # Mesh is unconfigured in tests, so the run must be flagged degraded.
        assert record.degraded is True

    def test_products_carry_pitches_and_provenance(
        self, db: Session, user: User, products: list[Product], make_events: Any
    ) -> None:
        make_events(user, products)
        result = run_agent(user.id, reason="manual")
        record = db.get(Recommendation, result.recommendation_id)

        assert record is not None
        for product in record.products:
            assert product["id"] is not None
            assert product["title"]
            assert "pitch" in product
            assert product["retrieval_mode"] in {
                "dense", "keyword", "hybrid", "catalog_fallback",
            }

    def test_trace_records_every_node(
        self, db: Session, user: User, products: list[Product], make_events: Any
    ) -> None:
        make_events(user, products)
        result = run_agent(user.id, reason="manual")
        record = db.get(Recommendation, result.recommendation_id)

        assert record is not None
        visited = {entry["node"] for entry in record.agent_trace["node_log"]}
        # Every node except the refiner runs on a healthy path.
        assert EXPECTED_NODES - {"retrieval_refiner"} <= visited
        assert record.agent_trace["retrieval_query"]
        assert record.latency_ms and record.latency_ms > 0

    def test_second_run_deactivates_the_first(
        self, db: Session, user: User, products: list[Product], make_events: Any
    ) -> None:
        make_events(user, products)
        first = run_agent(user.id, reason="manual")
        second = run_agent(user.id, reason="manual")

        assert first.recommendation_id != second.recommendation_id

        active = list(
            db.scalars(
                select(Recommendation).where(
                    Recommendation.user_id == user.id,
                    Recommendation.is_active.is_(True),
                )
            )
        )
        assert len(active) == 1, "exactly one recommendation may be active per user"
        assert active[0].id == second.recommendation_id

    def test_run_with_no_events_still_succeeds(
        self, db: Session, user: User, products: list[Product]
    ) -> None:
        """Cold start: no behaviour at all must not crash the graph."""
        result = run_agent(user.id, reason="first_time")

        assert result.ok
        record = db.get(Recommendation, result.recommendation_id)
        assert record is not None
        assert record.narrative

    def test_concurrent_run_is_skipped_by_the_lock(
        self, user: User, products: list[Product], make_events: Any
    ) -> None:
        from app.cache import acquire_agent_lock, release_agent_lock

        make_events(user, products)
        assert acquire_agent_lock(user.id) is True
        try:
            result = run_agent(user.id, reason="manual", respect_lock=True)
            assert result.skipped is True
            assert result.ok is False
        finally:
            release_agent_lock(user.id)

    def test_refinement_loop_runs_then_gives_up_gracefully(
        self, db: Session, user: User, products: list[Product],
        make_events: Any, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The conditional edge's retry path: refine, re-retrieve, then write anyway.

        The relevance threshold is raised beyond what the catalog can satisfy, which
        forces ``relevance_grader -> retrieval_refiner -> retrieval_node`` to loop
        until the retry budget is exhausted — at which point the graph must still
        produce a recommendation rather than returning nothing.
        """
        make_events(user, products)
        monkeypatch.setattr(settings, "agent_min_relevant_products", 99, raising=False)

        result = run_agent(user.id, reason="manual")

        assert result.ok, "an exhausted retry budget must still produce output"
        record = db.get(Recommendation, result.recommendation_id)
        assert record is not None

        trace = record.agent_trace
        assert trace["retry_count"] == settings.agent_max_retrieval_retries
        assert len(trace["refinement_history"]) == settings.agent_max_retrieval_retries

        visited = [entry["node"] for entry in trace["node_log"]]
        assert visited.count("retrieval_refiner") == settings.agent_max_retrieval_retries
        # Retrieval re-runs once per refinement, plus the initial pass.
        assert visited.count("retrieval_node") == settings.agent_max_retrieval_retries + 1
        assert visited[-1] == "recommendation_storer"
        assert record.narrative, "the writer must run even on the degraded path"

    def test_state_summary_is_json_safe(self, user: User) -> None:
        state = make_initial_state(user.id, trigger_reason="manual")
        state["telemetry"] = MeshTelemetry()
        summary = summarise_state(state)

        import json

        json.dumps(summary)  # must not raise
        assert summary["user_id"] == user.id
        assert "mesh" in summary


# --------------------------------------------------------------------------- #
# Trigger policy                                                              #
# --------------------------------------------------------------------------- #

class TestTriggerPolicy:
    """The three trigger rules, plus the idle case."""

    def test_no_events_means_no_run(self, db: Session, user: User) -> None:
        decision = evaluate(db, user.id)
        assert decision.should_run is False
        assert decision.reason == REASON_NONE

    def test_first_recommendation_triggers(
        self, db: Session, user: User, products: list[Product], make_events: Any
    ) -> None:
        make_events(user, products, count=3)
        decision = evaluate(db, user.id)
        assert decision.should_run is True
        assert decision.reason == REASON_FIRST_TIME

    def test_event_threshold_triggers_after_interval(
        self, db: Session, user: User, products: list[Product], make_events: Any
    ) -> None:
        make_events(user, products, count=2)
        run_agent(user.id, reason="manual")

        # `start=now` so these land *after* the recommendation that was just
        # written; the default backdated timestamps would not count as new.
        now = datetime.now(timezone.utc)
        make_events(user, products, count=2, start=now)
        assert evaluate(db, user.id).should_run is False, "4 new events is below the threshold"

        # Cross the threshold.
        make_events(user, products, count=settings.agent_event_trigger_interval, start=now)
        decision = evaluate(db, user.id)
        assert decision.should_run is True
        assert decision.reason == REASON_EVENT_THRESHOLD

    def test_stale_recommendation_triggers(
        self, db: Session, user: User, products: list[Product], make_events: Any
    ) -> None:
        make_events(user, products, count=2)
        result = run_agent(user.id, reason="manual")

        record = db.get(Recommendation, result.recommendation_id)
        assert record is not None
        record.created_at = datetime.now(timezone.utc) - timedelta(
            hours=settings.agent_stale_hours + 1
        )
        db.commit()

        # After backdating, the existing events already count as "new". Add a few
        # more so the total sits above the stale minimum but below the event
        # threshold — otherwise the higher-priority threshold rule would fire.
        make_events(user, products, count=2)
        decision = evaluate(db, user.id)

        assert settings.agent_stale_min_events <= decision.new_events
        assert decision.new_events < settings.agent_event_trigger_interval
        assert decision.should_run is True
        assert decision.reason == "stale"

    def test_dispatch_is_idempotent_under_a_held_lock(
        self, user: User, products: list[Product], make_events: Any
    ) -> None:
        from app.cache import acquire_agent_lock, release_agent_lock

        make_events(user, products, count=3)
        assert acquire_agent_lock(user.id) is True
        try:
            decision = maybe_dispatch(user.id)
            assert decision.should_run is False
            assert "already in progress" in decision.detail
        finally:
            release_agent_lock(user.id)


# --------------------------------------------------------------------------- #
# Retrieval internals (BONUS 4)                                               #
# --------------------------------------------------------------------------- #

class TestRetrieval:
    """Hybrid retrieval, BM25 and metadata filtering."""

    def test_bm25_ranks_the_obvious_match_first(self) -> None:
        index = build_index(
            [
                BM25Document(1, "langgraph agents state machines python orchestration"),
                BM25Document(2, "kubernetes helm scaling deployments probes"),
                BM25Document(3, "writing design docs postmortems communication"),
            ]
        )
        ranked = index.search("langgraph agent orchestration", limit=3)

        assert ranked, "BM25 must return matches"
        assert ranked[0][0] == 1
        assert ranked[0][1] > 0

    def test_bm25_returns_nothing_for_unknown_terms(self) -> None:
        index = build_index([BM25Document(1, "kubernetes helm scaling")])
        assert index.search("zzzznonexistentterm") == []

    def test_keyword_search_finds_the_right_course(
        self, db: Session, products: list[Product]
    ) -> None:
        hits = keyword_search(db, "langgraph state machines", limit=5)
        assert hits
        top_id = hits[0][0]
        top = db.get(Product, top_id)
        assert top is not None
        assert "LangGraph" in top.title

    def test_hybrid_retrieval_prefers_the_relevant_category(
        self, db: Session, products: list[Product]
    ) -> None:
        hits = hybrid_retrieve(
            db, "building agentic systems with langgraph and retrieval", limit=4
        )
        assert hits
        categories = [hit.payload.get("category") for hit in hits[:3]]
        assert "Agentic AI" in categories

    def test_rrf_scores_are_bounded_and_descending(
        self, db: Session, products: list[Product]
    ) -> None:
        hits = hybrid_retrieve(db, "vector database hybrid search", limit=5)
        scores = [hit.fused_score for hit in hits]

        assert scores == sorted(scores, reverse=True)
        # A single ranking contributes at most 1/(RRF_K + 1); two rankings, twice that.
        assert all(0 < score <= 2 * (1.0 / (RRF_K + 1)) + 1e-9 for score in scores)

    def test_metadata_filter_excludes_products(
        self, db: Session, products: list[Product]
    ) -> None:
        excluded = products[0].id
        hits = hybrid_retrieve(
            db,
            "langgraph agents",
            limit=6,
            filters=SearchFilters(exclude_product_ids=[excluded]),
        )
        assert all(hit.product_id != excluded for hit in hits)

    def test_skill_level_filter_is_respected(
        self, db: Session, products: list[Product]
    ) -> None:
        hits = hybrid_retrieve(
            db, "agents", limit=6, filters=SearchFilters(skill_levels=["advanced"])
        )
        assert hits
        assert all(
            str(hit.payload.get("skill_level")).lower() == "advanced" for hit in hits
        )


class TestEmbeddings:
    """The offline embedding fallback must be deterministic and well-formed."""

    def test_fallback_is_used_without_a_mesh_key(self) -> None:
        result = embed_documents(["langgraph agents"])
        assert result.degraded is True
        assert result.model == FALLBACK_MODEL_NAME
        assert result.dimension == settings.mesh_embedding_dim
        assert len(result.vectors[0]) == settings.mesh_embedding_dim

    def test_provenance_is_stamped_for_auditability(self) -> None:
        provenance = embed_documents(["anything"]).provenance
        assert provenance["embedding_model"] == FALLBACK_MODEL_NAME
        assert provenance["embedding_degraded"] is True
        assert provenance["embedding_pipeline_version"]

    def test_hashing_embedding_is_deterministic_and_normalised(self) -> None:
        left = hashing_embedding("langgraph agents and state machines")
        right = hashing_embedding("langgraph agents and state machines")
        assert left == right

        norm = sum(value * value for value in left) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_similar_text_scores_higher_than_unrelated_text(self) -> None:
        anchor = hashing_embedding("langgraph agents state machines orchestration")
        similar = hashing_embedding("state machines and langgraph agent orchestration")
        unrelated = hashing_embedding("kubernetes helm autoscaling probes")

        assert cosine_similarity(anchor, similar) > cosine_similarity(anchor, unrelated)

    def test_empty_input_never_produces_a_zero_vector(self) -> None:
        vector = hashing_embedding("")
        assert any(value != 0.0 for value in vector)


# --------------------------------------------------------------------------- #
# Mesh client + observability                                                 #
# --------------------------------------------------------------------------- #

class TestMeshClient:
    """The Mesh gateway's parsing and failure semantics."""

    def test_missing_key_raises_a_typed_error(self) -> None:
        from app.agent.mesh_client import MeshUnavailableError, get_mesh_client

        with pytest.raises(MeshUnavailableError):
            get_mesh_client()

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('{"a": 1}', {"a": 1}),
            ('```json\n{"a": 1}\n```', {"a": 1}),
            ('Here you go:\n{"a": 1}\nHope that helps!', {"a": 1}),
            ('[{"a": 1}]', [{"a": 1}]),
            ("not json at all", None),
            ("", None),
        ],
    )
    def test_json_extraction_survives_model_chattiness(
        self, raw: str, expected: Any
    ) -> None:
        assert extract_json(raw) == expected

    def test_telemetry_aggregates_calls(self) -> None:
        from app.agent.mesh_client import MeshCallRecord

        telemetry = MeshTelemetry()
        telemetry.record(
            MeshCallRecord(model="m", purpose="p", latency_ms=10.0,
                           prompt_tokens=5, completion_tokens=7)
        )
        telemetry.record(
            MeshCallRecord(model="m2", purpose="p2", latency_ms=5.0, ok=False, error="boom")
        )

        summary = telemetry.summary()
        assert summary["mesh_calls"] == 2
        assert summary["mesh_failures"] == 1
        assert summary["total_tokens"] == 12
        assert set(summary["models_used"]) == {"m", "m2"}


class TestObservability:
    """LangSmith wiring (BONUS 3) — disabled in tests, but the config is asserted."""

    def test_tracing_is_off_without_a_key(self) -> None:
        assert tracing_enabled() is False

    def test_run_config_always_sets_a_thread_id(self, user: User) -> None:
        config = build_run_config(user.id, trigger_reason="manual", event_count=3)
        assert config["configurable"]["thread_id"] == f"user-{user.id}"
        assert config["recursion_limit"] > 0

    def test_run_config_omits_trace_metadata_when_disabled(self, user: User) -> None:
        config = build_run_config(user.id, trigger_reason="manual", event_count=3)
        # Selective tracing: no tags/metadata are attached when tracing is off.
        assert "metadata" not in config
        assert "tags" not in config


class TestRunnerHelpers:
    """`run_agent_now` bypasses the trigger policy but keeps the lock."""

    def test_run_agent_now_generates_immediately(
        self, db: Session, user: User, products: list[Product]
    ) -> None:
        result = run_agent_now(user.id, reason="manual")
        assert result.ok
        assert db.get(Recommendation, result.recommendation_id) is not None
