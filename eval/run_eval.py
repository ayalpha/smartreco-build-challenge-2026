"""Runs `nexora-decision-points` through the real pipeline via
`langsmith.evaluate()`.

Evaluates Nexora's specific architectural features:
1. Trigger rules (first_time, event_threshold, stale)
2. Refinement loop on sparse retrieval
3. Graceful degradation on Mesh API failure

Run directly: `python -m eval.run_eval`
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from langsmith import Client, evaluate

from app.agent.runner import run_agent
from app.agent.triggers import evaluate as trigger_evaluate
from app.database import SessionLocal
from app.models.event import Event, EventType
from app.models.recommendation import Recommendation
from eval.build_dataset import DATASET_NAME, build_dataset


def _seed_test_data():
    """Seed the database with the synthetic user events for our test scenarios."""
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        
        # User 1: Cold start (No events, no recommendations)
        # Database handles this by absence of rows.
        
        # User 2: Sparse retrieval (Needs refinement loop)
        # We give them a bunch of events looking for something that doesn't exist
        # to trigger the retrieval_refiner loop.
        for i in range(12):
            db.add(Event(
                user_id=2, session_id="test_sess_2",
                event_type=EventType.SEARCH_QUERY,
                product_id=None,
                metadata_json={"query": "advanced cobol mainframe programming from 1980"},
                timestamp=now - timedelta(minutes=i)
            ))
            
        # User 3: Stale trigger (old recommendation, 5+ new events)
        db.add(Recommendation(
            user_id=3, products=[], narrative="Old recommendation",
            is_active=False, created_at=now - timedelta(hours=3)
        ))
        for i in range(5):
            db.add(Event(
                user_id=3, session_id="test_sess_3",
                event_type=EventType.PRODUCT_CLICK, product_id=1,
                timestamp=now - timedelta(minutes=i)
            ))
            
        # User 4: Degraded fallback test
        for i in range(2):
            db.add(Event(
                user_id=4, session_id="test_sess_4",
                event_type=EventType.PRODUCT_CLICK, product_id=1,
                timestamp=now - timedelta(minutes=i)
            ))
            
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Test seeding failed or already exists: {e}")
    finally:
        db.close()


def target(inputs: dict) -> dict:
    user_id = inputs["user_id"]
    scenario = inputs["test_scenario"]
    
    db = SessionLocal()
    
    # 1. Check trigger evaluation
    trigger_decision = trigger_evaluate(db, user_id)
    db.close()
    
    # 2. Run agent (with special mock for degraded mode)
    original_api_key = os.environ.get("MESH_API_KEY", "")
    
    if scenario == "degraded_fallback":
        # Induce a Mesh API failure to verify fallback logic
        os.environ["MESH_API_KEY"] = "invalid_key_for_test"
        
    try:
        # Run synchronous entry point
        result = run_agent(
            user_id=user_id,
            reason=inputs["trigger_reason"],
            event_count=inputs["event_count"],
            respect_lock=False # We are running sequentially in tests
        )
    finally:
        if scenario == "degraded_fallback":
            os.environ["MESH_API_KEY"] = original_api_key
            
    trace = result.trace or {}
    
    return {
        "trigger_reason": trigger_decision.reason,
        "refinement_fires": trace.get("retry_count", 0) > 0,
        "degraded": result.degraded,
        "status": "ok" if result.ok else "error",
        "error_message": result.error,
        "recommendation_id": result.recommendation_id
    }


# ==================== EVALUATORS ====================
def trigger_gate_correct(run, example) -> dict:
    actual = run.outputs["trigger_reason"]
    expected = example.outputs["expected_trigger_reason"]
    return {"key": "trigger_gate_correct", "score": actual == expected}


def refinement_fires(run, example) -> dict:
    actual = run.outputs["refinement_fires"]
    expected = example.outputs["expected_refinement"]
    return {"key": "refinement_fires", "score": actual == expected}


def degraded_fallback_as_expected(run, example) -> dict:
    actual = run.outputs["degraded"]
    expected = example.outputs["expected_degraded"]
    return {"key": "degraded_fallback_as_expected", "score": actual == expected}


def structured_output_valid(run, example) -> dict:
    # Pass iff the run reached a valid conclusion without crashing
    # For Nexora, it should NEVER error, even when degraded.
    # So "ok" should be True, meaning it generated a recommendation ID.
    return {
        "key": "structured_output_valid",
        "score": run.outputs["status"] == "ok" and run.outputs["recommendation_id"] is not None
    }


EVALUATORS = [
    trigger_gate_correct,
    refinement_fires,
    degraded_fallback_as_expected,
    structured_output_valid,
]


if __name__ == "__main__":
    _seed_test_data()
    build_dataset()
    
    client = Client()
    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=EVALUATORS,
        experiment_prefix="nexora-decision-points",
        client=client,
        max_concurrency=1,
    )
    client.flush()
    print(results)
