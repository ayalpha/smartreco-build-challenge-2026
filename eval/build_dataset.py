"""Idempotent LangSmith Dataset builder for Nexora Evals.

Creates (or reuses) the `nexora-decision-points` dataset with examples
that test the advanced architecture of Nexora: trigger logic, hybrid search,
refinement loops, and graceful degradation.

Run directly: `python -m eval.build_dataset` (needs LANGCHAIN_API_KEY set).
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

# Load environment variables
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DATASET_NAME = "nexora-decision-points"

# We map synthetic users to their user_id that will be seeded in the DB.
# user_id 1: First-time trigger (no previous recommendations)
# user_id 2: Refinement loop test (searches for extremely niche topic)
# user_id 3: Normal event threshold (10 events)
# user_id 4: Stale recommendation
# user_id 5: Degraded fallback test (API key disabled during test)

EXAMPLES = [
    {
        "inputs": {"user_id": 1, "trigger_reason": "first_time", "event_count": 0, "test_scenario": "cold_start"},
        "outputs": {
            "expected_trigger_reason": "first_time",
            "expected_refinement": False,
            "expected_degraded": False,
        },
    },
    {
        "inputs": {"user_id": 2, "trigger_reason": "event_threshold", "event_count": 12, "test_scenario": "sparse_retrieval"},
        "outputs": {
            "expected_trigger_reason": "event_threshold",
            "expected_refinement": True, # Should trigger refinement loop because of niche topic
            "expected_degraded": False,
        },
    },
    {
        "inputs": {"user_id": 3, "trigger_reason": "stale", "event_count": 5, "test_scenario": "stale_trigger"},
        "outputs": {
            "expected_trigger_reason": "stale",
            "expected_refinement": False,
            "expected_degraded": False,
        },
    },
    {
        "inputs": {"user_id": 4, "trigger_reason": "manual", "event_count": 2, "test_scenario": "degraded_fallback"},
        "outputs": {
            "expected_trigger_reason": "manual",
            "expected_refinement": False,
            "expected_degraded": True, # We will mock Mesh API failure in the runner for this scenario
        },
    },
]


def build_dataset() -> str:
    client = Client()

    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        dataset = existing[0]
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=(
                "Nexora decision points: Tests trigger logic, refinement loops, "
                "hybrid search, and graceful degradation."
            ),
        )

    already_covered = {
        ex.inputs.get("user_id") for ex in client.list_examples(dataset_id=dataset.id)
    }
    for example in EXAMPLES:
        if example["inputs"]["user_id"] in already_covered:
            continue
        client.create_example(
            inputs=example["inputs"],
            outputs=example["outputs"],
            dataset_id=dataset.id,
        )

    return dataset.id


if __name__ == "__main__":
    dataset_id = build_dataset()
    print(f"Dataset '{DATASET_NAME}' ready: {dataset_id}")
