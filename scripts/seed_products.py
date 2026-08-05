"""Idempotent catalog seeder.

Loads a curated 50 AI / data / engineering courses into SQL **and** mirrors them
into Qdrant through the same dual-write path the admin UI uses.

Idempotency
-----------
Courses are matched on ``title``: an existing row is updated in place (preserving
its id, so recommendation history and event foreign keys stay valid) and a
missing row is inserted.  Running this script five times leaves exactly 50
courses, five times.

Usage
-----
::

    python -m scripts.seed_products                     # courses only
    python -m scripts.seed_products --demo-users        # + admin & learner accounts
    python -m scripts.seed_products --demo-users --with-events
                                                        # + synthetic behaviour,
                                                        #   so the agent has signal
    python -m scripts.seed_products --reindex           # drop & rebuild the vectors
    python -m scripts.seed_products --run-agent         # generate a recommendation

``--with-events`` is the fastest way to make the demo interesting: it fabricates a
plausible browsing session for ``learner@smartreco.ai`` (searches, clicks, dwell
times and a cart add, all weighted toward agentic AI), which is exactly the shape
of input the recommendation graph is designed to read.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# Make the project importable when run as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import func, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.database import init_db, session_scope  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.models.event import Event, EventType  # noqa: E402
from app.models.product import Product  # noqa: E402
from app.models.user import User, UserRole  # noqa: E402
from app.security import hash_password  # noqa: E402
from app.vector_store.sync import reindex_all, verify_sync  # noqa: E402

logger = logging.getLogger("scripts.seed_products")

#: Course covers are first-party generated artwork committed to the repo — one
#: coherent abstract-technical style across the whole catalog: near-black ink
#: ground, faint technical grid, a single luminous schematic in signal-green
#: and iris-violet, no lettering. All 50 are AI-generated JPEGs at 800x447,
#: ~31KB each (~1.5MB for the set). No external hotlinks, no stock photos,
#: nothing that can 404 or render something absurd on a course it doesn't match.
COVER_URL = "/static/img/courses/{filename}"
COVER_DIR = PROJECT_ROOT / "app" / "static" / "img" / "courses"

#: Extensions probed, in order of preference. Every cover is currently .jpg;
#: this stays a tuple so another format can be introduced without touching
#: the resolver below.
COVER_EXTENSIONS = (".jpg",)

#: Fallback used only if a title is not in :data:`COVER_BY_TITLE`.
DEFAULT_COVER = "agentic-ai"

#: Every curated course maps to its OWN cover — 50 courses, 50 distinct
#: images, no slug reused. Each motif is drawn from what the course actually
#: teaches (a retry-looping DAG for Airflow, sharded parameter planes for
#: distributed training), so the art carries information rather than decorating.
COVER_BY_TITLE: dict[str, str] = {
    'Building Production Agents with LangGraph': 'agentic-ai',
    'Agentic RAG: Retrieval That Reasons': 'rag-retrieval',
    'Multi-Agent Systems: Coordination Patterns': 'multi-agent',
    'LLM Observability with LangSmith': 'llm-observability',
    'Prompt Engineering for Structured Output': 'prompt-engineering',
    'Evaluation-Driven LLM Development': 'ai-evaluation',
    'RAG Systems in Production': 'rag-production',
    'Machine Learning Foundations with scikit-learn': 'machine-learning',
    'Feature Engineering That Actually Moves Metrics': 'feature-engineering',
    'Recommender Systems from Scratch': 'recommender-systems',
    'MLOps: Shipping Models That Survive Contact With Users': 'mlops',
    'Statistics for Machine Learning Practitioners': 'statistics',
    'Gradient Boosting in Depth: XGBoost and LightGBM': 'gradient-boosting',
    'Deep Learning with PyTorch: Fundamentals to Fine-Tuning': 'deep-learning',
    'Transformers and Attention, Implemented Line by Line': 'transformers',
    'Fine-Tuning LLMs with LoRA and QLoRA': 'fine-tuning',
    'Computer Vision in Production': 'computer-vision',
    'Distributed Training with FSDP and DeepSpeed': 'distributed-training',
    'Data Engineering with dbt and Modern SQL': 'data-engineering',
    'Apache Airflow: Orchestration You Can Debug at 3am': 'airflow',
    'Streaming Data with Kafka and Flink': 'streaming',
    'Vector Databases and Semantic Search at Scale': 'vector-database',
    'Analytics at Speed with DuckDB': 'duckdb',
    'Lakehouse Architecture with Apache Iceberg': 'lakehouse',
    'Modern Python: Type Hints, Async and Packaging': 'python',
    'FastAPI in Production': 'fastapi',
    'Testing Python: Pytest, Fixtures and Property-Based Testing': 'testing-python',
    'Rust for Python Engineers': 'rust',
    'JavaScript Deep Dive: The Event Loop and Beyond': 'javascript',
    'TypeScript for Large Codebases': 'typescript',
    'Modern React Patterns and Server Components': 'react',
    'Full-Stack Web Development with Modern Tooling': 'web-development',
    'API Design: REST, GraphQL and When to Use Which': 'api-design',
    'Web Security Essentials for Application Developers': 'web-security',
    'Frontend Performance: Core Web Vitals in Practice': 'frontend-performance',
    'Accessibility Engineering for Web Applications': 'accessibility',
    'Docker and Containers: A Working Mental Model': 'devops-containers',
    'Kubernetes for Application Teams': 'kubernetes',
    'CI/CD with GitHub Actions': 'cicd',
    'Observability: Logs, Metrics and Traces': 'observability',
    'Incident Response and On-Call Engineering': 'incident-response',
    'Cloud Architecture Fundamentals': 'cloud',
    'Infrastructure as Code with Terraform': 'infrastructure-as-code',
    'Serverless Patterns and Anti-Patterns': 'serverless',
    'Cloud Cost Engineering and FinOps': 'finops',
    'Event-Driven Architecture on the Cloud': 'event-driven',
    'Technical Interviews for Data and ML Roles': 'career-skills',
    'Writing for Engineers: Design Docs and Postmortems': 'technical-writing',
    'From Engineer to Tech Lead': 'tech-lead',
    'System Design Interviews for Senior Engineers': 'system-design',
}

DEMO_PASSWORD = "smartreco123"


def _cover_slug(title: str) -> str:
    """Return the generated-cover slug for a course title."""
    return COVER_BY_TITLE.get(title, DEFAULT_COVER)


def _thumbnail(title: str) -> str:
    """Build the static path to this course's cover.

    Falls back to :data:`DEFAULT_COVER` for any title without an explicit
    mapping or with a missing file, so a course always renders a real image
    rather than a broken one.
    """
    slug = _cover_slug(title)
    for candidate in (slug, DEFAULT_COVER):
        for extension in COVER_EXTENSIONS:
            if not COVER_DIR.exists() or (COVER_DIR / f"{candidate}{extension}").exists():
                if candidate != slug:
                    logger.warning("Cover for %r missing — falling back to %s", title, candidate)
                return COVER_URL.format(filename=f"{candidate}{extension}")
    return COVER_URL.format(filename=f"{DEFAULT_COVER}.jpg")


# --------------------------------------------------------------------------- #
# Catalog                                                                     #
# --------------------------------------------------------------------------- #

#: 50 curated courses across the ten categories the brief calls for.  Descriptions are
#: written as prose because they are embedded for semantic retrieval — vague copy
#: measurably degrades recommendation quality.
CATALOG: list[dict[str, Any]] = [
    {
        "title": "Building Production Agents with LangGraph",
        "category": "Agentic AI",
        "skill_level": "intermediate",
        "price": 89.0,
        "duration": "14 hours",
        "instructor": "Priya Raghavan",
        "rating": 4.9,
        "tags": ["langgraph", "agents", "state machines", "python", "orchestration"],
        "description": (
            "Design agents as explicit state machines instead of prompt spaghetti. You will "
            "build a typed graph with conditional edges, add checkpointing so runs are "
            "replayable, implement retry and refinement loops, and ship an agent that degrades "
            "gracefully when a model call fails. Ends with a production deployment that "
            "handles concurrency, idempotency and observability."
        ),
    },
    {
        "title": "Agentic RAG: Retrieval That Reasons",
        "category": "Agentic AI",
        "skill_level": "intermediate",
        "price": 79.0,
        "duration": "10 hours",
        "instructor": "Daniel Okafor",
        "rating": 4.8,
        "tags": ["rag", "retrieval", "agents", "reranking", "vector search"],
        "description": (
            "Naive RAG retrieves once and hopes. Agentic RAG grades what it retrieved, rewrites "
            "the query when results are thin, and decides when it has enough context. Covers "
            "relevance grading, query rewriting loops, hybrid dense plus keyword retrieval, "
            "reciprocal rank fusion and LLM-as-judge re-ranking, with honest evaluation of "
            "when each technique actually helps."
        ),
    },
    {
        "title": "Multi-Agent Systems: Coordination Patterns",
        "category": "Agentic AI",
        "skill_level": "advanced",
        "price": 129.0,
        "duration": "16 hours",
        "instructor": "Priya Raghavan",
        "rating": 4.7,
        "tags": ["multi-agent", "orchestration", "supervisor", "planning", "tools"],
        "description": (
            "When one agent is not enough: supervisor architectures, hierarchical delegation, "
            "shared scratchpads and message-passing protocols. You will implement a research "
            "team that plans, splits work, critiques its own output and reconciles conflicting "
            "findings, then measure whether the added complexity beat a single well-prompted agent."
        ),
    },
    {
        "title": "LLM Observability with LangSmith",
        "category": "Agentic AI",
        "skill_level": "intermediate",
        "price": 59.0,
        "duration": "6 hours",
        "instructor": "Marta Alves",
        "rating": 4.6,
        "tags": ["langsmith", "tracing", "observability", "evaluation", "monitoring"],
        "description": (
            "You cannot debug what you cannot see. Instrument agent runs with distributed "
            "tracing, attach custom metadata and tags so runs are filterable, build evaluation "
            "datasets from production traffic, and set up regression tests that catch quality "
            "drops before users do. Includes selective tracing so your traces stay signal, not noise."
        ),
    },
    {
        "title": "Prompt Engineering for Structured Output",
        "category": "Agentic AI",
        "skill_level": "beginner",
        "price": 39.0,
        "duration": "5 hours",
        "instructor": "Marta Alves",
        "rating": 4.5,
        "tags": ["prompting", "json", "structured output", "validation", "llm"],
        "description": (
            "Getting reliable JSON out of a language model is an engineering problem, not a "
            "prompting trick. Schema-first prompt design, defensive parsing, repair loops, "
            "grammar constraints and validation with Pydantic. You will build a pipeline that "
            "survives the model wrapping its answer in prose or a code fence."
        ),
    },
    {
        "title": "Machine Learning Foundations with scikit-learn",
        "category": "Machine Learning",
        "skill_level": "beginner",
        "price": 49.0,
        "duration": "18 hours",
        "instructor": "Sofia Duarte",
        "rating": 4.8,
        "tags": ["scikit-learn", "regression", "classification", "python", "sklearn"],
        "description": (
            "The honest introduction: how to frame a problem, build a baseline, and know when "
            "your model is fooling you. Linear and tree-based models, proper cross-validation, "
            "leakage detection, class imbalance, and calibration. Every concept is implemented "
            "on real tabular data rather than a toy dataset."
        ),
    },
    {
        "title": "Feature Engineering That Actually Moves Metrics",
        "category": "Machine Learning",
        "skill_level": "intermediate",
        "price": 69.0,
        "duration": "11 hours",
        "instructor": "Sofia Duarte",
        "rating": 4.7,
        "tags": ["feature engineering", "pandas", "encoding", "time series", "tabular"],
        "description": (
            "Most model gains come from features, not architecture. Target encoding without "
            "leakage, temporal features that respect causality, aggregation windows, interaction "
            "discovery and feature stores. Includes a rigorous ablation methodology so you can "
            "prove which features earned their place."
        ),
    },
    {
        "title": "Recommender Systems from Scratch",
        "category": "Machine Learning",
        "skill_level": "intermediate",
        "price": 89.0,
        "duration": "15 hours",
        "instructor": "Hiroshi Tanaka",
        "rating": 4.9,
        "tags": ["recommenders", "collaborative filtering", "embeddings", "ranking", "cold start"],
        "description": (
            "Build a recommender end to end: implicit-feedback matrix factorisation, "
            "content-based retrieval with embeddings, two-stage retrieve-then-rank "
            "architectures, and the cold-start problem every real system faces. Evaluated with "
            "offline ranking metrics and an honest discussion of why they disagree with A/B tests."
        ),
    },
    {
        "title": "MLOps: Shipping Models That Survive Contact With Users",
        "category": "Machine Learning",
        "skill_level": "advanced",
        "price": 119.0,
        "duration": "20 hours",
        "instructor": "Ahmed Rahal",
        "rating": 4.7,
        "tags": ["mlops", "mlflow", "monitoring", "drift", "deployment"],
        "description": (
            "Experiment tracking, model registries, reproducible training pipelines, shadow "
            "deployments, drift detection and automated rollback. You will build a pipeline where "
            "a retrained model cannot reach production without passing evaluation gates, and "
            "where a bad deploy is detected in minutes rather than quarters."
        ),
    },
    {
        "title": "Statistics for Machine Learning Practitioners",
        "category": "Machine Learning",
        "skill_level": "beginner",
        "price": 45.0,
        "duration": "12 hours",
        "instructor": "Elena Vasquez",
        "rating": 4.6,
        "tags": ["statistics", "probability", "hypothesis testing", "bayesian", "inference"],
        "description": (
            "The statistical intuition that separates practitioners from people running "
            "library calls. Sampling distributions, confidence intervals, the multiple-comparisons "
            "trap, effect sizes, and Bayesian reasoning about uncertainty. Written for engineers "
            "who need to interpret an experiment, not to prove theorems."
        ),
    },
    {
        "title": "Deep Learning with PyTorch: Fundamentals to Fine-Tuning",
        "category": "Deep Learning",
        "skill_level": "intermediate",
        "price": 99.0,
        "duration": "22 hours",
        "instructor": "Hiroshi Tanaka",
        "rating": 4.9,
        "tags": ["pytorch", "neural networks", "backpropagation", "training", "gpu"],
        "description": (
            "Autograd from first principles, then real training loops: learning-rate schedules, "
            "mixed precision, gradient accumulation, distributed data parallel, and the debugging "
            "discipline for a loss that will not go down. Finishes by fine-tuning a pretrained "
            "model on a custom dataset with a proper evaluation harness."
        ),
    },
    {
        "title": "Transformers and Attention, Implemented Line by Line",
        "category": "Deep Learning",
        "skill_level": "advanced",
        "price": 109.0,
        "duration": "17 hours",
        "instructor": "Hiroshi Tanaka",
        "rating": 4.9,
        "tags": ["transformers", "attention", "llm", "architecture", "pytorch"],
        "description": (
            "Build a transformer from scratch: scaled dot-product attention, multi-head "
            "projections, positional encodings, KV caching and the modern variants (RoPE, "
            "grouped-query attention, RMSNorm) that make inference cheap. You will train a small "
            "model, profile it, and understand precisely where the FLOPs go."
        ),
    },
    {
        "title": "Fine-Tuning LLMs with LoRA and QLoRA",
        "category": "Deep Learning",
        "skill_level": "advanced",
        "price": 119.0,
        "duration": "13 hours",
        "instructor": "Ahmed Rahal",
        "rating": 4.8,
        "tags": ["fine-tuning", "lora", "peft", "quantization", "llm"],
        "description": (
            "Adapt a large model on a single GPU. Parameter-efficient fine-tuning, 4-bit "
            "quantisation, dataset curation and formatting, catastrophic-forgetting mitigation, "
            "and evaluation that detects when your tune made the model worse at everything else. "
            "Includes the decision framework for fine-tune versus RAG versus prompt engineering."
        ),
    },
    {
        "title": "Computer Vision in Production",
        "category": "Deep Learning",
        "skill_level": "intermediate",
        "price": 89.0,
        "duration": "16 hours",
        "instructor": "Lucas Meyer",
        "rating": 4.5,
        "tags": ["computer vision", "cnn", "detection", "segmentation", "onnx"],
        "description": (
            "Classification, detection and segmentation with modern backbones, then the parts "
            "tutorials skip: data augmentation that reflects real distribution shift, annotation "
            "quality control, ONNX export, quantised inference at the edge, and monitoring for "
            "the day your camera angle changes."
        ),
    },
    {
        "title": "Data Engineering with dbt and Modern SQL",
        "category": "Data Engineering",
        "skill_level": "intermediate",
        "price": 79.0,
        "duration": "14 hours",
        "instructor": "Nadia Hussain",
        "rating": 4.8,
        "tags": ["dbt", "sql", "warehouse", "modeling", "testing"],
        "description": (
            "Treat analytics code like software. Layered model architecture, incremental "
            "materialisations, snapshots for slowly changing dimensions, data tests as contracts, "
            "and documentation that stays current because it is generated. Includes CI that "
            "blocks a merge when a data test fails."
        ),
    },
    {
        "title": "Apache Airflow: Orchestration You Can Debug at 3am",
        "category": "Data Engineering",
        "skill_level": "intermediate",
        "price": 69.0,
        "duration": "12 hours",
        "instructor": "Nadia Hussain",
        "rating": 4.5,
        "tags": ["airflow", "orchestration", "dags", "scheduling", "etl"],
        "description": (
            "Idempotent tasks, correct backfills, sensible retry semantics and the "
            "execution-date confusion that causes most Airflow incidents. You will build a "
            "pipeline that is safe to re-run, alerts on the right failures, and does not silently "
            "skip a day when a DAG is paused."
        ),
    },
    {
        "title": "Streaming Data with Kafka and Flink",
        "category": "Data Engineering",
        "skill_level": "advanced",
        "price": 129.0,
        "duration": "19 hours",
        "instructor": "Kwame Mensah",
        "rating": 4.6,
        "tags": ["kafka", "flink", "streaming", "exactly-once", "event driven"],
        "description": (
            "Event-time versus processing-time, watermarks, windowing, and what exactly-once "
            "actually guarantees. You will build a stateful stream processor with checkpointing, "
            "handle out-of-order and late events correctly, and reason about the throughput and "
            "latency trade-offs under real partitioning."
        ),
    },
    {
        "title": "Vector Databases and Semantic Search at Scale",
        "category": "Data Engineering",
        "skill_level": "intermediate",
        "price": 79.0,
        "duration": "10 hours",
        "instructor": "Daniel Okafor",
        "rating": 4.8,
        "tags": ["qdrant", "vector database", "embeddings", "hnsw", "hybrid search"],
        "description": (
            "How approximate nearest-neighbour indexes actually work (HNSW, IVF, product "
            "quantisation) and what you trade for speed. Metadata filtering, hybrid dense plus "
            "sparse retrieval, reciprocal rank fusion, re-indexing strategies for embedding "
            "migrations, and keeping a vector store in sync with a relational system of record."
        ),
    },
    {
        "title": "Modern Python: Type Hints, Async and Packaging",
        "category": "Python",
        "skill_level": "intermediate",
        "price": 59.0,
        "duration": "13 hours",
        "instructor": "Marta Alves",
        "rating": 4.8,
        "tags": ["python", "typing", "asyncio", "packaging", "mypy"],
        "description": (
            "The Python that ships in 2026: expressive type hints and generics, structural "
            "subtyping with protocols, asyncio concurrency patterns that avoid the common "
            "deadlocks, dependency management and reproducible builds, and a strict mypy "
            "configuration you can actually adopt on an existing codebase."
        ),
    },
    {
        "title": "FastAPI in Production",
        "category": "Python",
        "skill_level": "intermediate",
        "price": 69.0,
        "duration": "12 hours",
        "instructor": "Tomas Nowak",
        "rating": 4.9,
        "tags": ["fastapi", "api", "pydantic", "async", "sqlalchemy"],
        "description": (
            "Dependency injection that stays testable, Pydantic models as your contract, "
            "SQLAlchemy 2.0 sessions scoped correctly, background tasks that do not block "
            "responses, structured logging, and graceful shutdown. Ends with a service that has "
            "readiness probes, migrations and a real test suite."
        ),
    },
    {
        "title": "Testing Python: Pytest, Fixtures and Property-Based Testing",
        "category": "Python",
        "skill_level": "beginner",
        "price": 45.0,
        "duration": "8 hours",
        "instructor": "Tomas Nowak",
        "rating": 4.7,
        "tags": ["pytest", "testing", "fixtures", "hypothesis", "mocking"],
        "description": (
            "Tests that catch regressions instead of cementing current behaviour. Fixture "
            "composition and scoping, parametrisation, sensible mocking boundaries, property-based "
            "testing with Hypothesis, and coverage read as a diagnostic rather than a target."
        ),
    },
    {
        "title": "JavaScript Deep Dive: The Event Loop and Beyond",
        "category": "JavaScript",
        "skill_level": "intermediate",
        "price": 55.0,
        "duration": "11 hours",
        "instructor": "Lucas Meyer",
        "rating": 4.6,
        "tags": ["javascript", "event loop", "promises", "closures", "performance"],
        "description": (
            "Why your await resolved in that order. Microtasks and macrotasks, closure and scope "
            "semantics, prototype chains, memory leaks in long-lived pages, and profiling with "
            "the performance panel. The mental model that makes async JavaScript predictable."
        ),
    },
    {
        "title": "TypeScript for Large Codebases",
        "category": "JavaScript",
        "skill_level": "intermediate",
        "price": 65.0,
        "duration": "12 hours",
        "instructor": "Lucas Meyer",
        "rating": 4.7,
        "tags": ["typescript", "generics", "types", "refactoring", "monorepo"],
        "description": (
            "Types as design tools. Generics and conditional types, discriminated unions that "
            "make invalid states unrepresentable, declaration files, strictness flags introduced "
            "incrementally, and monorepo project references that keep builds fast as the codebase grows."
        ),
    },
    {
        "title": "Frontend Performance: Core Web Vitals in Practice",
        "category": "Web Development",
        "skill_level": "intermediate",
        "price": 59.0,
        "duration": "9 hours",
        "instructor": "Aisha Bello",
        "rating": 4.5,
        "tags": ["performance", "web vitals", "lighthouse", "caching", "bundling"],
        "description": (
            "Diagnose and fix what users actually feel: layout shift, input delay, and the "
            "largest paint. Critical-path CSS, code splitting, image strategy, font loading, "
            "caching headers, and measuring with real-user monitoring rather than a lab score "
            "on your own laptop."
        ),
    },
    {
        "title": "Full-Stack Web Development with Modern Tooling",
        "category": "Web Development",
        "skill_level": "beginner",
        "price": 65.0,
        "duration": "24 hours",
        "instructor": "Aisha Bello",
        "rating": 4.6,
        "tags": ["html", "css", "javascript", "http", "fullstack"],
        "description": (
            "Build and deploy a complete application. Semantic HTML and accessibility, modern "
            "CSS layout, progressive enhancement, HTTP and REST fundamentals, forms and "
            "validation, session handling, and a deployment pipeline. Deliberately framework-light "
            "so the fundamentals stick."
        ),
    },
    {
        "title": "API Design: REST, GraphQL and When to Use Which",
        "category": "Web Development",
        "skill_level": "intermediate",
        "price": 69.0,
        "duration": "10 hours",
        "instructor": "Tomas Nowak",
        "rating": 4.6,
        "tags": ["api design", "rest", "graphql", "versioning", "openapi"],
        "description": (
            "Resource modelling, pagination and filtering conventions, idempotency keys, error "
            "shapes clients can act on, versioning that does not break consumers, and honest "
            "trade-offs between REST, GraphQL and RPC. Includes contract testing and "
            "OpenAPI-driven client generation."
        ),
    },
    {
        "title": "Web Security Essentials for Application Developers",
        "category": "Web Development",
        "skill_level": "intermediate",
        "price": 75.0,
        "duration": "11 hours",
        "instructor": "Ahmed Rahal",
        "rating": 4.8,
        "tags": ["security", "owasp", "authentication", "jwt", "csrf"],
        "description": (
            "The vulnerability classes that actually get exploited: injection, broken access "
            "control, XSS, CSRF and session fixation. Password storage, token design and cookie "
            "flags that matter, secret management, and a threat-modelling habit you can apply in "
            "a design review."
        ),
    },
    {
        "title": "Docker and Containers: A Working Mental Model",
        "category": "DevOps",
        "skill_level": "beginner",
        "price": 49.0,
        "duration": "9 hours",
        "instructor": "Kwame Mensah",
        "rating": 4.7,
        "tags": ["docker", "containers", "images", "compose", "networking"],
        "description": (
            "What a container really is (namespaces, cgroups, layered filesystems), then the "
            "practice: small reproducible images, multi-stage builds, layer caching that speeds "
            "up CI, container networking and volumes, and multi-service local environments with "
            "Compose."
        ),
    },
    {
        "title": "Kubernetes for Application Teams",
        "category": "DevOps",
        "skill_level": "advanced",
        "price": 119.0,
        "duration": "21 hours",
        "instructor": "Kwame Mensah",
        "rating": 4.6,
        "tags": ["kubernetes", "helm", "operators", "scaling", "observability"],
        "description": (
            "Enough Kubernetes to run a service well, without becoming a full-time platform "
            "engineer. Deployments and rollout strategies, resource requests and limits that "
            "prevent noisy neighbours, probes, autoscaling, secrets, Helm packaging, and "
            "debugging a pod that will not start."
        ),
    },
    {
        "title": "CI/CD with GitHub Actions",
        "category": "DevOps",
        "skill_level": "beginner",
        "price": 45.0,
        "duration": "7 hours",
        "instructor": "Marta Alves",
        "rating": 4.5,
        "tags": ["ci", "cd", "github actions", "automation", "testing"],
        "description": (
            "Pipelines that fail fast and tell you why. Workflow and job structure, matrix builds, "
            "effective caching, reusable workflows, OIDC for keyless cloud auth, environment "
            "protection rules, and deployment gates. Includes a supply-chain hardening pass."
        ),
    },
    {
        "title": "Observability: Logs, Metrics and Traces",
        "category": "DevOps",
        "skill_level": "intermediate",
        "price": 79.0,
        "duration": "12 hours",
        "instructor": "Ahmed Rahal",
        "rating": 4.7,
        "tags": ["observability", "opentelemetry", "prometheus", "tracing", "slo"],
        "description": (
            "Instrument a distributed system so incidents are short. Structured logging, metric "
            "cardinality discipline, OpenTelemetry tracing across service boundaries, SLOs and "
            "error budgets, and alerts that page a human only when a human is needed."
        ),
    },
    {
        "title": "Cloud Architecture Fundamentals",
        "category": "Cloud",
        "skill_level": "beginner",
        "price": 59.0,
        "duration": "13 hours",
        "instructor": "Elena Vasquez",
        "rating": 4.5,
        "tags": ["cloud", "aws", "architecture", "iam", "networking"],
        "description": (
            "The primitives every cloud gives you and how to compose them: compute options, "
            "object storage, managed databases, queues, VPC networking and identity. Includes "
            "cost modelling and the failure modes of each pattern, so your design survives both "
            "an outage and a finance review."
        ),
    },
    {
        "title": "Infrastructure as Code with Terraform",
        "category": "Cloud",
        "skill_level": "intermediate",
        "price": 85.0,
        "duration": "14 hours",
        "instructor": "Kwame Mensah",
        "rating": 4.6,
        "tags": ["terraform", "iac", "state", "modules", "automation"],
        "description": (
            "Declarative infrastructure that a team can share. State management and locking, "
            "module design and composition, workspaces versus directories, importing existing "
            "resources, plan review as a code-review artefact, and drift detection in CI."
        ),
    },
    {
        "title": "Serverless Patterns and Anti-Patterns",
        "category": "Cloud",
        "skill_level": "intermediate",
        "price": 69.0,
        "duration": "10 hours",
        "instructor": "Sofia Duarte",
        "rating": 4.3,
        "tags": ["serverless", "lambda", "event driven", "cold start", "queues"],
        "description": (
            "Where functions-as-a-service genuinely wins and where it quietly costs you more than "
            "a small server. Event-driven composition, cold-start mitigation, idempotent handlers, "
            "queue-based load levelling, and the observability gaps you have to close yourself."
        ),
    },
    {
        "title": "Technical Interviews for Data and ML Roles",
        "category": "Career Skills",
        "skill_level": "intermediate",
        "price": 55.0,
        "duration": "10 hours",
        "instructor": "Nadia Hussain",
        "rating": 4.7,
        "tags": ["interviews", "career", "system design", "ml design", "communication"],
        "description": (
            "What ML and data interviews are really testing. Case-study framing, ML system design, "
            "SQL and coding under time pressure, communicating a trade-off out loud, and the "
            "project narrative that makes a portfolio memorable. Includes graded mock rubrics."
        ),
    },
    {
        "title": "Writing for Engineers: Design Docs and Postmortems",
        "category": "Career Skills",
        "skill_level": "beginner",
        "price": 35.0,
        "duration": "6 hours",
        "instructor": "Aisha Bello",
        "rating": 4.8,
        "tags": ["writing", "communication", "design docs", "postmortems", "career"],
        "description": (
            "Writing is the highest-leverage engineering skill nobody teaches. Structure a design "
            "doc so reviewers engage with the decision instead of the prose, run a blameless "
            "postmortem that produces real action items, and write updates that make your work "
            "legible to people two levels away."
        ),
    },
    {
        "title": "From Engineer to Tech Lead",
        "category": "Career Skills",
        "skill_level": "advanced",
        "price": 79.0,
        "duration": "9 hours",
        "instructor": "Priya Raghavan",
        "rating": 4.6,
        "tags": ["leadership", "career", "mentoring", "planning", "influence"],
        "description": (
            "The transition from writing the most code to making the most decisions. Technical "
            "planning and sequencing, delegation that develops people, code review as mentorship, "
            "managing stakeholder expectations, and protecting focus time for a team that is "
            "constantly interrupted."
        ),
    },
    {
        "title": "Evaluation-Driven LLM Development",
        "category": "Agentic AI",
        "skill_level": "advanced",
        "price": 99.0,
        "duration": "12 hours",
        "instructor": "Marta Alves",
        "rating": 4.8,
        "tags": ["evaluation", "llm", "testing", "datasets", "regression"],
        "description": (
            "Shipping an LLM feature without an eval suite is shipping a rumour. Build golden "
            "datasets from production traffic, write graders that correlate with human judgement, "
            "and wire regression gates into CI so a prompt change cannot quietly degrade quality."
        ),
    },
    {
        "title": "RAG Systems in Production",
        "category": "Agentic AI",
        "skill_level": "intermediate",
        "price": 89.0,
        "duration": "13 hours",
        "instructor": "Daniel Okafor",
        "rating": 4.9,
        "tags": ["rag", "chunking", "reranking", "embeddings", "latency"],
        "description": (
            "The gap between a RAG demo and a RAG product is chunking strategy, retrieval "
            "evaluation and latency budgets. Covers document parsing, chunk boundary design, "
            "hybrid retrieval, cross-encoder re-ranking, and caching layers that keep p95 sane."
        ),
    },
    {
        "title": "Gradient Boosting in Depth: XGBoost and LightGBM",
        "category": "Machine Learning",
        "skill_level": "intermediate",
        "price": 75.0,
        "duration": "11 hours",
        "instructor": "Hiroshi Tanaka",
        "rating": 4.8,
        "tags": ["xgboost", "lightgbm", "boosting", "tabular", "tuning"],
        "description": (
            "Still the strongest default for tabular problems, and still widely misused. How the "
            "boosting objective actually works, which hyperparameters matter and which are noise, "
            "categorical handling, and reading SHAP values without over-reading them."
        ),
    },
    {
        "title": "Distributed Training with FSDP and DeepSpeed",
        "category": "Deep Learning",
        "skill_level": "advanced",
        "price": 129.0,
        "duration": "15 hours",
        "instructor": "Hiroshi Tanaka",
        "rating": 4.7,
        "tags": ["distributed", "fsdp", "deepspeed", "gpu", "scaling"],
        "description": (
            "When a model no longer fits on one GPU. Data, tensor and pipeline parallelism, fully "
            "sharded data parallel, activation checkpointing, and the communication patterns that "
            "decide whether adding GPUs actually makes training faster."
        ),
    },
    {
        "title": "Analytics at Speed with DuckDB",
        "category": "Data Engineering",
        "skill_level": "beginner",
        "price": 49.0,
        "duration": "8 hours",
        "instructor": "Elena Vasquez",
        "rating": 4.7,
        "tags": ["duckdb", "olap", "parquet", "analytics", "sql"],
        "description": (
            "An entire analytics engine in a single process, and often faster than the cluster you "
            "were about to provision. Columnar execution, querying Parquet in place, larger-than-"
            "memory workloads, and where DuckDB stops being the right answer."
        ),
    },
    {
        "title": "Lakehouse Architecture with Apache Iceberg",
        "category": "Data Engineering",
        "skill_level": "advanced",
        "price": 105.0,
        "duration": "14 hours",
        "instructor": "Kwame Mensah",
        "rating": 4.5,
        "tags": ["iceberg", "lakehouse", "table format", "time travel", "partitioning"],
        "description": (
            "Table formats brought database guarantees to object storage. Snapshot isolation, "
            "hidden partitioning, schema evolution without rewrites, compaction strategy, and "
            "time-travel queries that make debugging a bad load tractable."
        ),
    },
    {
        "title": "Rust for Python Engineers",
        "category": "Python",
        "skill_level": "advanced",
        "price": 99.0,
        "duration": "16 hours",
        "instructor": "Tomas Nowak",
        "rating": 4.8,
        "tags": ["rust", "pyo3", "systems", "performance", "interop"],
        "description": (
            "Learn Rust from a Python mental model, then use it where Python hurts. Ownership and "
            "borrowing explained against garbage collection, error handling without exceptions, "
            "and shipping a PyO3 extension that speeds up a real hot path in your codebase."
        ),
    },
    {
        "title": "Modern React Patterns and Server Components",
        "category": "JavaScript",
        "skill_level": "intermediate",
        "price": 79.0,
        "duration": "13 hours",
        "instructor": "Aisha Bello",
        "rating": 4.6,
        "tags": ["react", "server components", "hooks", "state", "rendering"],
        "description": (
            "React changed shape and a lot of received wisdom expired with it. Server versus client "
            "components, the boundaries between them, data fetching and caching, state colocation, "
            "and the rendering model that explains why your component ran three times."
        ),
    },
    {
        "title": "Accessibility Engineering for Web Applications",
        "category": "Web Development",
        "skill_level": "intermediate",
        "price": 69.0,
        "duration": "9 hours",
        "instructor": "Aisha Bello",
        "rating": 4.8,
        "tags": ["accessibility", "wcag", "aria", "screen readers", "keyboard"],
        "description": (
            "Accessibility is a set of engineering constraints, not a checklist bolted on at the "
            "end. Semantic structure, keyboard operability, ARIA for dynamic regions, focus "
            "management in modals, and testing with an actual screen reader rather than a linter."
        ),
    },
    {
        "title": "Incident Response and On-Call Engineering",
        "category": "DevOps",
        "skill_level": "intermediate",
        "price": 69.0,
        "duration": "8 hours",
        "instructor": "Ahmed Rahal",
        "rating": 4.7,
        "tags": ["incidents", "on call", "sre", "postmortems", "alerting"],
        "description": (
            "The skill of staying calm and effective while production burns. Incident command "
            "roles, communication cadence, mitigation before diagnosis, alert hygiene that "
            "protects sleep, and blameless postmortems that produce fixes instead of blame."
        ),
    },
    {
        "title": "Cloud Cost Engineering and FinOps",
        "category": "Cloud",
        "skill_level": "intermediate",
        "price": 79.0,
        "duration": "9 hours",
        "instructor": "Sofia Duarte",
        "rating": 4.6,
        "tags": ["finops", "cost", "cloud", "rightsizing", "architecture"],
        "description": (
            "Cloud spend is an architecture problem wearing a finance costume. Cost attribution and "
            "tagging discipline, rightsizing with real utilisation data, commitment strategy, "
            "egress traps, and the design choices that quietly triple a monthly bill."
        ),
    },
    {
        "title": "Event-Driven Architecture on the Cloud",
        "category": "Cloud",
        "skill_level": "advanced",
        "price": 99.0,
        "duration": "13 hours",
        "instructor": "Kwame Mensah",
        "rating": 4.6,
        "tags": ["event driven", "queues", "sagas", "idempotency", "messaging"],
        "description": (
            "Decoupling with events buys flexibility and bills you in complexity. Event schema "
            "design and versioning, idempotent consumers, the saga pattern for distributed "
            "workflows, dead-letter handling, and debugging a flow with no single call stack."
        ),
    },
    {
        "title": "System Design Interviews for Senior Engineers",
        "category": "Career Skills",
        "skill_level": "advanced",
        "price": 89.0,
        "duration": "12 hours",
        "instructor": "Nadia Hussain",
        "rating": 4.8,
        "tags": ["system design", "interviews", "architecture", "scaling", "communication"],
        "description": (
            "Senior system design interviews test judgement under ambiguity, not memorised "
            "diagrams. Requirement clarification, capacity estimation, explicit trade-offs, "
            "failure modes, and narrating your reasoning so the interviewer can follow it."
        ),
    },
]


# --------------------------------------------------------------------------- #
# Seeding                                                                     #
# --------------------------------------------------------------------------- #

def seed_catalog(db: Session) -> tuple[int, int]:
    """Insert or update every course in :data:`CATALOG`.

    Matching is by ``title``, so ids — and therefore event and recommendation
    foreign keys — survive repeated runs.

    Args:
        db: Open session.

    Returns:
        ``(created, updated)`` counts.
    """
    created = 0
    updated = 0

    for entry in CATALOG:
        existing = db.scalars(select(Product).where(Product.title == entry["title"])).first()
        payload = dict(entry)
        payload["tags"] = Product.normalise_tags(entry["tags"])
        # Always authoritative: migrates any previously-seeded external URL.
        payload["thumbnail_url"] = _thumbnail(entry["title"])

        if existing is None:
            product = Product(**payload)
            db.add(product)
            created += 1
        else:
            changed = False
            for field, value in payload.items():
                if getattr(existing, field, None) != value:
                    setattr(existing, field, value)
                    changed = True
            if changed:
                existing.revision = (existing.revision or 1) + 1
                existing.is_active = True
                updated += 1

    db.flush()
    logger.info("Catalog seeded: %d created, %d updated, %d total in CATALOG",
                created, updated, len(CATALOG))
    return created, updated


def seed_demo_users(db: Session) -> dict[str, int]:
    """Create (or refresh) the demo admin and learner accounts.

    Returns:
        ``{"admin": id, "learner": id}``.
    """
    accounts = [
        ("admin@smartreco.ai", "Ada Admin", UserRole.ADMIN),
        ("learner@smartreco.ai", "Leo Learner", UserRole.USER),
    ]
    ids: dict[str, int] = {}

    for email, full_name, role in accounts:
        user = db.scalars(select(User).where(User.email == email)).first()
        if user is None:
            user = User(
                email=email,
                full_name=full_name,
                hashed_password=hash_password(DEMO_PASSWORD),
                role=role,
                digest_opt_in=True,
            )
            db.add(user)
            db.flush()
            logger.info("Created demo user %s (%s)", email, role.value)
        else:
            user.role = role
            user.is_active = True
            logger.info("Demo user %s already exists (id=%s)", email, user.id)

        ids["admin" if role == UserRole.ADMIN else "learner"] = int(user.id)

    db.flush()
    return ids


def seed_demo_events(db: Session, user_id: int, *, seed: int = 20260803) -> int:
    """Fabricate a plausible browsing session so the agent has signal to read.

    The synthetic session is intentionally *lopsided* — heavily weighted toward
    Agentic AI with a secondary Data Engineering thread — because a uniform
    random session produces a bland recommendation and demonstrates nothing.

    Args:
        db: Open session.
        user_id: The learner to attribute events to.
        seed: RNG seed, so the demo is reproducible.

    Returns:
        The number of events created (0 if the user already has events).
    """
    existing = int(
        db.scalar(select(func.count()).select_from(Event).where(Event.user_id == user_id)) or 0
    )
    if existing:
        logger.info("User %s already has %d events — skipping synthetic session",
                    user_id, existing)
        return 0

    rng = random.Random(seed)

    primary = list(
        db.scalars(
            select(Product).where(Product.category == "Agentic AI").order_by(Product.id)
        )
    )
    secondary = list(
        db.scalars(
            select(Product).where(Product.category == "Data Engineering").order_by(Product.id)
        )
    )
    tertiary = list(
        db.scalars(
            select(Product).where(Product.category == "Python").order_by(Product.id).limit(2)
        )
    )

    if not primary:
        logger.warning("Cannot build a synthetic session: seed the catalog first")
        return 0

    session_id = "seed-session-001"
    cursor = datetime.now(timezone.utc) - timedelta(hours=3)
    events: list[Event] = []

    def add(
        event_type: str,
        *,
        product: Optional[Product] = None,
        path: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        gap_seconds: int = 45,
    ) -> None:
        """Append one event and advance the synthetic clock."""
        nonlocal cursor
        cursor = cursor + timedelta(seconds=gap_seconds)
        events.append(
            Event(
                user_id=user_id,
                session_id=session_id,
                event_type=event_type,
                product_id=product.id if product else None,
                path=path or (f"/product/{product.id}" if product else "/"),
                metadata_json=metadata or {},
                timestamp=cursor,
            )
        )

    # Arrival and an exploratory search.
    add(EventType.PAGE_VIEW.value, path="/", metadata={"referrer": "https://google.com"},
        gap_seconds=5)
    add(EventType.SEARCH_QUERY.value, path="/catalog",
        metadata={"query": "langgraph agents", "result_count": len(primary)}, gap_seconds=25)

    # Deep engagement with the primary interest.
    for product in primary[:3]:
        add(EventType.PRODUCT_CLICK.value, product=product,
            metadata={"source": "search", "product_title": product.title}, gap_seconds=18)
        add(EventType.PAGE_VIEW.value, product=product,
            metadata={"product_title": product.title}, gap_seconds=2)
        add(EventType.TIME_SPENT.value, product=product,
            metadata={"seconds": rng.randint(95, 260), "product_title": product.title},
            gap_seconds=rng.randint(100, 270))

    # A second, weaker thread.
    add(EventType.SEARCH_QUERY.value, path="/catalog",
        metadata={"query": "vector database hybrid search", "result_count": len(secondary)},
        gap_seconds=40)
    for product in secondary[:2]:
        add(EventType.PRODUCT_CLICK.value, product=product,
            metadata={"source": "search", "product_title": product.title}, gap_seconds=15)
        add(EventType.TIME_SPENT.value, product=product,
            metadata={"seconds": rng.randint(35, 90), "product_title": product.title},
            gap_seconds=rng.randint(40, 95))

    # Shallow browsing — deliberately weak signal the agent should discount.
    for product in tertiary:
        add(EventType.PRODUCT_CLICK.value, product=product,
            metadata={"source": "catalog", "product_title": product.title}, gap_seconds=12)
        add(EventType.TIME_SPENT.value, product=product,
            metadata={"seconds": rng.randint(6, 18), "product_title": product.title},
            gap_seconds=20)

    # The strongest intent signal of all.
    add(EventType.ADD_TO_CART.value, product=primary[0],
        metadata={"product_title": primary[0].title}, gap_seconds=30)

    db.add_all(events)
    db.flush()
    logger.info("Created %d synthetic events for user %s", len(events), user_id)
    return len(events)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Seed the SmartReco catalog (idempotent).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m scripts.seed_products\n"
            "  python -m scripts.seed_products --demo-users --with-events --run-agent\n"
            "  python -m scripts.seed_products --reindex\n"
        ),
    )
    parser.add_argument("--demo-users", action="store_true",
                        help="create admin@smartreco.ai and learner@smartreco.ai")
    parser.add_argument("--with-events", action="store_true",
                        help="fabricate a synthetic browsing session for the demo learner")
    parser.add_argument("--run-agent", action="store_true",
                        help="run the recommendation agent for the demo learner afterwards")
    parser.add_argument("--reindex", action="store_true",
                        help="drop and rebuild the Qdrant collection before syncing")
    parser.add_argument("--skip-vectors", action="store_true",
                        help="seed SQL only (skips embedding, useful for offline smoke tests)")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point.

    Returns:
        Process exit code (0 on success).
    """
    args = build_parser().parse_args(argv)
    configure_logging()

    logger.info("=" * 74)
    logger.info("SmartReco catalog seeder")
    logger.info("=" * 74)

    init_db()

    learner_id: Optional[int] = None

    with session_scope() as db:
        seed_catalog(db)

        if args.demo_users:
            ids = seed_demo_users(db)
            learner_id = ids.get("learner")

            if args.with_events and learner_id:
                seed_demo_events(db, learner_id)

    if args.skip_vectors:
        logger.info("--skip-vectors set: SQL seeded, vector store untouched.")
    else:
        with session_scope() as db:
            result = reindex_all(db, reset=args.reindex)
            logger.info("Vector sync: %s", result.message)
            status = verify_sync(db)
            logger.info(
                "Sync check — SQL: %s, vectors: %s, in_sync: %s, qdrant_mode: %s",
                status["sql_count"], status["vector_count"], status["in_sync"],
                "embedded" if status["embedded_mode"] else "server",
            )
            if not result.ok:
                logger.error("Vector sync failed: %s", result.error)
                return 1

    if args.run_agent:
        if learner_id is None:
            logger.warning("--run-agent needs --demo-users to know which learner to run for")
        else:
            from app.agent.runner import run_agent_now

            logger.info("Running the recommendation agent for user %s…", learner_id)
            outcome = run_agent_now(learner_id, reason="manual")
            logger.info("Agent result: %s", outcome.as_dict())
            if not outcome.ok:
                logger.error("Agent run did not produce a recommendation: %s", outcome.error)

    logger.info("Seeding complete.")
    if args.demo_users:
        logger.info(
            "Demo credentials — admin@smartreco.ai / learner@smartreco.ai, password: %s",
            DEMO_PASSWORD,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
