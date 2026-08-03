"""Mesh API client — the **only** gateway for AI in this project.

★ HARD ARCHITECTURAL CONSTRAINT ★
Every LLM completion and every embedding in SmartReco is issued through the Mesh
API gateway at ``https://api.meshapi.ai/v1``.  There is no direct Anthropic,
OpenAI or Gemini SDK usage anywhere in the codebase.  The ``openai`` package is
used strictly as an OpenAI-*compatible* HTTP transport whose ``base_url`` points
at Mesh, which is Mesh's documented integration pattern:

    https://developers.meshapi.ai/docs/introduction/product-overview

Everything AI-shaped therefore funnels through the handful of functions below,
which gives us one place to add retries, timeouts, token accounting, structured
JSON coercion and graceful degradation.

Degradation contract
--------------------
When ``MESH_API_KEY`` is absent or Mesh returns a persistent error, these helpers
raise :class:`MeshUnavailableError`.  Every agent node catches it and falls back
to a deterministic heuristic, so the product keeps working (flagged
``degraded=True``) instead of 500-ing.  This is what makes the app safe to boot
in CI, where no key is configured.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

#: Errors worth retrying: transient network / rate-limit / 5xx conditions.
_RETRYABLE_MARKERS = (
    "rate limit",
    "rate_limit",
    "timeout",
    "timed out",
    "temporarily",
    "overloaded",
    "connection",
    "502",
    "503",
    "504",
    "429",
    "500",
)

_client_lock = threading.Lock()
_client: Optional[OpenAI] = None


class MeshError(Exception):
    """Base class for all Mesh transport failures."""


class MeshUnavailableError(MeshError):
    """Mesh could not fulfil the request (missing key, or retries exhausted).

    Callers are expected to catch this and degrade gracefully rather than fail.
    """


@dataclass
class MeshCallRecord:
    """Telemetry for a single Mesh call, surfaced in the agent trace."""

    model: str
    purpose: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ok: bool = True
    error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable projection stored on ``Recommendation.agent_trace``."""
        return {
            "model": self.model,
            "purpose": self.purpose,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ok": self.ok,
            "error": self.error,
        }


@dataclass
class MeshTelemetry:
    """Accumulates every Mesh call made during one agent run."""

    calls: list[MeshCallRecord] = field(default_factory=list)

    def record(self, record: MeshCallRecord) -> None:
        """Append a call record."""
        self.calls.append(record)

    @property
    def total_tokens(self) -> int:
        """Prompt + completion tokens across all recorded calls."""
        return sum(c.prompt_tokens + c.completion_tokens for c in self.calls)

    def summary(self) -> dict[str, Any]:
        """Aggregate view for logging and the ``agent_trace`` column."""
        return {
            "mesh_calls": len(self.calls),
            "mesh_failures": sum(0 if c.ok else 1 for c in self.calls),
            "total_tokens": self.total_tokens,
            "total_latency_ms": round(sum(c.latency_ms for c in self.calls), 2),
            "models_used": sorted({c.model for c in self.calls}),
            "detail": [c.as_dict() for c in self.calls],
        }


# --------------------------------------------------------------------------- #
# Client factory                                                              #
# --------------------------------------------------------------------------- #

def get_mesh_client() -> OpenAI:
    """Return the process-wide Mesh client (OpenAI SDK pointed at Mesh).

    This is the exact integration pattern from the Mesh docs — an OpenAI-shaped
    client with ``base_url`` overridden to the Mesh gateway.

    Returns:
        A configured, thread-safe :class:`openai.OpenAI` instance.

    Raises:
        MeshUnavailableError: If ``MESH_API_KEY`` is not configured.
    """
    global _client

    if not settings.mesh_api_key:
        raise MeshUnavailableError(
            "MESH_API_KEY is not configured. All AI features route through Mesh; "
            "set MESH_API_KEY in your environment or .env file."
        )

    if _client is not None:
        return _client

    with _client_lock:
        if _client is None:  # pragma: no branch - race guard
            logger.info("Initialising Mesh API client (base_url=%s)", settings.mesh_base_url)
            _client = OpenAI(
                base_url=settings.mesh_base_url,
                api_key=settings.mesh_api_key,
                timeout=settings.mesh_timeout_seconds,
                max_retries=0,  # retries handled here so they can be logged/traced
            )
    return _client


def reset_mesh_client() -> None:
    """Drop the cached client (used by tests and after config changes)."""
    global _client
    with _client_lock:
        _client = None


def mesh_available() -> bool:
    """True when a Mesh key is configured (does not perform a network call)."""
    return settings.mesh_configured


def _is_retryable(exc: BaseException) -> bool:
    """Heuristically decide whether an exception is worth retrying."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


# --------------------------------------------------------------------------- #
# Chat completions                                                            #
# --------------------------------------------------------------------------- #

def call_llm(
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    *,
    max_tokens: Optional[int] = None,
    purpose: str = "generic",
    telemetry: Optional[MeshTelemetry] = None,
) -> str:
    """Run a chat completion through Mesh and return the assistant text.

    Retries transient failures with exponential backoff
    (``MESH_MAX_RETRIES`` attempts, 0.75s × 2ⁿ delays).

    Args:
        messages: OpenAI-style ``[{"role": ..., "content": ...}]`` list.
        model: Mesh model id.  Defaults to ``MESH_MODEL_REASONING``.
        temperature: Sampling temperature.
        max_tokens: Optional completion cap.
        purpose: Short label recorded in telemetry (e.g. ``"interest_extractor"``).
        telemetry: Optional collector for this agent run.

    Returns:
        The assistant message content, stripped.

    Raises:
        MeshUnavailableError: If every attempt failed or no key is configured.
    """
    target_model = model or settings.mesh_model_reasoning
    attempts = max(1, settings.mesh_max_retries)
    last_error: Optional[BaseException] = None

    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            client = get_mesh_client()
            kwargs: dict[str, Any] = {
                "model": target_model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            response = client.chat.completions.create(**kwargs)
            latency_ms = (time.perf_counter() - started) * 1000.0

            content = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)

            if telemetry is not None:
                telemetry.record(
                    MeshCallRecord(
                        model=target_model,
                        purpose=purpose,
                        latency_ms=latency_ms,
                        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    )
                )

            logger.debug(
                "Mesh completion ok | purpose=%s model=%s attempt=%s latency=%.0fms chars=%d",
                purpose, target_model, attempt, latency_ms, len(content),
            )
            if not content:
                raise MeshError("Mesh returned an empty completion")
            return content

        except MeshUnavailableError:
            raise  # no key configured — retrying cannot help
        except Exception as exc:  # noqa: BLE001 - transport errors are heterogeneous
            last_error = exc
            latency_ms = (time.perf_counter() - started) * 1000.0
            if telemetry is not None:
                telemetry.record(
                    MeshCallRecord(
                        model=target_model,
                        purpose=purpose,
                        latency_ms=latency_ms,
                        ok=False,
                        error=str(exc)[:400],
                    )
                )
            if attempt < attempts and _is_retryable(exc):
                delay = 0.75 * (2 ** (attempt - 1))
                logger.warning(
                    "Mesh completion failed (attempt %s/%s, purpose=%s): %s — retrying in %.2fs",
                    attempt, attempts, purpose, exc, delay,
                )
                time.sleep(delay)
                continue
            logger.error(
                "Mesh completion failed permanently (purpose=%s, model=%s): %s",
                purpose, target_model, exc,
            )
            break

    raise MeshUnavailableError(
        f"Mesh completion failed for purpose={purpose!r} model={target_model!r}: {last_error}"
    ) from last_error


def call_llm_json(
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    *,
    purpose: str = "structured",
    telemetry: Optional[MeshTelemetry] = None,
    max_tokens: Optional[int] = None,
) -> Any:
    """Run a completion through Mesh and parse the reply as JSON.

    LLMs habitually wrap JSON in prose or ```json fences, so the response is
    coerced with :func:`extract_json` before parsing.

    Args:
        messages: Chat messages; the system prompt should demand JSON only.
        model: Mesh model id.  Defaults to ``MESH_MODEL_REASONING``.
        temperature: Sampling temperature — low by default for determinism.
        purpose: Telemetry label.
        telemetry: Optional collector for this agent run.
        max_tokens: Optional completion cap.

    Returns:
        The parsed JSON value (``dict`` or ``list``).

    Raises:
        MeshUnavailableError: On transport failure, or if no JSON can be parsed.
    """
    raw = call_llm(
        messages,
        model=model,
        temperature=temperature,
        purpose=purpose,
        telemetry=telemetry,
        max_tokens=max_tokens,
    )
    payload = extract_json(raw)
    if payload is None:
        logger.error("Mesh reply for purpose=%s was not valid JSON: %.400s", purpose, raw)
        raise MeshUnavailableError(f"Mesh reply for purpose={purpose!r} was not valid JSON")
    return payload


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Optional[Any]:
    """Best-effort extraction of a JSON value from free-form model output.

    Tries, in order: the whole string, any fenced code block, then the widest
    ``{...}`` / ``[...]`` span in the text.

    Args:
        text: Raw model output.

    Returns:
        The parsed value, or None if nothing parseable was found.
    """
    if not text:
        return None

    candidates: list[str] = [text.strip()]
    candidates.extend(match.strip() for match in _FENCE_RE.findall(text))

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


# --------------------------------------------------------------------------- #
# Embeddings                                                                  #
# --------------------------------------------------------------------------- #

def embed_text(text: str, *, telemetry: Optional[MeshTelemetry] = None) -> list[float]:
    """Embed a single string via Mesh.

    Args:
        text: The document or query to embed.
        telemetry: Optional collector for this agent run.

    Returns:
        The embedding vector.

    Raises:
        MeshUnavailableError: On transport failure or missing key.
    """
    vectors = embed_texts([text], telemetry=telemetry)
    return vectors[0]


def embed_texts(
    texts: list[str],
    *,
    model: Optional[str] = None,
    telemetry: Optional[MeshTelemetry] = None,
    batch_size: int = 64,
) -> list[list[float]]:
    """Embed a batch of strings via Mesh, preserving input order.

    Batching matters for the seed script, which embeds 30+ courses; sending them
    in chunks keeps request bodies small while avoiding 30 round-trips.

    Args:
        texts: Documents to embed (empty strings are replaced with a space so the
            provider does not reject them).
        model: Embedding model id.  Defaults to ``MESH_EMBEDDING_MODEL``.
        telemetry: Optional collector for this agent run.
        batch_size: Maximum inputs per HTTP request.

    Returns:
        One vector per input, in the same order.

    Raises:
        MeshUnavailableError: On transport failure or missing key.
    """
    if not texts:
        return []

    target_model = model or settings.mesh_embedding_model
    cleaned = [(t.strip() or " ") for t in texts]
    vectors: list[list[float]] = []
    attempts = max(1, settings.mesh_max_retries)

    for offset in range(0, len(cleaned), batch_size):
        chunk = cleaned[offset : offset + batch_size]
        last_error: Optional[BaseException] = None

        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            try:
                client = get_mesh_client()
                response = client.embeddings.create(model=target_model, input=chunk)
                latency_ms = (time.perf_counter() - started) * 1000.0

                # Sort defensively: the spec allows out-of-order `index` fields.
                ordered = sorted(response.data, key=lambda item: getattr(item, "index", 0))
                batch_vectors = [list(item.embedding) for item in ordered]
                if len(batch_vectors) != len(chunk):
                    raise MeshError(
                        f"Mesh returned {len(batch_vectors)} embeddings for {len(chunk)} inputs"
                    )

                usage = getattr(response, "usage", None)
                if telemetry is not None:
                    telemetry.record(
                        MeshCallRecord(
                            model=target_model,
                            purpose="embedding",
                            latency_ms=latency_ms,
                            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                        )
                    )
                vectors.extend(batch_vectors)
                break

            except MeshUnavailableError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < attempts and _is_retryable(exc):
                    delay = 0.75 * (2 ** (attempt - 1))
                    logger.warning(
                        "Mesh embedding failed (attempt %s/%s): %s — retrying in %.2fs",
                        attempt, attempts, exc, delay,
                    )
                    time.sleep(delay)
                    continue
                logger.error("Mesh embedding failed permanently: %s", exc)
                raise MeshUnavailableError(f"Mesh embedding failed: {exc}") from exc
        else:  # pragma: no cover - loop always breaks or raises
            raise MeshUnavailableError(f"Mesh embedding failed: {last_error}")

    return vectors


def describe_models() -> dict[str, str]:
    """Return the configured Mesh model routing (for /health and the README)."""
    return {
        "reasoning": settings.mesh_model_reasoning,
        "writer": settings.mesh_model_writer,
        "grader": settings.mesh_model_grader,
        "embedding": settings.mesh_embedding_model,
        "base_url": settings.mesh_base_url,
    }
