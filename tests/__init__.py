"""SmartReco test-suite.

The suite runs with **no external services and no API keys**: SQLite replaces
PostgreSQL, the embedded Qdrant fallback replaces the vector server, an
in-process dict replaces Redis, and every agent node takes its graceful
degradation path because ``MESH_API_KEY`` is unset.  That means CI exercises the
real compiled LangGraph state machine end to end rather than a mock of it.
"""
