"""ORM model package.

Importing this package registers every mapper on ``Base.metadata``, which is
what :func:`app.database.init_db` and Alembic's autogenerate rely on.
"""

from __future__ import annotations

from app.models.email_digest import EmailDigest
from app.models.event import Event, EventType
from app.models.product import Product
from app.models.recommendation import Recommendation
from app.models.user import User, UserRole

__all__ = [
    "EmailDigest",
    "Event",
    "EventType",
    "Product",
    "Recommendation",
    "User",
    "UserRole",
]
