"""Pydantic request/response models for every JSON endpoint."""

from __future__ import annotations

from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserOut
from app.schemas.event import EventBatchIn, EventBatchResponse, EventIn, EventOut
from app.schemas.product import (
    ProductCreate,
    ProductListResponse,
    ProductOut,
    ProductUpdate,
)
from app.schemas.recommendation import (
    InterestSignalOut,
    RecommendationOut,
    RecommendationProductOut,
    RecommendationStatusOut,
    TriggerDecisionOut,
)

__all__ = [
    "EventBatchIn",
    "EventBatchResponse",
    "EventIn",
    "EventOut",
    "InterestSignalOut",
    "LoginRequest",
    "ProductCreate",
    "ProductListResponse",
    "ProductOut",
    "ProductUpdate",
    "RecommendationOut",
    "RecommendationProductOut",
    "RecommendationStatusOut",
    "RegisterRequest",
    "TokenResponse",
    "TriggerDecisionOut",
    "UserOut",
]
