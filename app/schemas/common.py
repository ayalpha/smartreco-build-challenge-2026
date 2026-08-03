"""Shared parameter types that tolerate the way real HTML forms submit data.

The problem this solves
-----------------------
A browser submits *every* field in a GET form, including the ones the user left
blank. So the catalog filter form — category chosen, price left empty — sends::

    /catalog?q=&category=Agentic+AI&skill_level=&max_price=

FastAPI parses ``max_price`` as ``Optional[float]``. An empty string is not a
valid float, so the request never reaches the route: it fails validation and the
user gets a raw 422 JSON body instead of their filtered courses. Text params are
unaffected (``""`` is a valid string), which is why only the numeric fields broke.

The fix is a ``BeforeValidator`` that normalises an all-whitespace string to
``None`` *before* coercion runs, so "left blank" means "no filter" — which is
what the user meant.

Note the nesting: the ``ge``/``le`` constraints are attached to the **float**
member of the union, not to the ``Optional``. Applying them at the outer level
makes Pydantic run the comparison against ``None`` and raise
``TypeError: Unable to apply constraint 'ge' to supplied value None``. Real
numbers are still range-checked exactly as before — a negative price is still a
422, only *blank* now means "unset".
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import Form, Query
from pydantic import BeforeValidator, Field
from typing_extensions import Annotated


def empty_str_to_none(value: Any) -> Any:
    """Treat a blank submitted field as "not provided".

    Args:
        value: The raw submitted value.

    Returns:
        ``None`` when the value is a blank/whitespace-only string, otherwise the
        value unchanged (so real numbers still validate normally).
    """
    if isinstance(value, str) and not value.strip():
        return None
    return value


#: A price value, constrained only where a value actually exists.
_Price = Annotated[float, Field(ge=0)]

#: A rating value, constrained only where a value actually exists.
_Rating = Annotated[float, Field(ge=0, le=5)]


#: Optional non-negative price filter that survives an empty form field.
OptionalPriceQuery = Annotated[
    Optional[_Price],
    BeforeValidator(empty_str_to_none),
    Query(description="Maximum price. Blank means no price filter."),
]

#: Optional 0–5 rating submitted from an admin form that may be left blank.
OptionalRatingForm = Annotated[
    Optional[_Rating],
    BeforeValidator(empty_str_to_none),
    Form(description="Rating out of 5. Blank means unrated."),
]
