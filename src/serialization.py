"""
Convert database and numpy values into something json.dumps accepts.

Shared by the artifact channel, the Redis cache and the HTTP layer so that a
chart handed to the browser, a payload stored in Redis and a REST response all
see the same types.
"""

from __future__ import annotations

import datetime as dt
import math
from decimal import Decimal
from typing import Any


def to_jsonable(value: Any) -> Any:
    """
    Convert database and numpy scalars into something json.dumps accepts.

    Postgres hands back Decimal for numeric columns and date/datetime for
    temporal ones, and numpy types leak out of pandas. NaN and infinity become
    None because they are not valid JSON and every downstream consumer, from
    json.dumps to Plotly, treats them as missing anyway.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Decimal):
        number = float(value)
        return number if math.isfinite(number) else None

    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()

    if isinstance(value, dt.timedelta):
        return value.total_seconds()

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    # numpy scalars and anything else that knows how to become a Python object.
    if hasattr(value, "item") and callable(value.item):
        try:
            return to_jsonable(value.item())
        except (ValueError, TypeError):
            pass

    return str(value)
