"""Shared numerical helpers."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np


@contextmanager
def quiet_linear_algebra() -> Iterator[None]:
    """
    Silence spurious floating-point warnings raised from inside BLAS.

    Some BLAS builds (notably Accelerate on Apple silicon) report divide,
    overflow and invalid conditions from `a @ b` even when both operands are
    finite and the result is correct. Callers check their inputs for finiteness
    before entering this block, so what is suppressed here is noise from the
    backend rather than a signal about the data.
    """
    with (
        np.errstate(divide="ignore", over="ignore", invalid="ignore"),
        warnings.catch_warnings(),
    ):
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul.*",
            category=RuntimeWarning,
        )
        yield
