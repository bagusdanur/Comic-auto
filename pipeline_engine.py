"""Version gate for production pipeline implementations.

``legacy`` names the currently deployed pipeline.  Future engines must be
registered here only after their implementation and tests are available; this
prevents an environment typo from silently changing production output.
"""

from __future__ import annotations

import os


ENGINE_LEGACY = "legacy"
ENGINE_V3 = "v3"
ENGINE_V4 = "v4"
ENGINE_V5 = "v5"
ENGINE_SHADOW = "shadow"
KNOWN_ENGINES = (ENGINE_LEGACY, ENGINE_V3, ENGINE_V4, ENGINE_V5, ENGINE_SHADOW)
IMPLEMENTED_ENGINES = (ENGINE_LEGACY, ENGINE_V3, ENGINE_V4, ENGINE_V5)


def requested_engine(value: str | None = None) -> str:
    """Return a normalized, known engine name."""
    engine = (value if value is not None else os.getenv("PIPELINE_ENGINE", ENGINE_LEGACY))
    engine = str(engine).strip().casefold()
    if engine not in KNOWN_ENGINES:
        choices = ", ".join(KNOWN_ENGINES)
        raise ValueError(f"Unknown pipeline engine {engine!r}; expected one of: {choices}")
    return engine


def require_implemented(engine: str) -> str:
    """Fail closed when a reserved engine has not been implemented yet."""
    normalized = requested_engine(engine)
    if normalized not in IMPLEMENTED_ENGINES:
        raise RuntimeError(
            f"Pipeline engine {normalized!r} is reserved but not implemented; "
            f"use {ENGINE_LEGACY!r}"
        )
    return normalized
