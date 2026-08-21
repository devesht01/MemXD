"""Lightweight JSON shape checks for LLM outputs (required keys + types).

Schemas use a small DSL:
  - Python types / (type, ...) unions for scalars
  - dict  → object with those required keys (extra keys allowed)
  - [X]   → list whose items match X
"""

from __future__ import annotations

from typing import Any

from llms.errors import LLMFailed

# --- schemas used by prod call sites ---

DOMAIN_CLASSIFICATION_SCHEMA: dict = {
    "domain": str,
}

LATENT_PREFERENCE_SCHEMA: dict = {
    "latent_preference": (str, type(None)),
    "questions": [str],
}


PROBE_GENERATION_SCHEMA: dict = {
    "trait": [str],
    "direct": [str],
}


def validate_json(data: Any, schema: Any, *, path: str = "$") -> None:
    """Raise LLMFailed if data does not match schema."""
    if isinstance(schema, tuple):
        if not isinstance(data, schema):
            names = " | ".join(
                "null" if t is type(None) else t.__name__ for t in schema
            )
            raise LLMFailed(
                f"json schema: {path} expected {names}, "
                f"got {type(data).__name__}"
            )
        return

    if schema is type(None):
        if data is not None:
            raise LLMFailed(
                f"json schema: {path} expected null, got {type(data).__name__}"
            )
        return

    if schema in (str, int, float, bool):
        if not isinstance(data, schema):
            raise LLMFailed(
                f"json schema: {path} expected {schema.__name__}, "
                f"got {type(data).__name__}"
            )
        return

    if isinstance(schema, dict):
        if not isinstance(data, dict):
            raise LLMFailed(
                f"json schema: {path} expected object, got {type(data).__name__}"
            )
        for key, sub_schema in schema.items():
            if key not in data:
                raise LLMFailed(f"json schema: {path} missing required key {key!r}")
            validate_json(data[key], sub_schema, path=f"{path}.{key}")
        return

    if isinstance(schema, list):
        if len(schema) != 1:
            raise ValueError("list schemas must contain exactly one item schema")
        if not isinstance(data, list):
            raise LLMFailed(
                f"json schema: {path} expected array, got {type(data).__name__}"
            )
        item_schema = schema[0]
        for i, item in enumerate(data):
            validate_json(item, item_schema, path=f"{path}[{i}]")
        return

    raise ValueError(f"unsupported schema node at {path}: {schema!r}")
