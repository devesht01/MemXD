from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(kw_only=True)
class Memory:
    memory_text: str
    source_domain: str
    conflict_links: list[str]
    history: list[dict]
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)
    custom_metadata: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class ExplicitMemory(Memory):
    source: dict
    latent_preference: str | None
    additional_context: list[str] = field(default_factory=list)
    old_memories: list[ExplicitMemory] = field(default_factory=list)
    source_preference_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.source_preference_id = self.memory_id


@dataclass(kw_only=True)
class LatentMemory(Memory):
    """Question row linked to a latent trait (memory_text is the embedded question)."""

    source_preference_ids: list[str]
    latent_preference: str
