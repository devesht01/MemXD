from pathlib import Path
from datetime import datetime
import shutil

import json

import chromadb
import yaml
from chromadb.api.models.Collection import Collection

from db.utils import strip_user_prefix
from embeddings.factory import embedder_from_config
from memory.memory import (
    ExplicitMemory,
    LatentMemory,
)
from paths import DEFAULT_CHROMA_PATH, DEFAULT_CONFIG_PATH


def _explicit_to_dict(memory: ExplicitMemory) -> dict:
    return {
        "memory_id": memory.memory_id,
        "memory_text": memory.memory_text,
        "source_domain": memory.source_domain,
        "conflict_links": memory.conflict_links,
        "history": memory.history,
        "created_at": memory.created_at.isoformat(),
        "updated_at": memory.updated_at.isoformat(),
        "source": memory.source,
        "latent_preference": memory.latent_preference,
        "additional_context": memory.additional_context,
        "old_memories": [_explicit_to_dict(m) for m in memory.old_memories],
        "custom_metadata": memory.custom_metadata,
    }


def _dict_to_explicit(data: dict) -> ExplicitMemory:
    return ExplicitMemory(
        memory_text=data["memory_text"],
        source_domain=data["source_domain"],
        conflict_links=list(data.get("conflict_links", [])),
        history=list(data.get("history", [])),
        memory_id=data["memory_id"],
        created_at=datetime.fromisoformat(data["created_at"]),
        updated_at=datetime.fromisoformat(data["updated_at"]),
        source=data["source"],
        latent_preference=data.get("latent_preference"),
        additional_context=list(data.get("additional_context", [])),
        old_memories=[
            _dict_to_explicit(m) for m in data.get("old_memories", [])
        ],
        custom_metadata=dict(data.get("custom_metadata", {})),
    )


def _load_additional_context(metadata: dict) -> list[str]:
    raw = metadata.get("additional_context", "[]")
    if isinstance(raw, list):
        return list(raw)
    if not isinstance(raw, str):
        raise ValueError(
            f"additional_context must be JSON list or str, got {type(raw)!r}"
        )
    if not raw:
        return []
    # Legacy free-text additional_context → []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return parsed
    return []


def _load_old_memories(metadata: dict) -> list[ExplicitMemory]:
    raw = metadata.get("old_memories", "[]")
    if not raw:
        return []
    if isinstance(raw, list):
        return [_dict_to_explicit(item) for item in raw]
    parsed = json.loads(raw)
    return [_dict_to_explicit(item) for item in parsed]


class ChromaController:
    def __init__(
        self,
        user_id: str,
        path: str | Path = DEFAULT_CHROMA_PATH,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        if not user_id:
            raise ValueError("user_id is required")
        self.user_id = user_id
        self.chroma_root = Path(path).resolve()
        self.path = str(self.chroma_root / user_id)
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config_path).resolve()
        self.client = chromadb.PersistentClient(path=self.path)
        with open(self.config_path) as f:
            full_config = yaml.safe_load(f)
        memory_config = full_config["memory"]
        self.collections = dict(memory_config["collections"])
        self.embedder = embedder_from_config(self.config_path)

    def embed_for_retrieval(self, text: str) -> list[float]:
        """Embed text with leading 'User'/'The user' removed so it cannot bias ranks."""
        return self.embedder.embed_query(strip_user_prefix(text))

    def collections_exist(self) -> bool:
        existing = set(self.list_collections())
        return all(name in existing for name in self.collections.values())

    def create_collections(self) -> None:
        existing = set(self.list_collections())
        for name in self.collections.values():
            if name not in existing:
                self.add_collection(name)

    def reset_user(self, user_id: str) -> None:
        if user_id == self.user_id:
            existing = set(self.list_collections())
            for name in self.collections.values():
                if name in existing:
                    self.delete_collection(name)
            return
        other = self.chroma_root / user_id
        if other.exists():
            shutil.rmtree(other)

    def reset(self) -> None:
        self.reset_user(self.user_id)
        self.create_collections()

    def add_collection(self, name: str) -> Collection:
        return self.client.create_collection(name=name)

    def get_collection(self, name: str) -> Collection:
        return self.client.get_collection(name=name)

    def list_collections(self) -> list[str]:
        return [collection.name for collection in self.client.list_collections()]

    def delete_collection(self, name: str) -> None:
        self.client.delete_collection(name=name)

    def add_explicit(self, memory: ExplicitMemory) -> None:
        embedding = self.embed_for_retrieval(memory.memory_text)
        metadata = {
            "source_domain": memory.source_domain,
            "source_preference_id": memory.source_preference_id,
            "conflict_links": json.dumps(memory.conflict_links),
            "history": json.dumps(memory.history),
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "source": json.dumps(memory.source),
            "latent_preference": (
                memory.latent_preference
                if memory.latent_preference is not None
                else ""
            ),
            "additional_context": json.dumps(memory.additional_context),
            "old_memories": json.dumps(
                [_explicit_to_dict(m) for m in memory.old_memories]
            ),
            "custom_metadata": json.dumps(memory.custom_metadata),
        }
        self._add(
            collection=self.collections["explicit"],
            ids=[memory.memory_id],
            documents=[memory.memory_text],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def add_latent(self, memory: LatentMemory) -> None:
        embedding = self.embed_for_retrieval(memory.memory_text)
        metadata = {
            "source_domain": memory.source_domain,
            "source_preference_ids": json.dumps(memory.source_preference_ids),
            "conflict_links": json.dumps(memory.conflict_links),
            "history": json.dumps(memory.history),
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "latent_preference": memory.latent_preference,
            "custom_metadata": json.dumps(memory.custom_metadata),
        }
        self._add(
            collection=self.collections["latent"],
            ids=[memory.memory_id],
            documents=[memory.memory_text],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def _add(
        self,
        collection: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict] | None = None,
    ) -> None:
        col = self.get_collection(collection)
        col.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def list_memories_for_query(self, kind: str) -> list[dict]:
        """Load every memory of one kind with embedding for Client.query scoring.

        kind must be "explicit" or "latent".
        """
        if kind not in ("explicit", "latent"):
            raise ValueError(f"kind must be 'explicit' or 'latent', got {kind!r}")
        to_memory = (
            self._to_explicit_memory if kind == "explicit" else self._to_latent_memory
        )
        col = self.get_collection(self.collections[kind])
        if col.count() == 0:
            return []
        data = col.get(include=["documents", "metadatas", "embeddings"])
        embeddings = data["embeddings"]
        pool: list[dict] = []
        for memory_id, document, metadata, embedding in zip(
            data["ids"],
            data["documents"],
            data["metadatas"],
            embeddings,
        ):
            if embedding is None:
                embedding = self.embed_for_retrieval(document)
            pool.append(
                {
                    "kind": kind,
                    "memory_id": memory_id,
                    "text": document,
                    "embedding": embedding,
                    "memory": to_memory(memory_id, document, metadata),
                }
            )
        return pool

    def list_all_memories(self) -> dict[str, list]:
        """Return every stored memory object for both collections (no embeddings)."""
        result: dict[str, list] = {"explicit": [], "latent": []}
        for kind, to_memory in (
            ("explicit", self._to_explicit_memory),
            ("latent", self._to_latent_memory),
        ):
            col = self.get_collection(self.collections[kind])
            if col.count() == 0:
                continue
            data = col.get(include=["documents", "metadatas"])
            for memory_id, document, metadata in zip(
                data["ids"],
                data["documents"],
                data["metadatas"],
            ):
                result[kind].append(to_memory(memory_id, document, metadata))
        return result

    def _to_explicit_memory(
        self,
        memory_id: str,
        document: str,
        metadata: dict,
    ) -> ExplicitMemory:
        latent_preference = metadata["latent_preference"]
        return ExplicitMemory(
            memory_text=document,
            source_domain=metadata["source_domain"],
            conflict_links=json.loads(metadata.get("conflict_links", "[]")),
            history=json.loads(metadata.get("history", "[]")),
            memory_id=memory_id,
            created_at=datetime.fromisoformat(metadata["created_at"]),
            updated_at=datetime.fromisoformat(metadata["updated_at"]),
            source=json.loads(metadata["source"]),
            latent_preference=latent_preference if latent_preference else None,
            additional_context=_load_additional_context(metadata),
            old_memories=_load_old_memories(metadata),
            custom_metadata=json.loads(metadata["custom_metadata"])
            if metadata.get("custom_metadata")
            else {},
        )

    def _to_latent_memory(
        self,
        memory_id: str,
        document: str,
        metadata: dict,
    ) -> LatentMemory:
        return LatentMemory(
            memory_text=document,
            source_domain=metadata["source_domain"],
            conflict_links=json.loads(metadata.get("conflict_links", "[]")),
            history=json.loads(metadata.get("history", "[]")),
            memory_id=memory_id,
            created_at=datetime.fromisoformat(metadata["created_at"]),
            updated_at=datetime.fromisoformat(metadata["updated_at"]),
            source_preference_ids=json.loads(
                metadata.get("source_preference_ids", "[]")
            ),
            latent_preference=metadata["latent_preference"],
            custom_metadata=json.loads(metadata["custom_metadata"])
            if metadata.get("custom_metadata")
            else {},
        )

    def get(self, collection: str, ids: list[str]) -> dict:
        col = self.get_collection(collection)
        return col.get(ids=ids, include=["documents", "metadatas"])

    def update_metadata(
        self,
        collection: str,
        ids: list[str],
        metadatas: list[dict],
        documents: list[str] | None = None,
    ) -> None:
        col = self.get_collection(collection)
        if documents is not None:
            embeddings = [self.embed_for_retrieval(doc) for doc in documents]
            col.update(
                ids=ids,
                metadatas=metadatas,
                documents=documents,
                embeddings=embeddings,
            )
        else:
            col.update(ids=ids, metadatas=metadatas)

    def delete(self, collection: str, ids: list[str]) -> None:
        col = self.get_collection(collection)
        col.delete(ids=ids)
