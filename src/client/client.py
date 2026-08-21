from pathlib import Path

from db.controller import ChromaController
from paths import DEFAULT_CHROMA_PATH, DEFAULT_CONFIG_PATH

from . import add as add_module
from . import retrieve as retrieve_module


class Client:
    def __init__(
        self,
        user_id: str,
        path: str | Path = DEFAULT_CHROMA_PATH,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        self.user_id = user_id
        self.vector_db = ChromaController(
            user_id=user_id,
            path=path,
            config_path=config_path,
        )
        if not self.vector_db.collections_exist():
            self.vector_db.create_collections()

    def add(self, message: dict):
        return add_module.add_memory(message, self.vector_db)

    def add_with_latent(self, message: dict, latent_preference: str | None):
        """Same as add(), with caller-provided latent preference.

        None: forced null (no latent LLM, no latent rows).
        Non-null: latent LLM still runs for questions; provided latent overwrites
        the extracted trait text.
        """
        return add_module.add_memory(
            message,
            self.vector_db,
            latent_preference=latent_preference,
        )

    def query(self, user_query: str) -> tuple[str, dict]:
        return retrieve_module.query(user_query, self.vector_db)

    def retrieve(self, query: str) -> tuple[str, dict]:
        return retrieve_module.retrieve(query, self.vector_db)

    def retrieve_context(self, query: str) -> tuple[str, dict]:
        return self.retrieve(query)

    def list_all(self) -> dict:
        """Return every explicit + latent memory with all fields."""
        return self.vector_db.list_all_memories()

    def reset_user(self, user_id: str) -> None:
        self.vector_db.reset_user(user_id)

    def reset(self) -> None:
        self.vector_db.reset()
