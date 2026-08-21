from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return their vector representations."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text and return its vector representation."""
        pass
