from pathlib import Path

import yaml

from embeddings.base import BaseEmbedder
from embeddings.openai import OpenAIEmbedder
from paths import DEFAULT_CONFIG_PATH


def create_embedder(
    provider: str,
    model: str,
    api_key: str | None = None,
) -> BaseEmbedder:
    if provider == "openai":
        return OpenAIEmbedder(model=model, api_key=api_key)

    raise ValueError(f"Unsupported embedding provider: {provider}")


def embedder_from_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    api_key: str | None = None,
) -> BaseEmbedder:
    with open(Path(config_path).resolve()) as f:
        config = yaml.safe_load(f)

    embedding_config = config["embedding"]
    return create_embedder(
        provider=embedding_config["provider"],
        model=embedding_config["model"],
        api_key=api_key,
    )
