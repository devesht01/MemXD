from pathlib import Path

import yaml

from llms.base import BaseLLM
from llms.openai import OpenAILLM
from paths import DEFAULT_CONFIG_PATH


def create_llm(
    provider: str,
    model: str,
    temperature: float,
    api_key: str | None = None,
) -> BaseLLM:
    if provider == "openai":
        return OpenAILLM(model=model, temperature=temperature, api_key=api_key)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def llm_from_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    api_key: str | None = None,
) -> BaseLLM:
    with open(Path(config_path).resolve()) as f:
        config = yaml.safe_load(f)

    extraction_config = config["extraction_model"]
    return create_llm(
        provider=extraction_config["provider"],
        model=extraction_config["model"],
        temperature=extraction_config["temperature"],
        api_key=api_key,
    )
