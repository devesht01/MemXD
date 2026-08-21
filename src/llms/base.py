from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def generate(
        self,
        prompt: str,
        instructions: str | None = None,
        *,
        expect_json: bool = False,
        json_schema: Any | None = None,
        retry_prompt_suffix: str | None = None,
    ) -> str:
        """Generate a text response from the model."""
        pass

    def generate_json(
        self,
        prompt: str,
        instructions: str | None = None,
        *,
        json_schema: Any,
        retry_prompt_suffix: str | None = None,
    ) -> Any:
        """Generate, parse, and shape-validate JSON (retries inside generate)."""
        import json

        text = self.generate(
            prompt=prompt,
            instructions=instructions,
            expect_json=True,
            json_schema=json_schema,
            retry_prompt_suffix=retry_prompt_suffix,
        )
        return json.loads(text)
