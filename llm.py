from abc import ABC, abstractmethod
import json
import urllib.request

from openai import OpenAI as OpenAIClient


class LLM(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> str:
        pass


class OpenAI(LLM):
    def __init__(self, model: str):
        super().__init__(model)
        self.client = OpenAIClient()

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_output_tokens is not None:
            kwargs["max_completion_tokens"] = max_output_tokens
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


class Ollama(LLM):
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        super().__init__(model)
        self.host = host

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> str:
        options = {"temperature": temperature}
        if max_output_tokens is not None:
            options["num_predict"] = max_output_tokens
        system_parts = []
        prompt_parts = []
        for message in messages:
            if message["role"] == "system":
                system_parts.append(message["content"])
            else:
                prompt_parts.append(message["content"])
        payload = {
            "model": self.model,
            "prompt": "\n\n".join(prompt_parts),
            "stream": False,
            "options": options,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read())
        content = data["response"]
        if not content:
            raise RuntimeError("Ollama returned a response without text content.")
        return content
