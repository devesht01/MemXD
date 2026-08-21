import json
import re
from typing import Any

from openai import APITimeoutError, NotFoundError, OpenAI

from llms.base import BaseLLM
from llms.errors import LLMFailed
from llms.json_schema import validate_json

# Per-attempt API wait budget (not a sleep between retries).
# Attempt 1 waits up to 90s, then 120s, 150s, 180s, 210s (5 attempts).
_TIMEOUTS_SECONDS = (90, 120, 150, 180, 210)

# Optional markdown fences around otherwise-valid JSON, e.g. ```json ... ```
_JSON_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL | re.IGNORECASE,
)


def _unwrap_json_payload(text: str) -> str:
    cleaned = text.strip()
    match = _JSON_FENCE_RE.match(cleaned)
    if match:
        return match.group(1).strip()
    return cleaned


def _loads_json_maybe_fix_extra_brace(text: str) -> Any:
    """Parse JSON; if it fails on the known trailing extra-`}` pattern, strip it.

    Bad:  ...,"domain":"Creativity"}}]}
    Good: ...,"domain":"Creativity"}]}
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = text
        if "}}]}" in fixed:
            fixed = fixed.replace("}}]}", "}]}", 1)
            return json.loads(fixed)
        raise


class OpenAILLM(BaseLLM):
    def __init__(self, model: str, temperature: float, api_key: str | None = None):
        super().__init__(model, temperature)
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        instructions: str | None = None,
        *,
        expect_json: bool = False,
        json_schema: Any | None = None,
        retry_prompt_suffix: str | None = None,
    ) -> str:
        if json_schema is not None and not expect_json:
            raise ValueError("json_schema requires expect_json=True")

        last_error: Exception | None = None
        total = len(_TIMEOUTS_SECONDS)
        for attempt, timeout in enumerate(_TIMEOUTS_SECONDS, start=1):
            raw: str | None = None
            attempt_prompt = prompt
            if attempt > 1 and retry_prompt_suffix:
                attempt_prompt = f"{prompt}\n\n{retry_prompt_suffix}"
            try:
                text = self._generate_once(
                    prompt=attempt_prompt,
                    instructions=instructions,
                    timeout=timeout,
                )
                raw = text
                if not text or not str(text).strip():
                    raise LLMFailed("empty response")
                if expect_json:
                    text = _unwrap_json_payload(text)
                    parsed = _loads_json_maybe_fix_extra_brace(text)
                    if json_schema is not None:
                        validate_json(parsed, json_schema)
                    # Re-serialize so callers get consistent canonical JSON text.
                    text = json.dumps(parsed)
                return text
            except LLMFailed as exc:
                last_error = exc
                reason = str(exc)
            except json.JSONDecodeError as exc:
                last_error = LLMFailed(f"malformed json: {exc}")
                reason = str(last_error)
            except APITimeoutError as exc:
                last_error = LLMFailed(f"timeout after {timeout}s: {exc}")
                reason = str(last_error)
            except NotFoundError as exc:
                # Transient 404 when the model is temporarily unavailable, e.g.
                # "The model is not available. Please try your request again..."
                last_error = LLMFailed(f"model not available: {exc}")
                reason = str(last_error)

            will_retry = attempt < total
            if will_retry:
                next_timeout = _TIMEOUTS_SECONDS[attempt]
                print(
                    f"LLM failed on attempt {attempt}/{total}: {reason}. "
                    f"Retrying with timeout budget={next_timeout}s. "
                    f"raw={raw!r}",
                    flush=True,
                )
            else:
                print(
                    f"LLM failed on attempt {attempt}/{total}: {reason}. "
                    f"No retries left. raw={raw!r}",
                    flush=True,
                )

        raise LLMFailed(
            f"LLM failed after {total} attempts: {last_error}"
        ) from last_error

    def _generate_once(
        self,
        prompt: str,
        instructions: str | None,
        timeout: float,
    ) -> str:
        kwargs = {
            "model": self.model,
            "input": prompt,
            "temperature": self.temperature,
        }
        if instructions is not None:
            kwargs["instructions"] = instructions

        response = self.client.with_options(timeout=timeout).responses.create(**kwargs)
        return response.output_text
