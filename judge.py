import json

from openai import OpenAI

MODEL = "gpt-5-2025-08-07"
REASONING_EFFORT = "minimal"
MAX_OUTPUT_TOKENS = 4096

JUDGE_SYSTEM_PROMPT = "You are a strict benchmark judge."


class Judge:
    def __init__(self):
        self.client = OpenAI()

    def run(self, prompt: str) -> dict:
        response = self.client.responses.create(
            model=MODEL,
            reasoning={"effort": REASONING_EFFORT},
            max_output_tokens=MAX_OUTPUT_TOKENS,
            instructions=JUDGE_SYSTEM_PROMPT,
            input=prompt,
        )
        return self._parse_response(response.output_text)

    @staticmethod
    def _parse_response(response: str) -> dict:
        text = response.strip()
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        result = json.loads(text)
        if "score" not in result:
            raise ValueError(f"Judge response missing score: {result}")
        if "explanation" not in result:
            raise ValueError(f"Judge response missing explanation: {result}")
        return {
            "score": float(result["score"]),
            "explanation": result["explanation"],
        }
