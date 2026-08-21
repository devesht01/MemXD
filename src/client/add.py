import yaml

from client.prompt_controller import (
    build_latent_preference_prompt,
    build_domain_classification_prompt,
)
from db.controller import ChromaController
from llms.base import BaseLLM
from llms.factory import llm_from_config
from llms.json_schema import (
    LATENT_PREFERENCE_SCHEMA,
    DOMAIN_CLASSIFICATION_SCHEMA,
)
from memory.memory import ExplicitMemory, LatentMemory
from paths import DEFAULT_CONFIG_PATH

with open(DEFAULT_CONFIG_PATH) as f:
    config = yaml.safe_load(f)

domains = list(config["domain-transfer"]["life_domains"].keys())

_LATENT_UNSET = object()


def add_memory(
    message: dict,
    vector_db: ChromaController,
    *,
    latent_preference: str | None | object = _LATENT_UNSET,
) -> None:
    """
    message = {
        "user": str,
        "agent": str,
        "timestamp": ...,
        "custom_metadata": dict,  # optional — copied onto every stored memory
    }

    If latent_preference is passed as None: forced null latent (no LLM, no
    latent rows). If passed as a non-null string: still run the latent LLM for
    questions only; caller latent overwrites the extracted trait.

    Domain classification tags a domain from the config list; the stored statement
    is always the raw user text.
    """
    llm = llm_from_config()

    instructions, prompt = build_domain_classification_prompt(message, domains)
    extracted = llm.generate_json(
        prompt=prompt,
        instructions=instructions,
        json_schema=DOMAIN_CLASSIFICATION_SCHEMA,
        retry_prompt_suffix=(
            "Return JSON with a domain field chosen from the domain list."
        ),
    )
    print(f"\n=== DOMAIN CLASSIFICATION ===\n{extracted}", flush=True)

    domain = extracted["domain"]
    if "user" in message:
        raw_user = message["user"]
    else:
        raw_user = "\n".join(
            m["content"]
            for m in message["messages"]
            if m.get("role") == "user"
        )
    memories = {
        "statements": [
            {
                "statement": raw_user,
                "domain": domain,
            }
        ]
    }
    print(
        f"\n=== DOMAIN EXTRACT ===\n"
        f"using raw user statement with domain={domain!r}\n"
        f"{memories}",
        flush=True,
    )

    custom_metadata = message.get("custom_metadata", {})

    for i, memory in enumerate(memories["statements"], start=1):
        memory["custom_metadata"] = custom_metadata
        memory["source"] = message
        print(
            f"\n=== ADD [{i}] ===\n"
            f"statement: {memory['statement']!r}\n"
            f"domain: {memory['domain']}",
            flush=True,
        )

        # 1. Latent extract (trait + questions), unless caller forced null or
        # provided a latent override (override still runs LLM for questions).
        if latent_preference is _LATENT_UNSET:
            statement_latent, questions = _extract_latent_preference(
                memory["statement"],
                memory["domain"],
                llm,
            )
        elif latent_preference is None:
            statement_latent = None
            questions = []
            print("🍐 Latent preference (provided): None — skip LLM, no questions")
        else:
            _extracted_latent, questions = _extract_latent_preference(
                memory["statement"],
                memory["domain"],
                llm,
            )
            statement_latent = latent_preference
            if not questions:
                raise ValueError(
                    "non-null provided latent_preference requires non-empty "
                    "LLM questions"
                )
            print(
                f"🍐 Latent preference (provided, overwrites extracted "
                f"{_extracted_latent!r}): {statement_latent!r}",
                flush=True,
            )
            print(f"🍐 Questions (from LLM): {questions!r}", flush=True)

        _store_new(memory, statement_latent, questions, vector_db)


def _extract_latent_preference(
    statement: str,
    source_domain: str,
    llm: BaseLLM,
) -> tuple[str | None, list[str]]:
    instructions, prompt = build_latent_preference_prompt(statement, source_domain)
    parsed = llm.generate_json(
        prompt=prompt,
        instructions=instructions,
        json_schema=LATENT_PREFERENCE_SCHEMA,
    )
    latent_preference = parsed["latent_preference"]
    questions = list(parsed["questions"])
    if latent_preference is None:
        if questions:
            raise ValueError(
                "null latent_preference requires empty questions, "
                f"got {questions!r}"
            )
        print("🍐 Latent preference: None", flush=True)
        print("🍐 Questions: []", flush=True)
        return None, []
    if not questions:
        raise ValueError(
            "non-null latent_preference requires non-empty questions"
        )
    print(f"🍐 Latent preference: {latent_preference}", flush=True)
    print(f"🍐 Questions: {questions}", flush=True)
    return latent_preference, questions


def _build_explicit_memory(
    memory: dict,
    *,
    latent_preference: str | None,
    conflict_links: list[str] | None = None,
    history: list[dict] | None = None,
    old_memories: list[ExplicitMemory] | None = None,
) -> ExplicitMemory:
    return ExplicitMemory(
        memory_text=memory["statement"],
        source_domain=memory["domain"],
        conflict_links=conflict_links if conflict_links is not None else [],
        history=history if history is not None else [],
        source=memory["source"],
        latent_preference=latent_preference,
        additional_context=[],
        old_memories=old_memories if old_memories is not None else [],
        custom_metadata=memory.get("custom_metadata", {}),
    )


def _build_latent_memory(
    *,
    question: str,
    latent_preference: str,
    source_domain: str,
    source_preference_id: str,
    custom_metadata: dict | None = None,
) -> LatentMemory:
    return LatentMemory(
        memory_text=question,
        source_domain=source_domain,
        conflict_links=[],
        history=[],
        source_preference_ids=[source_preference_id],
        latent_preference=latent_preference,
        custom_metadata=custom_metadata if custom_metadata is not None else {},
    )


def _store_new(
    memory: dict,
    latent_preference: str | None,
    questions: list[str],
    vector_db: ChromaController,
) -> ExplicitMemory:
    """Store explicit; if latent non-null, store one LatentMemory per question."""
    print(f"🍎 Storing explicit: {memory['statement']}")
    explicit = _build_explicit_memory(memory, latent_preference=latent_preference)
    vector_db.add_explicit(explicit)
    if latent_preference is not None:
        print(
            f"🌿 Storing {len(questions)} latent question row(s) for "
            f"{latent_preference!r}",
            flush=True,
        )
        for question in questions:
            vector_db.add_latent(
                _build_latent_memory(
                    question=question,
                    latent_preference=latent_preference,
                    source_domain=memory["domain"],
                    source_preference_id=explicit.memory_id,
                    custom_metadata=memory.get("custom_metadata", {}),
                )
            )
    else:
        print(f"🍒 Not transferable: {memory['statement']}\n")
    return explicit
