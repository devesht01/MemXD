from pathlib import Path
import json
#
import yaml

from db.controller import ChromaController
from paths import DEFAULT_CONFIG_PATH

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _read_prompt(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")


def get_prompt_domain_classification() -> str:
    return _read_prompt("domain_classification.txt")


def get_prompt_latent_preference() -> str:
    return _read_prompt("latent_preference.txt")


def get_prompt_probe_generation() -> str:
    return _read_prompt("probe_generation.txt")


# Kept for existing imports in other modules.
LATENT_PREFERENCE_SYSTEM_PROMPT = get_prompt_latent_preference()

_AGENT_CONTEXT_PREFIX = (
    "[CONTEXT ONLY - do NOT extract preferences SOLEY from this section] Agent said: "
)


def _conversation_from_message(message: dict) -> list[dict]:
    """Resolve message dict → list of {role, content}.

    Preferred: message["messages"] = [{"role", "content"}, ...]
    Fallback: single-turn {"user", "agent"}.
    """
    if message.get("messages"):
        return list(message["messages"])
    conversation = [{"role": "user", "content": message["user"]}]
    agent = message.get("agent")
    if agent:
        conversation.append({"role": "assistant", "content": agent})
    return conversation


def _format_domain_classification_conversation(conversation: list[dict]) -> str:
    """Sequential user-prompt lines (b0a10ad-compatible for one user+agent).

    Walk messages in order — no pairing:
      user  → User said: {content}
      agent/assistant → [CONTEXT ONLY …] Agent said: {content}
    """
    lines: list[str] = []
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            lines.append(f"User said: {content}")
        elif role in ("assistant", "agent"):
            lines.append(f"{_AGENT_CONTEXT_PREFIX}{content}")
        else:
            raise ValueError(
                f"unsupported role for domain classification prompt: {role!r}"
            )
    return "\n".join(lines)


def build_domain_classification_prompt(messages: dict, domains: list[str]) -> tuple[str, str]:
    domain_str = ", ".join(domains)
    system = get_prompt_domain_classification().replace("{domains}", domain_str)
    prompt = _format_domain_classification_conversation(
        _conversation_from_message(messages)
    )
    return system, prompt


def build_probe_generation_prompt(query: str) -> tuple[str, str]:
    with open(DEFAULT_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    query_config = config["query"]
    # Template uses {{ / }} around the JSON example (format-style escapes).
    instructions = (
        get_prompt_probe_generation()
        .replace("{n_direct}", str(query_config["n_direct"]))
        .replace("{n_trait}", str(query_config["n_trait"]))
        .replace("{query}", query)
        .replace("{{", "{")
        .replace("}}", "}")
    )
    return instructions, f'Query: "{query}"'



def build_latent_preference_prompt(
    statement: str,
    source_domain: str,
) -> tuple[str, str]:
    with open(DEFAULT_CONFIG_PATH) as f:
        extraction_questions = yaml.safe_load(f)["memory"]["extraction_questions"]
    instructions = get_prompt_latent_preference().replace(
        "{extraction_questions}",
        str(extraction_questions),
    )
    prompt = (
        f'Is this transferable across life domains? If yes, extract the trait.\n\n'
        f'Source domain: {source_domain}\n'
        f'Statement: "{statement}"'
    )
    return instructions, prompt


def format_conflict_link(memory_id: str, vector_db: ChromaController) -> str:
    for kind in ("explicit", "latent"):
        result = vector_db.get(
            collection=vector_db.collections[kind],
            ids=[memory_id],
        )
        if not result["ids"]:
            continue
        content = result["documents"][0]
        metadata = result["metadatas"][0]
        if kind == "latent":
            content = metadata["latent_preference"]
        return (
            f'{content} [updated at: {metadata["updated_at"]}] '
            f'[domain: {metadata["source_domain"]}]'
        )
    raise ValueError(
        f"conflict link memory_id {memory_id!r} not found in explicit or latent "
        f"collection"
    )


def _resolve_additional_context_id(
    memory_id: str,
    vector_db: ChromaController,
) -> str:
    result = vector_db.get(
        collection=vector_db.collections["explicit"],
        ids=[memory_id],
    )
    if not result["ids"]:
        raise ValueError(
            f"additional_context memory_id {memory_id!r} not found in "
            f"explicit collection"
        )
    return result["documents"][0]


def format_memories_for_agent(
    explicit_memories: list,
    latent_memories: list,
    vector_db: ChromaController,
) -> str:
    lines = [
        "You are an personalized, empathetic assistant that answers the user based on the provided context. Your task is to provide accurate and personalized answers to the questions by leveraging the information given in the memories, especially the latent ones.",
        "You MUST UTILIZE this memory context in order to PERSONALIZE responses -- no exceptions.",
    ]

    if explicit_memories:
        lines.append("")
        lines.append("[User statements]")
        lines.append("These are statements that the user has stated.")
        for m in explicit_memories:
            entry = f"- {m.memory_text}"
            if m.additional_context:
                resolved = [
                    _resolve_additional_context_id(cid, vector_db)
                    for cid in m.additional_context
                ]
                entry += (
                    "\n  Additional context: "
                    + "; ".join(resolved)
                )
            if m.old_memories:
                older_bits = []
                for old in m.old_memories:
                    older_bits.append(
                        f'"{old.memory_text}" '
                        f"(updated {old.updated_at.isoformat()}, "
                        f"domain: {old.source_domain})"
                    )
                entry += (
                    "\n  Older representation attached for additional context: "
                    + "; ".join(older_bits)
                )
            entry += f"\n  Domain: {m.source_domain}"
            if m.history:
                history_bits = []
                for prev in m.history:
                    prev_text = prev.get("memory_text", "")
                    prev_updated = prev.get("updated_at", "")
                    history_bits.append(f'"{prev_text}" (updated {prev_updated})')
                entry += f"\n  History: {'; '.join(history_bits)}"
            if m.conflict_links:
                conflicts = [
                    format_conflict_link(cid, vector_db) for cid in m.conflict_links
                ]
                entry += f"\n  Conflicts with: {'; '.join(conflicts)}"
            entry += f"\n  Updated: {m.updated_at.isoformat()}"
            entry += f"\n  Created: {m.created_at.isoformat()}"
            lines.append(entry)

    if latent_memories:
        lines.append("")
        lines.append(
            "[LATENT: Behavioral patterns: traits of the user that apply for personalization. These are VERY important]"
        )
        lines.append(
            "These preferences are high-confidence behavioral patterns extracted from explicit user utterances in order to give personalized responses across life domains. Base your recommendations primarily on these preferences. They reveal behavioral insights about the user and should actively guide your recommendations, especially when the user is making a decision. You are given both the underlying behavioral preference, which should be used to personalize recommendations across domains, and the explicit user utterance that led to the extraction of the preference, which gives additional context about the underlying trait. Apply these preferences even when the current topic differs from the original context -- these preferences are MEANT to transfer across domains. Please utilize these preferences without fear to give the BEST responses."
        )
        for m in latent_memories:
            entry = f"- {m.latent_preference}"
            if not m.source_preference_ids:
                raise ValueError(
                    f"latent {m.memory_id!r} has empty source_preference_ids; "
                    "cannot show Underlying preference extracted from"
                )
            utterance = _resolve_additional_context_id(
                m.source_preference_ids[0],
                vector_db,
            )
            entry += f"\n  Underlying preference extracted from: {utterance}"
            if m.history:
                history_bits = []
                for prev in m.history:
                    prev_text = prev.get("memory_text", "")
                    prev_updated = prev.get("updated_at", "")
                    history_bits.append(f'"{prev_text}" (updated {prev_updated})')
                entry += f"\n  History: {'; '.join(history_bits)}"
            if m.conflict_links:
                conflicts = [
                    format_conflict_link(cid, vector_db) for cid in m.conflict_links
                ]
                entry += f"\n  Conflicts with: {'; '.join(conflicts)}"
            entry += f"\n  Updated: {m.updated_at.isoformat()}"
            entry += f"\n  Created: {m.created_at.isoformat()}"
            lines.append(entry)

    return "\n".join(lines)


