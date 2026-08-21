"""Agent-facing query retrieval."""

from __future__ import annotations

import yaml

from client.prompt_controller import (
    build_probe_generation_prompt,
    format_memories_for_agent,
)
from db.controller import ChromaController
from db.utils import cosine_similarity
from llms.factory import llm_from_config
from llms.json_schema import PROBE_GENERATION_SCHEMA
from memory.memory import ExplicitMemory, LatentMemory
from paths import DEFAULT_CONFIG_PATH


def _load_query_config() -> dict:
    with open(DEFAULT_CONFIG_PATH) as f:
        return yaml.safe_load(f)["query"]


def _score_question_against_pool(
    question: str,
    pool: list[dict],
    *,
    threshold: float,
    vector_db: ChromaController,
) -> tuple[list[tuple[dict, float]], tuple[dict, float] | None]:
    """Returns (survivors, best_non_survivor_or_None)."""
    if not pool:
        return [], None

    question_embedding = vector_db.embed_for_retrieval(question)

    survivors: list[tuple[dict, float]] = []
    best_dropped: tuple[dict, float] | None = None
    for item in pool:
        cosine = cosine_similarity(question_embedding, item["embedding"])
        if cosine >= threshold:
            survivors.append((item, cosine))
        elif best_dropped is None or cosine > best_dropped[1]:
            best_dropped = (item, cosine)
    return survivors, best_dropped


def _update_bank(
    bank: dict[str, dict],
    survivors: list[tuple[dict, float]],
    *,
    question: str | None = None,
) -> None:
    for item, score in survivors:
        mid = item["memory_id"]
        if mid not in bank:
            entry: dict = {
                "max_score": score,
                "hit_count": 1,
                "item": item,
            }
            if question is not None:
                entry["best_question"] = question
            bank[mid] = entry
        else:
            entry = bank[mid]
            entry["hit_count"] += 1
            if score > entry["max_score"]:
                entry["max_score"] = score
                if question is not None:
                    entry["best_question"] = question


def _finalize_bank(
    bank: dict[str, dict],
    *,
    threshold: float,
    top_k: int,
    label: str,
) -> list[dict]:
    ranked = sorted(
        bank.values(),
        key=lambda entry: entry["max_score"],
        reverse=True,
    )
    kept = [
        entry for entry in ranked if entry["max_score"] >= threshold
    ][:top_k]

    print(f"\n=== QUERY FINAL RANK [{label}] ===", flush=True)
    for i, entry in enumerate(kept, start=1):
        print(
            f"  {i}. [score={entry['max_score']:.4f} hits={entry['hit_count']}] "
            f"{entry['item']['text']!r}",
            flush=True,
        )
    return [entry["item"] for entry in kept]


def _print_raw_bank(bank: dict[str, dict], label: str) -> None:
    rows = sorted(
        (
            {
                "memory_id": mid,
                "max_score": entry["max_score"],
                "hit_count": entry["hit_count"],
                "text": entry["item"]["text"],
                "latent_preference": getattr(
                    entry["item"]["memory"], "latent_preference", None
                ),
                "kind": entry["item"]["kind"],
            }
            for mid, entry in bank.items()
        ),
        key=lambda row: row["max_score"],
        reverse=True,
    )
    print(
        f"\n=== RAW BANK [{label}] ===\n{rows}",
        flush=True,
    )


def _dedupe_latent_bank_by_preference(bank: dict[str, dict]) -> dict[str, dict]:
    """Collapse question-rows by latent_preference text; keep highest max_score."""
    winners: dict[str, tuple[str, dict]] = {}
    for mid, entry in bank.items():
        trait = entry["item"]["memory"].latent_preference
        if trait not in winners or entry["max_score"] > winners[trait][1]["max_score"]:
            winners[trait] = (mid, entry)
    return {mid: entry for mid, entry in winners.values()}


def _retrieve_query_memories(
    user_query: str,
    vector_db: ChromaController,
) -> tuple[list[ExplicitMemory], list[LatentMemory]]:
    """Shared query pipeline → (explicit, latent) after per-bank top_k."""
    query_config = _load_query_config()
    llm = llm_from_config()

    instructions, prompt = build_probe_generation_prompt(user_query)
    extracted = llm.generate_json(
        prompt=prompt,
        instructions=instructions,
        json_schema=PROBE_GENERATION_SCHEMA,
    )
    print(f"\nALL QUESTIONS FROM PROBE GENERATION: {extracted}", flush=True)

    trait_questions: list[str] = list(extracted["trait"])
    direct_questions: list[str] = list(extracted["direct"])

    threshold_explicit = query_config["retrieval_score_threshold_explicit"]
    threshold_latent = query_config["retrieval_score_threshold_latent"]
    top_k_explicit = query_config["top_k_explicit"]
    top_k_latent = query_config["top_k_latent"]

    latent_pool = vector_db.list_memories_for_query("latent")
    explicit_pool = vector_db.list_memories_for_query("explicit")

    latent_bank: dict[str, dict] = {}
    explicit_bank: dict[str, dict] = {}

    for question in trait_questions:
        survivors, best_dropped = _score_question_against_pool(
            question,
            latent_pool,
            threshold=threshold_latent,
            vector_db=vector_db,
        )
        print(
            f"\n=== TRAIT QUESTION SCORE ===\nquestion={question!r}\n"
            f"survivors={len(survivors)}",
            flush=True,
        )
        if best_dropped is not None:
            dropped_item, dropped_score = best_dropped
            print(
                f"highest non-survivor: [{dropped_score:.4f}] "
                f"{dropped_item['text']!r}",
                flush=True,
            )
        else:
            print("highest non-survivor: (none)", flush=True)
        _update_bank(latent_bank, survivors)

    for question in direct_questions:
        survivors, best_dropped = _score_question_against_pool(
            question,
            explicit_pool,
            threshold=threshold_explicit,
            vector_db=vector_db,
        )
        print(
            f"\n=== DIRECT QUESTION SCORE ===\nquestion={question!r}\n"
            f"survivors={len(survivors)}",
            flush=True,
        )
        if best_dropped is not None:
            dropped_item, dropped_score = best_dropped
            print(
                f"highest non-survivor: [{dropped_score:.4f}] "
                f"{dropped_item['text']!r}",
                flush=True,
            )
        else:
            print("highest non-survivor: (none)", flush=True)
        _update_bank(explicit_bank, survivors)

    _print_raw_bank(latent_bank, "latent")
    _print_raw_bank(explicit_bank, "explicit")

    # Dedupe by trait.
    latent_bank = _dedupe_latent_bank_by_preference(latent_bank)
    _print_raw_bank(latent_bank, "latent-deduped")

    latent_items = _finalize_bank(
        latent_bank,
        threshold=threshold_latent,
        top_k=top_k_latent,
        label="latent",
    )
    explicit_items = _finalize_bank(
        explicit_bank,
        threshold=threshold_explicit,
        top_k=top_k_explicit,
        label="explicit",
    )

    explicit = [item["memory"] for item in explicit_items]
    latent = [item["memory"] for item in latent_items]
    return explicit, latent


def retrieve(user_query: str, vector_db: ChromaController) -> tuple[str, dict]:
    """Run query pipeline once → (formatted context, raw explicit/latent hits)."""
    explicit, latent = _retrieve_query_memories(user_query, vector_db)
    context = format_memories_for_agent(explicit, latent, vector_db)
    return context, {"explicit": explicit, "latent": latent}


def query(user_query: str, vector_db: ChromaController) -> tuple[str, dict]:
    """Alias for retrieve()."""
    return retrieve(user_query, vector_db)
    