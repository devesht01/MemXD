import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from client.client import Client
from llm import LLM, Ollama, OpenAI
from paths import DEFAULT_CONFIG_PATH

QUESTION_PATH = (
    REPO_ROOT / "benchmarks" / "PersonaMem" / "data" / "questions_32k.csv"
)
CONTEXT_PATH = (
    REPO_ROOT
    / "benchmarks"
    / "PersonaMem"
    / "data"
    / "shared_contexts_32k.jsonl"
)
RESULTS_DIR = REPO_ROOT / "results" / "personamem"
DEFAULT_AGENTS = [
    "openai:gpt-5.4-nano",
    "openai:gpt-4o",
    "ollama:llama3.2:3b",
]
AGENT_TEMPERATURE = 0.0
QUESTION_TYPES = {
    "generalize_to_new_scenarios",
    "generalizing_to_new_scenarios",
}
ASK_INSTRUCTIONS = (
    "Find the most appropriate model response and give "
    "your final answer (a), (b), (c), or (d) after the "
    "special token <final_answer>."
)


def write_query_top_k(explicit: int, latent: int) -> None:
    lines = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    found_explicit = 0
    found_latent = 0
    out = []
    for line in lines:
        if line.startswith("  top_k_explicit:"):
            out.append(f"  top_k_explicit: {explicit}\n")
            found_explicit += 1
        elif line.startswith("  top_k_latent:"):
            out.append(f"  top_k_latent: {latent}\n")
            found_latent += 1
        else:
            out.append(line)
    if found_explicit != 1 or found_latent != 1:
        raise ValueError(
            f"failed to set top_k in {DEFAULT_CONFIG_PATH}: "
            f"explicit={found_explicit} latent={found_latent}"
        )
    DEFAULT_CONFIG_PATH.write_text("".join(out), encoding="utf-8")


def parse_agent_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(f"invalid --agent {spec!r}; expected provider:model")
    provider, model = spec.split(":", 1)
    if provider not in ("openai", "ollama"):
        raise ValueError(f"unknown agent provider: {provider}")
    if not model:
        raise ValueError(f"invalid --agent {spec!r}; missing model")
    return provider, model


def make_llm(provider: str, model: str) -> LLM:
    if provider == "openai":
        return OpenAI(model)
    if provider == "ollama":
        return Ollama(model)
    raise ValueError(f"unknown agent provider: {provider}")


def build_jsonl_index(jsonl_path: Path) -> dict:
    index = {}
    with jsonl_path.open(encoding="utf-8") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            key = next(iter(json.loads(line).keys()))
            index[key] = offset
    return index


def load_context_by_id(jsonl_path: Path, offset: int):
    with jsonl_path.open(encoding="utf-8") as f:
        f.seek(offset)
        item = json.loads(f.readline())
        return next(iter(item.values()))


def load_rows(csv_path: Path):
    with csv_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            yield dict(row)


def load_rows_with_context(csv_path: Path, jsonl_path: Path):
    jsonl_index = build_jsonl_index(jsonl_path)
    with csv_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        prev_sid = None
        prev_context = None
        for row in reader:
            row_data = dict(row)
            sid = row_data["shared_context_id"]
            if sid != prev_sid:
                current_context = load_context_by_id(jsonl_path, jsonl_index[sid])
                prev_sid = sid
                prev_context = current_context
            else:
                current_context = prev_context
            yield row_data, current_context


def normalize_context_message(message: dict) -> dict:
    role = message["role"]
    content = message["content"]
    if role == "system":
        role = "user"
    elif role == "user" and content.startswith("User: "):
        content = content[len("User: ") :]
    elif role == "assistant" and content.startswith("Assistant: "):
        content = content[len("Assistant: ") :]
    return {"role": role, "content": content}


def extract_answer(predicted_answer, correct_answer):
    def _extract_only_options(text):
        text = text.lower()
        in_parens = re.findall(r"\(([a-d])\)", text)
        if in_parens:
            return set(in_parens)
        return set(re.findall(r"\b([a-d])\b", text))

    correct = correct_answer.lower().strip("() ")
    full_response = predicted_answer
    predicted_answer = predicted_answer.strip()
    if "<final_answer>" in predicted_answer:
        predicted_answer = predicted_answer.split("<final_answer>")[-1].strip()
    if predicted_answer.endswith("</final_answer>"):
        predicted_answer = predicted_answer[: -len("</final_answer>")].strip()
    pred_options = _extract_only_options(predicted_answer)
    if pred_options == {correct}:
        return True, predicted_answer
    response_options = _extract_only_options(full_response)
    if response_options == {correct}:
        return True, predicted_answer
    return False, predicted_answer


def write_results_json(json_path: Path, payload: dict) -> None:
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def debug_list_all(client: Client) -> None:
    dump = client.list_all()
    explicit = dump.get("explicit", [])
    latent = dump.get("latent", [])
    print(
        f"MEMXD LIST_ALL (user={client.user_id}): "
        f"explicit={len(explicit)} latent={len(latent)} "
        f"total={len(explicit) + len(latent)}"
    )
    print(f"MEMXD LIST_ALL CONTENTS (user={client.user_id}): {dump}")


def overall_accuracy(results: list[dict]) -> float:
    if not results:
        raise ValueError("No evaluation results found.")
    correct = 0
    for row in results:
        if row["score"] is True:
            correct += 1
        elif row["score"] is not False:
            raise ValueError(f"Invalid score value: {row['score']!r}")
    accuracy = correct / len(results)
    print(f"Overall accuracy: {accuracy:.2%} ({correct}/{len(results)} correct)")
    return accuracy


def add_turn(client: Client, raw_message: dict) -> None:
    if raw_message.get("role") != "user":
        return
    normalized = normalize_context_message(raw_message)
    if not (normalized.get("content") or "").strip():
        return
    client.add(
        {
            "messages": [normalized],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


def result_row(question: dict, shared_context_id: str, memory_context: str, model_response: str) -> dict:
    score, predicted_answer = extract_answer(
        model_response, question["correct_answer"]
    )
    return {
        "score": score,
        "persona_id": question["persona_id"],
        "question_id": question["question_id"],
        "shared_context_id": shared_context_id,
        "user_question_or_message": question["user_question_or_message"],
        "question_type": question["question_type"],
        "topic": question["topic"],
        "context_length_in_tokens": question["context_length_in_tokens"],
        "context_length_in_letters": question["context_length_in_letters"],
        "distance_to_ref_in_blocks": question["distance_to_ref_in_blocks"],
        "distance_to_ref_in_tokens": question["distance_to_ref_in_tokens"],
        "num_irrelevant_tokens": question["num_irrelevant_tokens"],
        "distance_to_ref_proportion_in_context": question[
            "distance_to_ref_proportion_in_context"
        ],
        "end_index_in_shared_context": question["end_index_in_shared_context"],
        "all_options": question["all_options"],
        "memory_context": memory_context,
        "model_response": model_response,
        "len_of_model_response": len(model_response),
        "predicted_answer": predicted_answer,
        "correct_answer": question["correct_answer"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="append", default=None)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    specs = args.agent if args.agent else DEFAULT_AGENTS
    backbones = []
    for spec in specs:
        provider, model = parse_agent_spec(spec)
        backbones.append((provider, model, make_llm(provider, model)))

    question_path = str(QUESTION_PATH)
    context_path = str(CONTEXT_PATH)
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_id = run_dir.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"memxd_{timestamp}"
        run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    run_states = {}
    for provider, model, llm in backbones:
        results = []
        errors = []
        json_path = run_dir / f"{model}.json"
        run_states[model] = {
            "llm": llm,
            "provider": provider,
            "results": results,
            "errors": errors,
            "json_path": json_path,
            "payload": {
                "run_id": run_id,
                "started_at": started_at,
                "question_path": question_path,
                "context_path": context_path,
                "max_contexts": None,
                "top_contexts": None,
                "context_id": None,
                "shard_index": None,
                "num_shards": None,
                "provider": "memxd",
                "results": results,
                "errors": errors,
            },
        }

    write_query_top_k(explicit=5, latent=15)
    print(f"Writing run outputs to {run_dir}")
    print("top_k_explicit=5 top_k_latent=15")
    all_rows = list(load_rows(QUESTION_PATH))
    question_counts = Counter(
        row["shared_context_id"]
        for row in all_rows
        if row["question_type"] in QUESTION_TYPES
    )
    eligible_context_ids = set(question_counts)
    if not eligible_context_ids:
        raise ValueError(
            f"No questions of type {sorted(QUESTION_TYPES)} found in "
            f"{question_path}."
        )
    sorted_context_ids = sorted(eligible_context_ids)
    eligible_context_ids = set(sorted_context_ids)
    contexts_to_process = len(eligible_context_ids)
    processed_context_ids = set()
    processed_context_count = 0
    context_progress = tqdm(
        total=contexts_to_process,
        desc="Shared contexts",
        position=0,
    )

    for row, context in load_rows_with_context(QUESTION_PATH, CONTEXT_PATH):
        shared_context_id = row["shared_context_id"]
        if shared_context_id in processed_context_ids:
            continue
        if shared_context_id not in eligible_context_ids:
            continue
        if processed_context_count >= contexts_to_process:
            break
        processed_context_ids.add(shared_context_id)
        processed_context_count += 1
        persona_id = row["persona_id"]
        questions = sorted(
            (
                r
                for r in all_rows
                if r["shared_context_id"] == shared_context_id
                and r["question_type"] in QUESTION_TYPES
            ),
            key=lambda r: int(r["end_index_in_shared_context"]),
        )
        memory_user_id = f"persona_{persona_id}_{shared_context_id}"
        client = Client(user_id=memory_user_id)
        client.reset()
        print(
            f"RESET done for user={memory_user_id} "
            f"before Context {processed_context_count}/{contexts_to_process} "
            f"(shared_context_id={shared_context_id})"
        )
        turn_progress = tqdm(
            total=len(context),
            desc=(
                f"Shared context {processed_context_count}/"
                f"{contexts_to_process} messages"
            ),
            position=1,
            leave=False,
        )
        question_index = 0
        last_covered_end_index = 0

        def ask_due_questions():
            nonlocal question_index
            while (
                question_index < len(questions)
                and int(questions[question_index]["end_index_in_shared_context"])
                <= last_covered_end_index
            ):
                question = questions[question_index]
                question_text = question["user_question_or_message"]
                try:
                    memory_context, _ = client.retrieve(question_text)
                    prompt = (
                        question_text
                        + "\n\n"
                        + ASK_INSTRUCTIONS
                        + "\n\n"
                        + question["all_options"]
                    )
                except Exception as e:
                    print(f"Error: {e}")
                    error_row = {
                        "persona_id": question["persona_id"],
                        "question_id": question["question_id"],
                        "shared_context_id": shared_context_id,
                        "error": str(e),
                    }
                    for state in run_states.values():
                        state["errors"].append(error_row)
                        write_results_json(state["json_path"], state["payload"])
                    question_index += 1
                    continue
                messages = [
                    {"role": "system", "content": memory_context},
                    {"role": "user", "content": prompt},
                ]
                for model_name, state in run_states.items():
                    try:
                        model_response = state["llm"].complete(
                            messages, temperature=AGENT_TEMPERATURE
                        )
                        row_out = result_row(
                            question,
                            shared_context_id,
                            memory_context,
                            model_response,
                        )
                        state["results"].append(row_out)
                        write_results_json(state["json_path"], state["payload"])
                    except Exception as e:
                        print(f"Error [{state['provider']}/{model_name}]: {e}")
                        state["errors"].append(
                            {
                                "persona_id": question["persona_id"],
                                "question_id": question["question_id"],
                                "shared_context_id": shared_context_id,
                                "error": str(e),
                            }
                        )
                        write_results_json(state["json_path"], state["payload"])
                question_index += 1

        for message_index, raw_message in enumerate(context):
            turn_progress.update(1)
            last_covered_end_index = message_index + 1
            add_turn(client, raw_message)
            ask_due_questions()

        if question_index != len(questions):
            raise ValueError(
                f"{len(questions) - question_index} questions were not reached "
                f"for shared context {shared_context_id}."
            )
        debug_list_all(client)
        print(
            f"Context {processed_context_count}/{contexts_to_process} "
            f"(shared_context_id={shared_context_id})"
        )
        turn_progress.close()
        context_progress.update(1)

    context_progress.close()
    finished_at = datetime.now(timezone.utc).isoformat()
    for model_name, state in run_states.items():
        if state["errors"]:
            for error in state["errors"]:
                print(
                    f"Error [{state['provider']}/{model_name}] for persona_id "
                    f"{error['persona_id']} and question_id "
                    f"{error['question_id']}: {error['error']}"
                )
        print(f"Accuracy [{state['provider']}/{model_name}]:")
        accuracy = overall_accuracy(state["results"])
        state["payload"]["finished_at"] = finished_at
        state["payload"]["overall_accuracy"] = accuracy
        write_results_json(state["json_path"], state["payload"])


if __name__ == "__main__":
    main()
