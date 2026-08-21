import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from client.client import Client
from judge import Judge
from llm import LLM, Ollama, OpenAI
from paths import DEFAULT_CONFIG_PATH

DATA_DIR = REPO_ROOT / "benchmarks" / "CrossMemBench" / "data"
NOISE_PATH = REPO_ROOT / "benchmarks" / "CrossMemBench" / "data" / "noise.json"
AGENT_PROMPT_PATH = REPO_ROOT / "benchmarks" / "CrossMemBench" / "agent_prompt.md"
JUDGE_PROMPT_PATH = REPO_ROOT / "benchmarks" / "CrossMemBench" / "judge_prompt.md"
RESULTS_DIR = REPO_ROOT / "results" / "crossmembench"

USERS = [f"u{i:03d}" for i in range(1, 41)]
NOISE_SEED = 42
NOISE_COUNT = 100
MAX_RETRIES = 3
AGENT_TEMPERATURE = 0.0
AGENT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_AGENTS = [
    "openai:gpt-5.4-nano",
    "openai:gpt-4o",
    "ollama:llama3.2:3b",
]
WRONG_ANSWER_JUDGE = {
    "score": 0.0,
    "explanation": "Agent selected wrong answer -- auto generated judge response",
}


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


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_user(user_id: str) -> dict:
    return load_json(DATA_DIR / user_id / f"{user_id}.json")


def load_cmrt_questions(user: dict) -> list[dict]:
    return [q for q in user["questions"] if q["task_type"] == "CMRT"]


def memories_by_id(user: dict) -> dict[str, dict]:
    return {m["memory_id"]: m for m in user["memories"]}


def load_noise_slice() -> list[dict]:
    memories = list(load_json(NOISE_PATH)["memories"])
    rng = random.Random(NOISE_SEED)
    rng.shuffle(memories)
    if NOISE_COUNT > len(memories):
        raise ValueError(
            f"need {NOISE_COUNT} noise memories, file has {len(memories)}"
        )
    return memories[:NOISE_COUNT]


def format_options(options: dict) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in options.items())


def format_judge_memory(memory: dict) -> str:
    return f"{memory['memory_id']}: {memory['content']} ({memory['inferred_memory']})"


def strip_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        text = text[1:-1].strip()
    return text


def parse_agent_response(response: str) -> dict:
    result = json.loads(strip_json_text(response))
    if "selection" not in result:
        raise ValueError(f"Agent response missing selection: {result}")
    if "reasoning" not in result:
        raise ValueError(f"Agent response missing reasoning: {result}")
    return {
        "selection": str(result["selection"]).strip(),
        "reasoning": result["reasoning"],
    }


def assemble_agent_messages(
    question: dict,
    retrieved_context: str,
    cmrt: str,
) -> list[dict]:
    user = (
        cmrt.replace("{memories_section}", retrieved_context)
        .replace("{question}", question["question"])
        .replace("{options}", format_options(question["options"]))
    )
    return [{"role": "user", "content": user}]


def assemble_judge_prompt(
    question: dict,
    agent_response: dict,
    eval_memory: dict,
    judge_template: str,
) -> str:
    return (
        judge_template.replace("{memory}", format_judge_memory(eval_memory))
        .replace("{question}", question["question"])
        .replace("{options}", format_options(question["options"]))
        .replace(
            "{correct_answer}",
            f"{question['correct_answer']}. {question['options'][question['correct_answer']]}",
        )
        .replace("{agent_selection}", agent_response["selection"])
        .replace("{agent_reasoning}", agent_response["reasoning"])
    )


def insert_memory(client: Client, memory: dict) -> None:
    content = memory["content"]
    mem_id = memory["memory_id"]
    print(f"Inserting memory {mem_id}")
    print(f"Submitting memory: {content} to user: {client.user_id} for id {mem_id}")
    client.add(
        {
            "user": content,
            "timestamp": memory["timestamp"],
            "custom_metadata": {
                "memory_id": memory["memory_id"]
            },
        }
    )


def insert_memories(client: Client, memories: list[dict]) -> None:
    for memory in memories:
        insert_memory(client, memory)
    print(f"Inserted {len(memories)} memxd memories for user: {client.user_id}")


def retrieve_memories(client: Client, query: str) -> tuple[str, str | None]:
    if not query.strip():
        return "", None
    context, raw = client.retrieve(query)
    print(f"MEMXD RAW SEARCH RESULTS (user={client.user_id}): {raw}")
    return context, str(raw)


def debug_list_all(client: Client) -> None:
    dump = client.list_all()
    explicit = dump.get("explicit", [])
    latent = dump.get("latent", [])
    print(
        f"MEMXD LIST_ALL (user={client.user_id}): "
        f"explicit={len(explicit)} latent={len(latent)}"
    )
    print(f"MEMXD LIST_ALL CONTENTS (user={client.user_id}): {dump}")


def user_results_path(run_dir: Path, agent_model: str, user_id: str) -> Path:
    model_dir = run_dir / agent_model
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{user_id}.json"


def load_user_results(
    run_dir: Path, agent_model: str, user_id: str, noise_level: int
) -> dict:
    path = user_results_path(run_dir, agent_model, user_id)
    if not path.exists():
        return {
            "user_id": user_id,
            "memory_system": "memxd",
            "agent_model": agent_model,
            "noise_level": noise_level,
            "questions": [],
        }
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_results(
    run_dir: Path,
    user_id: str,
    question: dict,
    agent_response: dict,
    judge_response: dict | str | None,
    *,
    noise_level: int,
    agent_model: str,
    retrieved_context: str,
    raw_retrieval: str,
    final_score: float | None,
) -> None:
    user_results = load_user_results(run_dir, agent_model, user_id, noise_level)
    result = {
        "question_id": question["question_id"],
        "user_id": user_id,
        "task_type": question["task_type"],
        "target_domain": question["target_domain"],
        "correct_answer": question["correct_answer"],
        "eval_memory_ids": question["eval_memory_ids"],
        "retrieved_context": retrieved_context,
        "raw_retrieval": raw_retrieval,
        "agent_response": agent_response,
        "judge_response": judge_response,
    }
    if final_score is not None:
        result["final_score"] = final_score
    user_results["questions"].append(result)
    path = user_results_path(run_dir, agent_model, user_id)
    with path.open("w", encoding="utf-8") as f:
        json.dump(user_results, f, indent=2)


def write_run_failure(
    run_dir: Path,
    user_id: str,
    noise_level: int,
    *,
    agent_model: str,
    exception_type: str,
    error_message: str,
    failure_stage: str,
) -> None:
    payload = {
        "user_id": user_id,
        "memory_system": "memxd",
        "agent_model": agent_model,
        "noise_level": noise_level,
        "run_status": "failed",
        "failure_stage": failure_stage,
        "exception_type": exception_type,
        "error_message": error_message,
        "questions": [],
    }
    path = user_results_path(run_dir, agent_model, user_id)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def abort_run(
    run_dir: Path,
    user_id: str,
    noise_level: int,
    error: Exception,
    failure_stage: str,
    agent_models: list[str],
) -> None:
    print(
        f"Aborting run: {failure_stage} failed for {user_id} "
        f"noise_level={noise_level}: {type(error).__name__}: {error}"
    )
    for agent_model in agent_models:
        write_run_failure(
            run_dir,
            user_id,
            noise_level,
            agent_model=agent_model,
            exception_type=type(error).__name__,
            error_message=str(error),
            failure_stage=failure_stage,
        )


def run_agent(llm: LLM, messages: list[dict]) -> tuple[dict | None, str]:
    last_error = None
    last_raw = ""
    for _ in range(MAX_RETRIES):
        try:
            last_raw = llm.complete(
                messages,
                temperature=AGENT_TEMPERATURE,
                max_output_tokens=AGENT_MAX_OUTPUT_TOKENS,
            )
        except Exception as e:
            return None, str(e)
        try:
            return parse_agent_response(last_raw), last_raw
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            last_error = e
    reason = last_error or last_raw
    return None, str(reason)


def run_judge(judge: Judge, prompt: str) -> tuple[dict | None, str | None]:
    last_error = None
    for _ in range(MAX_RETRIES):
        try:
            return judge.run(prompt), None
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
        except Exception as e:
            return None, str(e)
    return None, str(last_error)


def evaluate_user(
    *,
    client: Client,
    user: dict,
    questions: list[dict],
    noise_level: int,
    backbones: list[tuple[str, LLM]],
    judge: Judge,
    cmrt: str,
    judge_template: str,
    run_dir: Path,
) -> bool:
    memory_index = memories_by_id(user)
    for question in questions:
        print(f"Asking question id {question['question_id']}")
        try:
            retrieved_context, raw_retrieval = retrieve_memories(
                client, question["question"]
            )
        except Exception as e:
            abort_run(
                run_dir,
                client.user_id,
                noise_level,
                e,
                "retrieval",
                [model_name for model_name, _ in backbones],
            )
            return False
        if raw_retrieval is None:
            raw_retrieval = ""
        print(
            f"Frozen retrieval/context for question "
            f"{question['question_id']} (shared across backbones)"
        )
        messages = assemble_agent_messages(question, retrieved_context, cmrt)
        eval_memory = memory_index[question["eval_memory_ids"][0]]
        for model_name, llm in backbones:
            print(
                f"========================================\n"
                f"Running agent backbone: {model_name}\n"
                f"========================================"
            )
            print(
                f"Asking question id {question['question_id']} "
                f"on backbone {model_name}"
            )
            parsed, raw_or_reason = run_agent(llm, messages)
            if parsed is None:
                agent_response = {"selection": "", "reasoning": raw_or_reason}
                judge_response = raw_or_reason
                final_score = 0.0
                print(f"Agent: FAILED — {raw_or_reason}")
            elif parsed["selection"] != question["correct_answer"]:
                agent_response = parsed
                judge_response = WRONG_ANSWER_JUDGE
                final_score = 0.0
                print(
                    f"Agent: selected {parsed['selection']} | "
                    f"correct={question['correct_answer']} | WRONG"
                )
                print(
                    f"Judge: score={judge_response['score']} | "
                    f"{judge_response['explanation']}"
                )
            else:
                agent_response = parsed
                judge_prompt = assemble_judge_prompt(
                    question, parsed, eval_memory, judge_template
                )
                judge_response, judge_fail = run_judge(judge, judge_prompt)
                if judge_response is None:
                    judge_response = judge_fail
                    final_score = 0.0
                    print(
                        f"Agent: selected {parsed['selection']} | "
                        f"correct={question['correct_answer']} | CORRECT"
                    )
                    print(f"Judge: FAILED — {judge_fail}")
                else:
                    final_score = float(judge_response["score"])
                    print(
                        f"Agent: selected {parsed['selection']} | "
                        f"correct={question['correct_answer']} | CORRECT"
                    )
                    print(
                        f"Judge: score={judge_response['score']} | "
                        f"{judge_response['explanation']}"
                    )
            write_results(
                run_dir,
                client.user_id,
                question,
                agent_response,
                judge_response,
                noise_level=noise_level,
                agent_model=model_name,
                retrieved_context=retrieved_context,
                raw_retrieval=raw_retrieval,
                final_score=final_score,
            )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="append", default=None)
    parser.add_argument("--user", action="append", default=None)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    specs = args.agent if args.agent else DEFAULT_AGENTS
    backbones = []
    for spec in specs:
        provider, model = parse_agent_spec(spec)
        backbones.append((model, make_llm(provider, model)))
    agent_models = [model for model, _ in backbones]
    users = args.user if args.user else USERS

    cmrt = AGENT_PROMPT_PATH.read_text(encoding="utf-8")
    judge_template = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    noise_memories = load_noise_slice()
    judge = Judge()
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_DIR / f"memxd_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    write_query_top_k(explicit=5, latent=5)
    print("Judge and agents successfully loaded.")
    print(f"Agent models: {agent_models}")
    print("top_k_explicit=5 top_k_latent=5")
    print(f"Noise shuffle seed: {NOISE_SEED}")
    print(f"Run directory: {run_dir}")

    for user_id in users:
        print(f"Running experiment for user: {user_id}")
        print("--------------------------------")
        user = load_user(user_id)
        questions = load_cmrt_questions(user)
        client = Client(user_id=user_id)
        client.reset()
        try:
            insert_memories(client, user["memories"] + noise_memories)
        except Exception as e:
            abort_run(run_dir, user_id, 100, e, "insert", agent_models)
            return
        print(f"User {user_id}, noise_level=100")
        if not evaluate_user(
            client=client,
            user=user,
            questions=questions,
            noise_level=100,
            backbones=backbones,
            judge=judge,
            cmrt=cmrt,
            judge_template=judge_template,
            run_dir=run_dir,
        ):
            return
        print(
            f"DUMPING MEMORY STORE store after user={user_id} noise_level=100"
        )
        debug_list_all(client)
        client.reset()


if __name__ == "__main__":
    main()
