import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
if not run_dir.exists():
    print("run name doesn't exist")
    raise SystemExit(1)

analysis_path = run_dir / "analysis.json"


def print_analysis(analysis):
    for row in analysis["task"]:
        n = row["n"]
        total = row["correct"]
        print(f"{row['model']}: task={total / n:.2f} ({int(total)}/{n})")
    for row in analysis["retrieval"]:
        n = row["n"]
        total = row["correct"]
        print(f"retrieval={total / n:.2f} ({int(total)}/{n})")
    if analysis["failures"]:
        print("FAILURES:")
        for line in analysis["failures"]:
            print(line)


if analysis_path.exists():
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    print_analysis(analysis)
    if analysis["failures"]:
        raise SystemExit(1)
    raise SystemExit(0)

files = sorted(run_dir.glob("*/u*.json"))
first_model = sorted(
    p.name for p in run_dir.iterdir() if p.is_dir() and p.name != "logs"
)[0]

failures = []
task = {}
retrieval = []

for path in files:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("run_status") == "failed":
        failures.append(
            f"{path}: run_status=failed "
            f"user={data.get('user_id')} "
            f"stage={data.get('failure_stage')} {data.get('error_message')}"
        )
        continue
    model = data["agent_model"]
    for question in data["questions"]:
        qid = question.get("question_id")
        if "final_score" not in question or "raw_retrieval" not in question:
            failures.append(
                f"{path}: incomplete question {qid} "
                f"user={data.get('user_id')}"
            )
            continue
        task.setdefault(model, []).append(float(question["final_score"]))
        if model == first_model:
            haystack = re.sub(r"\s+", "", str(question["raw_retrieval"]))
            hit = all(mid in haystack for mid in question["eval_memory_ids"])
            retrieval.append(1.0 if hit else 0.0)

analysis = {"task": [], "retrieval": [], "failures": failures}
for model in sorted(task):
    scores = task[model]
    n = len(scores)
    total = sum(scores)
    analysis["task"].append(
        {
            "model": model,
            "accuracy": total / n,
            "correct": int(total),
            "n": n,
        }
    )
if retrieval:
    n = len(retrieval)
    total = sum(retrieval)
    analysis["retrieval"].append(
        {
            "accuracy": total / n,
            "correct": int(total),
            "n": n,
        }
    )

analysis_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
print_analysis(analysis)
if failures:
    raise SystemExit(1)
