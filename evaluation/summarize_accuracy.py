"""
Summarize accuracy per-dataset (Sofia, Oxford, Synthetic) and overall.

Usage:
  python evaluation/summarize_accuracy.py --sofia_path /path/to/Sofia_server \
      --oxford_path /path/to/Oxford_server \
      --synthetic_path /path/to/synthetic_server \
      [--model gpt] [--split night]

Reads scores from qa_result/*_scores_*.json. Sofia/Oxford: filters by all_qa_filtered.json.
Synthetic: uses all scored questions (no QA-file filter).
"""
import argparse
import json
import os
from collections import defaultdict


def get_qa_questions(qa_path: str) -> set:
    """Load all_qa_filtered.json and return set of question strings for filtering."""
    if not os.path.exists(qa_path):
        return set()
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item.get("question", "") for item in data}


def load_and_filter_scores(score_path: str, q_questions: set | None) -> list:
    """Load score JSON. If q_questions is None, use all; else filter to Q in q_questions."""
    if not os.path.exists(score_path):
        return []
    with open(score_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Score format: list of [pred_obj, qa_obj] where qa_obj has "Q", "M", etc.
    pairs = [pair for pair in data if len(pair) > 1]
    if q_questions is None:
        return pairs
    return [p for p in pairs if p[1].get("Q") in q_questions]


def calculate_accuracy_by_qa(filtered_scores: list) -> dict:
    """Compute accuracy per QA type and overall. Returns {qa_type: {total, correct, accuracy}}."""
    stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for pair in filtered_scores:
        pred_obj = pair[0]
        qa_obj = pair[1] if len(pair) > 1 else {}
        qa_type = qa_obj.get("M", "Unknown")
        pred = pred_obj.get("pred", "").lower()
        if "pred" in pred_obj:
            stats[qa_type]["total"] += 1
            if pred == "correct":
                stats[qa_type]["correct"] += 1
    results = {}
    for qa_type, v in stats.items():
        total = v["total"]
        correct = v["correct"]
        results[qa_type] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
        }
    total_all = sum(v["total"] for v in stats.values())
    correct_all = sum(v["correct"] for v in stats.values())
    results["OVERALL"] = {
        "total": total_all,
        "correct": correct_all,
        "accuracy": correct_all / total_all if total_all > 0 else 0.0,
    }
    return results


def process_dataset(
    dataset_path: str,
    dataset_name: str,
    model: str,
    split: str,
    score_dir: str = "qa_result",
) -> dict:
    """Process one dataset (Sofia/Oxford/Synthetic) and return per-dataset accuracy."""
    subfolders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
    all_filtered = []
    for sub in subfolders:
        score_dir_path = os.path.join(sub, score_dir)
        if not os.path.isdir(score_dir_path):
            continue
        score_path = os.path.join(score_dir_path, f"{model}_scores_{split}.json")
        if not os.path.exists(score_path) and score_dir == "final_score":
            score_path = os.path.join(
                score_dir_path, f"filtered_{model}_final_scores_{split}.json"
            )
        if not os.path.exists(score_path):
            continue
        qa_path = os.path.join(sub, "qa_result", "all_qa_filtered.json")
        q_questions = get_qa_questions(qa_path)
        if not q_questions:
            continue
        filtered = load_and_filter_scores(score_path, q_questions)
        all_filtered.extend(filtered)
    return calculate_accuracy_by_qa(all_filtered)


def merge_accuracy(a: dict, b: dict) -> dict:
    """Merge two accuracy dicts by summing total/correct."""
    out = defaultdict(lambda: {"total": 0, "correct": 0})
    for d in (a, b):
        for qa_type, v in d.items():
            out[qa_type]["total"] += v.get("total", 0)
            out[qa_type]["correct"] += v.get("correct", 0)
    result = {}
    for qa_type, v in out.items():
        t, c = v["total"], v["correct"]
        result[qa_type] = {"total": t, "correct": c, "accuracy": c / t if t > 0 else 0.0}
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Summarize per-dataset and overall accuracy for Sofia, Oxford, Synthetic"
    )
    parser.add_argument("--sofia_path", type=str, default="", help="Path to Sofia_server")
    parser.add_argument("--oxford_path", type=str, default="", help="Path to Oxford_server")
    parser.add_argument("--synthetic_path", type=str, default="", help="Path to synthetic_server")
    parser.add_argument("--model", type=str, default="gpt", help="Model name (gpt, gemini, qwen7b, etc.)")
    parser.add_argument("--split", type=str, default="night", choices=["night", "day"])
    parser.add_argument("--output", type=str, default="", help="Output JSON path")
    parser.add_argument(
        "--oxford_score_dir",
        type=str,
        default="qa_result",
        help="Score dir for Oxford (qa_result or final_score)",
    )
    args = parser.parse_args()

    per_dataset = {}
    overall_acc = {}

    if args.sofia_path and os.path.isdir(args.sofia_path):
        per_dataset["Sofia"] = process_dataset(
            args.sofia_path, "sofia", args.model, args.split
        )
        overall_acc = merge_accuracy(overall_acc, per_dataset["Sofia"])

    if args.oxford_path and os.path.isdir(args.oxford_path):
        per_dataset["Oxford"] = process_dataset(
            args.oxford_path,
            "oxford",
            args.model,
            args.split,
            score_dir=args.oxford_score_dir,
        )
        overall_acc = merge_accuracy(overall_acc, per_dataset["Oxford"])

    if args.synthetic_path and os.path.isdir(args.synthetic_path):
        per_dataset["Synthetic"] = process_dataset(
            args.synthetic_path, "synthetic", args.model, args.split
        )
        overall_acc = merge_accuracy(overall_acc, per_dataset["Synthetic"])

    summary = {
        "model": args.model,
        "split": args.split,
        "per_dataset": per_dataset,
        "overall": overall_acc,
    }

    # Print
    print(f"\n=== Model: {args.model} | Split: {args.split} ===\n")
    for name, acc in per_dataset.items():
        print(f"--- {name} ---")
        for qa_type in sorted(acc.keys(), key=lambda x: (x != "OVERALL", x)):
            v = acc[qa_type]
            print(f"  {qa_type}: {v['accuracy']:.2%} ({v['correct']}/{v['total']})")
        print()
    print("--- OVERALL (all datasets combined) ---")
    for qa_type in sorted(overall_acc.keys(), key=lambda x: (x != "OVERALL", x)):
        v = overall_acc[qa_type]
        print(f"  {qa_type}: {v['accuracy']:.2%} ({v['correct']}/{v['total']})")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
