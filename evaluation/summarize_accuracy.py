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


# ---------------------------------------------------------------------------
# Difficulty metadata
# ---------------------------------------------------------------------------

OXFORD_DIFFICULTY = {
    # easy
    "bodleian_library_video3": "easy",
    "hb-allen-centre_video1": "easy",
    "observatory_quarter_video4": "easy",
    "hb-allen-centre_video2": "easy",
    "observatory_quarter_video3": "easy",
    "bodleian_library_video5": "easy",
    "bodleian_library_video7": "easy",
    # medium
    "keble_college_video4": "medium",
    "observatory_quarter_video1": "medium",
    "hb-allen-centre_video3": "medium",
    "keble_college_video1": "medium",
    "observatory_quarter_video2": "medium",
    "keble_college_video3": "medium",
    "bodleian_library_video8": "medium",
    "hb-allen-centre_video4": "medium",
    # hard
    "bodleian_library_video1": "hard",
    "bodleian_library_video2": "hard",
    "bodleian_library_video4": "hard",
    "bodleian_library_video6": "hard",
    "keble_college_video2": "hard",
}

SYNTHETIC_DIFFICULTY = {
    # easy
    "prototype": "easy", "1b6a581d": "easy", "3c4170c4": "easy",
    "4e55eb25": "easy", "4b47e41e": "easy", "5bd964a8": "easy",
    "4f88ad8f": "easy", "1b6f8fec": "easy", "1a7ad286": "easy",
    "1a7ad286t2": "easy", "33b1d89c": "easy", "17de636e": "easy",
    "18dabf13": "easy", "36c2b3d2": "easy", "7b1c837d": "easy",
    "6b9cab17": "easy", "6ef3f393": "easy", "5c7459a0": "easy",
    "18477db1t2": "easy", "4808fcc3": "easy", "6c42cee9": "easy",
    "52f85136": "easy", "5851f2a7": "easy", "18477db1": "easy",
    "10283943": "easy", "11648361": "easy", "264955de": "easy",
    "ce70da": "easy", "f3420e4": "easy", "5284d2fe": "easy",
    "68102576": "easy", "a062f52": "easy", "bce91ef": "easy",
    "c8bfc09": "easy", "4f7aa10f": "easy",
    # medium
    "53fe218a": "medium", "61d7519": "medium", "1754e196": "medium",
    "7abb8e01": "medium", "7b08481": "medium", "22f2a96b": "medium",
    "79d99a84": "medium", "304ba0d4": "medium", "518e795a": "medium",
    "1667a095": "medium",
    # hard
    "1704ce1": "hard", "2208e28e": "hard", "2399bb32": "hard",
    "2522e769": "hard", "3896eeeb": "hard",
}

SOFIA_DIFFICULTY = {
    "Apartment1": "medium", "Apartment2": "hard",
    "BurgerShop": "easy", "BusStop": "easy", "Cathedral": "easy",
    "GroceryShop": "hard", "INSAIT_left": "hard",
    "INSAIT_right1": "medium", "INSAIT_right2": "medium",
    "INSAIT_room11": "easy", "INSAIT_room14": "medium",
    "Palace": "easy", "Reception": "easy", "Roadway": "easy",
    "scene1": "easy", "scene2": "easy", "scene3": "easy",
    "Statue": "medium",  # normalized from "Medium"
    "TechPark_Table_Tennis": "medium", "TechPark_Trash_Bin": "medium",
}

# Map dataset name (lowercase) -> difficulty dict
DIFFICULTY_MAPS = {
    "oxford": OXFORD_DIFFICULTY,
    "synthetic": SYNTHETIC_DIFFICULTY,
    "sofia": SOFIA_DIFFICULTY,
}
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


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
        qa_type = qa_obj.get("M", "Unknown").strip().replace("_", " ").title()
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
    """
    Process one dataset (Sofia/Oxford/Synthetic).

    Returns:
        {
            "overall": {qa_type: {total, correct, accuracy}, ...},
            "by_difficulty": {
                "easy":   {qa_type: {...}, ...},
                "medium": {qa_type: {...}, ...},
                "hard":   {qa_type: {...}, ...},
            }
        }
    """
    difficulty_map = DIFFICULTY_MAPS.get(dataset_name.lower(), {})
    subfolders = [f.path for f in os.scandir(dataset_path) if f.is_dir()]

    all_filtered = []
    by_difficulty: dict = {level: [] for level in DIFFICULTY_LEVELS}

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

        # Determine difficulty from subfolder name
        scene_name = os.path.basename(sub)
        difficulty = difficulty_map.get(scene_name)
        if difficulty is None:
            # Try case-insensitive match
            scene_lower = scene_name.lower()
            for k, v in difficulty_map.items():
                if k.lower() == scene_lower:
                    difficulty = v
                    break
        if difficulty is not None:
            difficulty = difficulty.lower()
            if difficulty in by_difficulty:
                by_difficulty[difficulty].extend(filtered)

    return {
        "overall": calculate_accuracy_by_qa(all_filtered),
        "by_difficulty": {
            level: calculate_accuracy_by_qa(by_difficulty[level])
            for level in DIFFICULTY_LEVELS
        },
    }


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


def merge_difficulty(a: dict, b: dict) -> dict:
    """Merge two by_difficulty dicts {level: accuracy_dict}."""
    result = {}
    for level in DIFFICULTY_LEVELS:
        result[level] = merge_accuracy(a.get(level, {}), b.get(level, {}))
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
    per_difficulty = {}
    overall_acc = {}
    overall_diff = {level: {} for level in DIFFICULTY_LEVELS}

    if args.sofia_path and os.path.isdir(args.sofia_path):
        result = process_dataset(args.sofia_path, "sofia", args.model, args.split)
        per_dataset["Sofia"] = result["overall"]
        per_difficulty["Sofia"] = result["by_difficulty"]
        overall_acc = merge_accuracy(overall_acc, result["overall"])
        overall_diff = merge_difficulty(overall_diff, result["by_difficulty"])

    if args.oxford_path and os.path.isdir(args.oxford_path):
        result = process_dataset(
            args.oxford_path,
            "oxford",
            args.model,
            args.split,
            score_dir=args.oxford_score_dir,
        )
        per_dataset["Oxford"] = result["overall"]
        per_difficulty["Oxford"] = result["by_difficulty"]
        overall_acc = merge_accuracy(overall_acc, result["overall"])
        overall_diff = merge_difficulty(overall_diff, result["by_difficulty"])

    if args.synthetic_path and os.path.isdir(args.synthetic_path):
        result = process_dataset(args.synthetic_path, "synthetic", args.model, args.split)
        per_dataset["Synthetic"] = result["overall"]
        per_difficulty["Synthetic"] = result["by_difficulty"]
        overall_acc = merge_accuracy(overall_acc, result["overall"])
        overall_diff = merge_difficulty(overall_diff, result["by_difficulty"])

    summary = {
        "model": args.model,
        "split": args.split,
        "per_dataset": per_dataset,
        "overall": overall_acc,
        "per_dataset_difficulty": per_difficulty,
        "overall_difficulty": overall_diff,
    }

    # Print per-dataset QA-type breakdown
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

    # Print difficulty breakdown
    if per_difficulty:
        print("\n--- DIFFICULTY BREAKDOWN (per dataset) ---")
        for name, diff_acc in per_difficulty.items():
            print(f"  {name}:")
            for level in DIFFICULTY_LEVELS:
                v = diff_acc.get(level, {}).get("OVERALL", {"accuracy": 0.0, "correct": 0, "total": 0})
                print(f"    {level}: {v['accuracy']:.2%} ({v['correct']}/{v['total']})")
        print("\n--- DIFFICULTY BREAKDOWN (overall) ---")
        for level in DIFFICULTY_LEVELS:
            v = overall_diff.get(level, {}).get("OVERALL", {"accuracy": 0.0, "correct": 0, "total": 0})
            print(f"  {level}: {v['accuracy']:.2%} ({v['correct']}/{v['total']})")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")



if __name__ == "__main__":
    main()
