import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


VIDEO_PROMPT_TEMPLATE = (
    "Please carefully read the question and answer using the visual cues from the sampled "
    "video frames. The original FPS of the video is 30, and this frame set is sampled at "
    "{sample_fps} fps. All actions refer to the camera wearer.\n\n"
    "Question: {question}\n"
    "Answer briefly and directly."
)


@dataclass
class SampleRoot:
    dataset_name: str
    root: Path
    sample_fps: int


def find_scene_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def find_frame_files(frame_dir: Path) -> list[Path]:
    if not frame_dir.exists():
        return []
    out = []
    for p in frame_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            out.append(p)
    return sorted(out)


def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def pick_representative_image(frame_files: list[Path], start_frame: int, end_frame: int) -> str:
    if not frame_files:
        return ""
    idx = (start_frame + end_frame) // 2
    idx = max(0, min(idx, len(frame_files) - 1))
    return str(frame_files[idx].resolve())


def sample_frame_paths(frame_files: list[Path], start_frame: int, end_frame: int, max_frames: int = 16) -> list[str]:
    if not frame_files:
        return []
    start_frame = max(0, min(start_frame, len(frame_files) - 1))
    end_frame = max(0, min(end_frame, len(frame_files) - 1))
    if end_frame < start_frame:
        end_frame = start_frame
    span = frame_files[start_frame : end_frame + 1]
    if len(span) <= max_frames:
        return [str(p.resolve()) for p in span]
    step = max(1, len(span) // max_frames)
    return [str(p.resolve()) for p in span[::step][:max_frames]]


def build_records(roots: list[SampleRoot], use_day: bool = False):
    lmms_records = []
    vlmeval_rows = []
    stats = {
        "datasets": {},
        "total_scenes": 0,
        "total_questions": 0,
        "use_day": use_day,
    }

    global_idx = 0
    split_name = "day" if use_day else "night"
    split_dir_name = "Day" if use_day else "Night"

    for item in roots:
        scene_dirs = find_scene_dirs(item.root)
        stats["datasets"][item.dataset_name] = {
            "scenes": len(scene_dirs),
            "questions": 0,
        }
        stats["total_scenes"] += len(scene_dirs)

        for scene_dir in scene_dirs:
            qa_file = scene_dir / "qa_result" / "all_qa_filtered.json"
            frame_dir = scene_dir / "extracted_frames" / split_dir_name
            if not qa_file.exists() or not frame_dir.exists():
                continue

            frame_files = find_frame_files(frame_dir)
            if not frame_files:
                continue

            with qa_file.open("r", encoding="utf-8") as f:
                qa_data = json.load(f)

            scene_name = scene_dir.name
            for qa in qa_data:
                question = str(qa.get("question", "")).strip()
                answer = str(qa.get("answer", "")).strip()
                category = str(qa.get("question_type", "Unknown")).strip()
                start_frame = safe_int(qa.get("start_frame", 0), 0)
                end_frame = safe_int(qa.get("end_frame", start_frame), start_frame)

                if not question:
                    continue

                sample_id = f"{item.dataset_name}/{scene_name}/{global_idx}"
                prompt = VIDEO_PROMPT_TEMPLATE.format(sample_fps=item.sample_fps, question=question)

                lmms_records.append(
                    {
                        "id": sample_id,
                        "dataset": item.dataset_name,
                        "scene": scene_name,
                        "split": split_name,
                        "question_type": category,
                        "question": question,
                        "answer": answer,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "sample_fps": item.sample_fps,
                        "frame_dir": str(frame_dir.resolve()),
                        "images": sample_frame_paths(frame_files, start_frame, end_frame, max_frames=16),
                        "prompt": prompt,
                    }
                )

                vlmeval_rows.append(
                    {
                        "index": global_idx,
                        "image_path": pick_representative_image(frame_files, start_frame, end_frame),
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "split": split_name,
                        "dataset": item.dataset_name,
                        "scene": scene_name,
                        "sample_id": sample_id,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "sample_fps": item.sample_fps,
                        "frame_dir": str(frame_dir.resolve()),
                    }
                )

                global_idx += 1
                stats["total_questions"] += 1
                stats["datasets"][item.dataset_name]["questions"] += 1

    return lmms_records, vlmeval_rows, stats


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        headers = [
            "index",
            "image_path",
            "question",
            "answer",
            "category",
            "split",
            "dataset",
            "scene",
            "sample_id",
            "start_frame",
            "end_frame",
            "sample_fps",
            "frame_dir",
        ]
    else:
        headers = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Build EgoNight exports for LMMs-Eval and VLMEvalKit")
    parser.add_argument("--oxford", type=str, required=True, help="Path to EgoNight_Oxford")
    parser.add_argument("--sofia", type=str, required=True, help="Path to EgoNight_Sofia")
    parser.add_argument("--synthetic", type=str, required=True, help="Path to EgoNight_synthetic")
    parser.add_argument("--output_dir", type=str, default="exports/generated", help="Output directory")
    parser.add_argument("--use_day", action="store_true", help="Use extracted_frames/Day if available")
    args = parser.parse_args()

    roots = [
        SampleRoot("oxford", Path(args.oxford), sample_fps=1),
        SampleRoot("sofia", Path(args.sofia), sample_fps=1),
        SampleRoot("synthetic", Path(args.synthetic), sample_fps=2),
    ]

    lmms_records, vlmeval_rows, stats = build_records(roots, use_day=args.use_day)

    output_dir = Path(args.output_dir)
    lmms_jsonl = output_dir / "egonight_lmms_eval.jsonl"
    vlmeval_tsv = output_dir / "EgoNight.tsv"
    stats_json = output_dir / "egonight_export_stats.json"

    write_jsonl(lmms_jsonl, lmms_records)
    write_tsv(vlmeval_tsv, vlmeval_rows)
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    with stats_json.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Wrote LMMs-Eval JSONL: {lmms_jsonl}")
    print(f"Wrote VLMEvalKit TSV: {vlmeval_tsv}")
    print(f"Wrote stats JSON: {stats_json}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
