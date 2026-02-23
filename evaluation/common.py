"""
Shared evaluation utilities for EgoNight: QADataset, image loading, FPS, bulk inference.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

PAIRED_TYPES = [
    "Object Recognition",
    "Spatial Reasoning",
    "Scene Sequence",
    "Non Common",
    "Counting",
    "Navigation",
    "Text Recognition",
    "Action",
]


def get_sample_fps(dir_path: str) -> int:
    """Infer sample FPS from path: synthetic uses 2 fps, Sofia/Oxford use 1 fps."""
    path_lower = dir_path.lower()
    return 2 if "synthetic" in path_lower else 1


def read_images_from_directory(dir_path: str, target_size=None):
    """
    Read jpg/jpeg/png files from directory, optionally resize, return list of bytes.
    target_size: (w, h) for fixed size, or None for half resolution.
    """
    files = sorted([
        f for f in os.listdir(dir_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    image_buffers = []
    for file in files:
        file_path = os.path.join(dir_path, file)
        with open(file_path, "rb") as img_file:
            from io import BytesIO

            from PIL import Image

            img = Image.open(img_file)
            width, height = img.size
            if target_size:
                w, h = target_size
                img = img.resize((max(1, w), max(1, h)))
            else:
                img = img.resize((max(1, width // 2), max(1, height // 2)))
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            image_buffers.append(buffer.getvalue())
    return image_buffers


class QADataset:
    def __init__(self, data_file: str, use_day: bool = False, sample_fps: int = 1):
        with open(data_file, "r") as f:
            annotations = json.load(f)
        if use_day:
            self.annotations = [
                row for row in annotations
                if row.get("question_type") in PAIRED_TYPES
            ]
        else:
            self.annotations = annotations
        self.sample_fps = sample_fps

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations[index]
        question = row["question"]
        category = row["question_type"]
        answer = row["answer"]
        start_idx = row["start_frame"]
        end_idx = row["end_frame"]
        question_str = (
            f"Please carefully read the question, use the visual cues to answer the question: {question}."
            f"The original FPS of the video is 30. This image set is obtained by sampling at {self.sample_fps} fps."
            "Do not include any other content."
            "You need to answer the question in any case and not demand additional context information. "
            "Note: All actions mentioned refer to the person recording the video.\n\n"
        )
        return {
            "question_answer": question_str,
            "question": question,
            "answer": answer,
            "category": category,
            "start_idx": start_idx,
            "end_idx": end_idx,
        }


def perform_bulk_inference(
    dataset,
    output_file_path: str,
    image_buffers: list,
    call_model_fn,
    **call_model_kwargs,
):
    """
    Run inference over dataset. call_model_fn(start_idx, end_idx, image_buffers, prompt, **kwargs) -> str.
    """
    results = []
    existing_entries = set()

    if os.path.exists(output_file_path):
        with open(output_file_path, "r") as f:
            try:
                existing_data = json.load(f)
                results = [entry for entry in existing_data if entry.get("A") != ""]
                existing_entries = {entry["Q"] for entry in results}
            except Exception as e:
                print("Error loading previous results:", e)

    def process_item(batch, existing, buffers):
        question = batch["question"]
        if question in existing:
            return None
        try:
            output_text = call_model_fn(
                batch["start_idx"],
                batch["end_idx"],
                buffers,
                batch["question_answer"],
                **call_model_kwargs,
            )
            print(output_text)
        except Exception as e:
            print(f"Error processing {question}: {e}")
            return None
        return {
            "Q": question,
            "A": output_text,
            "C": batch["answer"],
            "M": batch["category"],
            "start_idx": batch["start_idx"],
            "end_idx": batch["end_idx"],
        }

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(process_item, dataset[i], existing_entries, image_buffers): i
            for i in range(len(dataset))
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running Inference"):
            result = future.result()
            if result:
                results.append(result)
                existing_entries.add(result["Q"])
                if len(results) % 50 == 0:
                    with open(output_file_path, "w") as f:
                        json.dump(results, f)

    with open(output_file_path, "w") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} results to {output_file_path}")


def run_evaluation_main(
    dir_path: str,
    use_day: bool,
    output_suffix: str,
    call_model_fn,
    input_file: str = None,
    read_images_kwargs: dict = None,
    **call_model_kwargs,
):
    """Common main flow: load images, build dataset, optionally skip if complete, run inference."""
    if use_day:
        print("Using day images")
    image_dir = os.path.join(dir_path, f"extracted_frames/{'Day' if use_day else 'Night'}")
    read_kw = read_images_kwargs or {}
    image_buffers = read_images_from_directory(image_dir, **read_kw)
    print(len(image_buffers))

    input_file = input_file or os.path.join(dir_path, "qa_result", "all_qa_filtered.json")
    sample_fps = get_sample_fps(dir_path)
    dataset = QADataset(input_file, use_day, sample_fps=sample_fps)

    output_file_path = os.path.join(dir_path, "qa_result", f"{output_suffix}.json")
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, "r") as f:
                existing_data = json.load(f)
            if len(existing_data) >= len(dataset):
                print(
                    f"{output_file_path} already exists and has {len(existing_data)} entries "
                    f"(dataset has {len(dataset)}). Skipping inference."
                )
                return
            print(
                f"{output_file_path} exists but has only {len(existing_data)} entries "
                f"(dataset has {len(dataset)}). Will continue inference."
            )
        except Exception as e:
            print(f"Error reading {output_file_path}: {e}. Will continue inference.")

    perform_bulk_inference(
        dataset, output_file_path, image_buffers, call_model_fn, **call_model_kwargs
    )
