import os
import base64
import json
from tqdm import tqdm
import yaml
import numpy as np
import time
import requests

# Load configuration
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load config at module level
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
parser.add_argument("--dir_path", type=str, default="", help="Path to directory")
parser.add_argument("--use_day", type=bool, default=False, help="Use day images")

args = parser.parse_args()
dir_path = args.dir_path

paired_types = ["Object Recognition", "Spatial Reasoning", "Scene Sequence", "Non Common", "Counting", "Navigation", "Text Recognition", "Action"]

def read_images_from_directory(dir_path):
    # Read all jpg/jpeg/png files from the directory and return as list of bytes
    files = sorted([
        f for f in os.listdir(dir_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    image_buffers = []
    for file in files:
        file_path = os.path.join(dir_path, file)
        with open(file_path, "rb") as img_file:
            # Downsample the image to half of its previous resolution before appending to image_buffers
            from PIL import Image
            from io import BytesIO
            img = Image.open(img_file)
            width, height = img.size
            img = img.resize((max(1, width // 2), max(1, height // 2)))
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            image_buffers.append(buffer.getvalue())
    return image_buffers



class QADataset:
    def __init__(self, data_file, use_day=False):
        with open(data_file, 'r') as f:
            annotations = json.load(f)
        if use_day:
            # Only keep questions whose type is in paired_types
            self.annotations = [row for row in annotations if row.get('question_type') in paired_types]
        else:
            self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations[index]
        question = row['question']
        category = row['question_type']
        answer = row['answer']
        start_idx = row['start_frame']
        end_idx = row['end_frame']
        question_str = (
            f"Please carefully read the question, use the visual cues to answer the question: {question}."
            "The original FPS of the video is 30. This image set is obtained by sampling at 1 fps."
            "Do not include any other content."
            f"You need to answer the question in any case and not demand additional context information. "
            f"Note: All actions mentioned refer to the person recording the video.\n\n"
        )

        return {
            'question_answer': question_str,
            'question': question,
            'answer': answer,
            'category': category,
            'start_idx': start_idx,
            'end_idx': end_idx
        }

def call_qwen_model(start_idx, end_idx, image_buffers, prompt):

        end_idx = min(end_idx + 1, len(image_buffers))

        image_buffers_used = image_buffers
        
        for attempt_id in range(5):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": 
                       [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                                }
                            }
                            for buffer in image_buffers_used[start_idx:end_idx]
                        ] +  [
                            {"type": "text", "text": prompt}
                        ] 
                    }
                ]
                response = requests.post(
                    "http://localhost:8004/v1/chat/completions",
                    json={
                        "model": "qwen2.5-vl-7b-instruct",
                        "messages": messages,
                    }
                )
                break
            except Exception as e:
                print(f"Error processing {prompt}: {e}")
                wait_time = 5*attempt_id
                time.sleep(wait_time)
        return  response.json()["choices"][0]["message"]["content"]

def process_qa_item(batch, existing_entries, image_buffers):
    question = batch['question']
    question_answer = batch['question_answer']
    category = batch['category']
    answer = batch['answer']

    if (question) in existing_entries:
        return None

    try:
        output_text = call_qwen_model(batch['start_idx'], batch['end_idx'], image_buffers, question_answer)
        print(output_text)
    except Exception as e:
        print(f"Error processing {question}: {e}")
        return None

    return {
        "Q": question,
        "A": output_text,
        "C": answer,
        "M": category,
        "start_idx": batch['start_idx'], 
        "end_idx": batch['end_idx']
    }

from concurrent.futures import ThreadPoolExecutor, as_completed
def perform_bulk_inference(dataset, output_file_path, image_buffers):
    results = []
    existing_entries = set()

    if os.path.exists(output_file_path):
        with open(output_file_path, 'r') as f:
            try:
                existing_data = json.load(f)
                results = [entry for entry in existing_data if entry["A"] != ""]
                existing_entries = {(entry["V"], entry["Q"]) for entry in results}
            except Exception as e:
                print("Error loading previous results:", e)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(process_qa_item, dataset[i], existing_entries, image_buffers): i
            for i in range(len(dataset))
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Running Inference"):
            result = future.result()
            if result:
                results.append(result)
                if len(results) % 50 == 0:
                    with open(output_file_path, 'w') as f:
                        json.dump(results, f)

    with open(output_file_path, 'w') as f:
        json.dump(results, f)
    print(f"Saved {len(results)} results to {output_file_path}")

def main():
    if args.use_day:
        print("Using day images")
    image_dir = os.path.join(dir_path, f"extracted_frames/{'Day' if args.use_day else 'Night'}")
    image_buffers = read_images_from_directory(image_dir)
    print(len(image_buffers))

    input_file = os.path.join(dir_path, "qa_result", "all_qa_filtered.json")
    
    dataset = QADataset(input_file, args.use_day)

    output_file_path = os.path.join(dir_path, "qa_result", 'qwen7b_results' + ('_day' if args.use_day else '_night') + '.json')
    if os.path.exists(output_file_path):
        try:
            with open(output_file_path, 'r') as f:
                existing_data = json.load(f)
            if len(existing_data) >= len(dataset):
                print(f"{output_file_path} already exists and has {len(existing_data)} entries (dataset has {len(dataset)}). Skipping inference.")
                return
            else:
                print(f"{output_file_path} exists but has only {len(existing_data)} entries (dataset has {len(dataset)}). Will continue inference.")
        except Exception as e:
            print(f"Error reading {output_file_path}: {e}. Will continue inference.")
    perform_bulk_inference(dataset, output_file_path, image_buffers)

if __name__ == "__main__":
    main()