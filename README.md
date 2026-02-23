# EgoNight

EgoNight is the first comprehensive benchmark designed to evaluate egocentric vision understanding in low-light and nighttime conditions—a critical gap in current research.

## Overview

The benchmark assesses vision-language models on egocentric video question answering across diverse scenarios. It supports both **night** (default) and **day** imagery, with a curated subset of question types for paired day/night comparison.

### Question Types

- Object Recognition
- Spatial Reasoning
- Scene Sequence
- Non Common
- Counting
- Navigation
- Text Recognition
- Action

---

## Project Structure

```
EgoNight/
├── evaluation/
│   ├── evaluate_gemini.py    # Gemini 2.5 Pro inference
│   ├── evaluate_gpt.py       # GPT-4.1 inference
│   ├── evaluate_qwen7b.py    # Qwen 2.5 VL 7B inference (local API)
│   ├── score_gpt.py          # GPT-4o as judge scoring (correct/incorrect, 0–5)
│   ├── summarize_accuracy.py # Per-dataset and overall accuracy summary
│   ├── evaluate_all.sh       # Batch evaluation over subfolders
│   ├── api_keys.py           # API key loading
│   ├── keys.env.example      # Template for API keys
│   └── keys.env              # Your keys (create from example, gitignored)
├── README.md
└── LICENSE
```

---

## Data Format

Each evaluation sample is a subfolder with:

```
<subfolder>/
├── qa_result/
│   ├── all_qa_filtered.json       # Question-answer annotations
│   ├── *_results*.json            # Model outputs (gpt, gemini, qwen7b)
│   └── *_scores*.json             # Score outputs (created by score_gpt.py)
└── extracted_frames/
    ├── Night/                     # Night images (jpg/png)
    └── Day/                       # Day images (optional)
```

### Frame sampling

- **EgoNight-Sofia** and **EgoNight-Oxford**: frames sampled at 1 fps
- **EgoNight-Synthetic**: frames sampled at 2 fps

Evaluators infer the dataset from the path and use the correct sampling rate in the prompt.

### all_qa_filtered.json

List of objects with fields:

| Field          | Description                          |
|----------------|--------------------------------------|
| `question`     | The question text                    |
| `question_type`| One of the question types above      |
| `answer`       | Ground-truth answer                  |
| `start_frame`  | First frame index (0-based)          |
| `end_frame`    | Last frame index (inclusive)         |

---

## Setup

### 1. Dependencies

```bash
pip install openai google-generativeai tqdm pyyaml numpy pillow requests
```

### 2. API Keys (GPT & Gemini)

1. Copy the example keys file:
   ```bash
   cp evaluation/keys.env.example evaluation/keys.env
   ```
2. Edit `evaluation/keys.env` with your keys:
   ```
   OPENAI_API_KEY=sk-your-openai-key
   GEMINI_API_KEY=your-gemini-api-key
   ```

   Alternatively, set `OPENAI_API_KEY` and `GEMINI_API_KEY` as environment variables.

### 3. Qwen 2.5 VL 7B (Optional)

`evaluate_qwen7b.py` expects a local API server at `http://localhost:8004` serving `qwen2.5-vl-7b-instruct`. Start your inference server before running that evaluator.

---

## Usage

### Single Sample

```bash
# GPT-4.1 (night images)
python evaluation/evaluate_gpt.py --dir_path /path/to/sample_folder

# GPT-4.1 (day images)
python evaluation/evaluate_gpt.py --dir_path /path/to/sample_folder --use_day True

# Gemini 2.5 Pro
python evaluation/evaluate_gemini.py --dir_path /path/to/sample_folder

# Qwen 7B (requires local server on port 8004)
python evaluation/evaluate_qwen7b.py --dir_path /path/to/sample_folder
```

### Batch Evaluation

Provide a parent directory containing one subfolder per sample:

```bash
bash evaluation/evaluate_all.sh /path/to/parent_directory
```

This runs the active evaluators (GPT, Gemini, Qwen7b) in parallel per sample, then scores results with GPT-4o. For `sofia_oxford`, scores are written to each subfolder’s `score/` directory.

### Scoring

`score_gpt.py` takes prediction JSONs (filenames containing `result` and `.json`), compares them to ground truth via GPT-4o, and writes scored files (`results` → `scores` in the filename):

```bash
python evaluation/score_gpt.py --dir_path /path/to/results_directory
```

### Summarize Accuracy

`summarize_accuracy.py` computes per-dataset (Sofia, Oxford, Synthetic) and overall accuracy by QA type. It reads `*_scores_*.json` from each subfolder and filters by `all_qa_filtered.json` (Sofia/Oxford) or `qa_human.json` (Synthetic).

```bash
python evaluation/summarize_accuracy.py \
    --sofia_path /path/to/Sofia_server \
    --oxford_path /path/to/Oxford_server \
    --synthetic_path /path/to/synthetic_server \
    [--model gpt] [--split night] [--output summary.json]
```

| Option | Description |
|--------|-------------|
| `--sofia_path` | Path to Sofia_server parent directory |
| `--oxford_path` | Path to Oxford_server parent directory |
| `--synthetic_path` | Path to synthetic_server parent directory |
| `--model` | Model name: `gpt`, `gemini`, `qwen7b`, etc. (default: `gpt`) |
| `--split` | `night` or `day` (default: `night`) |
| `--output` | Path to save JSON summary |
| `--oxford_score_dir` | Score directory for Oxford: `qa_result` or `final_score` (default: `qa_result`) |

---

## Output

| Evaluator | Output File           |
|-----------|------------------------|
| GPT       | `gpt_results.json` / `gpt_results_day.json` |
| Gemini    | `gemini_results.json` / `gemini_results_day.json` |
| Qwen 7B   | `qwen7b_results.json` / `qwen7b_results_day.json` |

Each result JSON contains entries with `Q` (question), `A` (prediction), `C` (ground truth), `M` (category), and frame indices.

Scoring produces `*_scores.json` with GPT-4o evaluations: correct/incorrect, 0–5 score, and reasoning.

---

## License

GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
