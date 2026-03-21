<p align="center">
    <img src="fig/icon.png" alt="EgoNight" width="120">
</p>

<div align="center">

<h1 align="center">EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark</h1>

<h4 align="center"><em>Deheng Zhang*, Yuqian Fu*✉, Runyi Yang, Yang Miao, Tianwen Qian, Xu Zheng,</em></h4>
<h4 align="center"><em>Guolei Sun, Ajad Chhatkuli, Xuanjing Huang, Yu-Gang Jiang, Luc Van Gool, Danda Pani Paudel</em></h4>

<p align="center">
    <img src="fig/institute.png" alt="Institutions" width="600">
</p>

\* *Equal Contribution* &nbsp; &nbsp; *Corresponding Author* ✉

</div>

<p align="center">
    <a href="https://arxiv.org/abs/2510.06218"><img src="https://img.shields.io/badge/arXiv-2510.06218-b31b1b.svg" alt="arXiv"></a>
    <a href="https://openreview.net/forum?id=DKD4QbOKBN"><img src="https://img.shields.io/badge/ICLR-2026-blue.svg" alt="ICLR 2026"></a>
    <a href="https://huggingface.co/datasets/dehezhang2/EgoNight"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-EgoNight-yellow.svg" alt="Hugging Face"></a>
</p>

<p align="center">
  <a href="#news">News</a> |
  <a href="#dataset">Dataset</a> |
  <a href="#overview">Overview</a> |
  <a href="#todo">TODO</a> |
  <a href="#acknowledgement">Acknowledgement</a> |
  <a href="#citation">Citation</a> |
  <a href="#license">License</a>
</p>

## News

- **[2026]** EgoNight is accepted by **ICLR 2026**. 
- **[2025/10]** Our paper "[EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark](https://arxiv.org/abs/2510.06218)" is available on [arXiv](https://arxiv.org/abs/2510.06218).
- **[2025/10]** Paper and supplementary materials available on [OpenReview](https://openreview.net/forum?id=DKD4QbOKBN).



---

## Overview

The benchmark assesses vision-language models on egocentric video question answering across diverse scenarios. It supports both **night** (default) and **day** imagery, with a curated subset of question types for paired day/night comparison. Open-source VLMs (GLM-4V, Qwen2.5-VL, InternVL, LLaVA-NeXT-Video, etc.) are evaluated using the [LLaMA Factory](https://github.com/hiyouga/LlamaFactory) API server.

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

## TODO

### Done

- [x] **VQA evaluation** — Full pipeline for EgoNight-VQA (GPT, Gemini, Qwen, scoring, summarization).
- [x] **[LLaMA Factory](https://github.com/hiyouga/LlamaFactory)** — Open-source VLM evaluation via API server (GLM-4V, Qwen2.5-VL, InternVL, LLaVA-NeXT-Video).

### Planned

- [ ] **Depth evaluation** — Evaluation pipeline for egocentric depth estimation at night (auxiliary task from the paper).
- [ ] **Retrieval evaluation** — Evaluation pipeline for day–night correspondence retrieval (auxiliary task from the paper).
- [ ] **[LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)** — Integrate EgoNight as a task in the unified multimodal evaluation toolkit.
- [ ] **[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)** — Add EgoNight benchmark support to the OpenCompass VLM evaluation toolkit.

---

## Project Structure

```
EgoNight/
├── evaluation/
│   ├── evaluate_gemini.py    # Gemini 2.5 Pro inference
│   ├── evaluate_gpt.py       # GPT-4.1 inference
│   ├── evaluate_qwen7b.py    # Qwen 2.5 VL 7B inference (LLaMA Factory API)
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

### 3. Open-Source VLMs via LLaMA Factory (Optional)

Part of the open-source VLM evaluation relies on the [LLaMA Factory](https://github.com/hiyouga/LlamaFactory?tab=readme-ov-file#deploy-with-openai-style-api-and-vllm) API server. Others are evaluated using the official repo. Start the API server with the desired model before running the corresponding evaluator. 

Example configs for supported VLMs (based on LLaMA Factory [`examples/inference/`](https://github.com/hiyouga/LlamaFactory/tree/main/examples/inference)):

| Model | Config | Hugging Face Model |
|-------|--------|--------------------|
| **GLM-4V** | `glm4v.yaml` | `zai-org/GLM-4.1V-9B-Base` |
| **Qwen2.5-VL-7B** | `qwen2_5vl_7B.yaml` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| **InternVL3** | `intern_vl.yaml` | `OpenGVLab/InternVL3-8B-hf` |
| **LLaVA-NeXT-Video** | `llava_video.yaml` | `llava-hf/LLaVA-NeXT-Video-7B-32K-hf` |

**Example configs** (save as YAML and run `llamafactory-cli api <config.yaml>`):

```yaml
# glm4v.yaml
model_name_or_path: zai-org/GLM-4.1V-9B-Base
template: glm4v
infer_backend: huggingface
trust_remote_code: true
```

```yaml
# qwen2_5vl_7B.yaml
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct
template: qwen2_vl
infer_backend: huggingface  # choices: [huggingface, vllm, sglang]
trust_remote_code: true
```

```yaml
# intern_vl.yaml
model_name_or_path: OpenGVLab/InternVL3-8B-hf
template: intern_vl
infer_backend: huggingface
trust_remote_code: true
```

```yaml
# llava_video.yaml
model_name_or_path: llava-hf/LLaVA-NeXT-Video-7B-32K-hf
template: llava_next_video
infer_backend: huggingface
trust_remote_code: true
```

Start the API server (default port 8000):
```bash
API_PORT=8000 llamafactory-cli api examples/inference/qwen2_5vl_7B.yaml
```

### 4. Qwen 2.5 VL 7B (Optional)

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

## Acknowledgement

We thank the following projects and resources:

- **[Oxford day and night dataset](https://oxdan.active.vision/)** for providing day and night egocentric sequences used in EgoNight-Oxford.
- **[LLaMA Factory](https://github.com/hiyouga/LlamaFactory)** for the unified API server enabling efficient evaluation of open-source vision-language models.
- **[Blender](https://www.blender.org/)** for the open-source 3D creation suite used to render synthetic day–night aligned videos in EgoNight-Synthetic.
- **[Infinigen](https://github.com/princeton-vl/infinigen)**  for the indoor scene generation used in EgoNight-Synthetic
---

## Citation

If you find EgoNight useful for your research, please cite our paper:

```bibtex
@inproceedings{zhang2026egonight,
  title={EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark},
  author={Zhang, Deheng and Fu, Yuqian and Yang, Runyi and Miao, Yang and Qian, Tianwen and Zheng, Xu and Sun, Guolei and Chhatkuli, Ajad and Huang, Xuanjing and Jiang, Yu-Gang and Van Gool, Luc and Paudel, Danda Pani},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://openreview.net/forum?id=DKD4QbOKBN}
}
```

---

## License

GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
