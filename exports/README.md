# EgoNight Exports for LMMs-Eval and VLMEvalKit

This folder contains integration assets to finish the two pending benchmark TODO items:

- **LMMs-Eval** task scaffold and data export format
- **VLMEvalKit** custom dataset scaffold and data export format

## 1) Build export files from local EgoNight data

```bash
python exports/build_egonight_exports.py \
  --oxford /path/to/EgoNight_Oxford \
  --sofia /path/to/EgoNight_Sofia \
  --synthetic /path/to/EgoNight_synthetic \
  --output_dir exports/generated
```

Outputs:

- `exports/generated/egonight_lmms_eval.jsonl`
- `exports/generated/EgoNight.tsv`
- `exports/generated/egonight_export_stats.json`

Use `--use_day` to export day split (if `extracted_frames/Day` exists).

## 2) LMMs-Eval integration

Copy `exports/lmms_eval_task/egonight/` into `lmms-eval/lmms_eval/tasks/egonight/`.

Then edit `egonight.yaml` and set:

- `dataset_kwargs.data_files` to your local JSONL path (for example `exports/generated/egonight_lmms_eval.jsonl`)

Example run (inside `lmms-eval` repo):

```bash
python -m lmms_eval \
  --model openai_compatible \
  --model_args model=Qwen/Qwen2.5-VL-7B-Instruct,base_url=http://127.0.0.1:8004/v1/chat/completions,api_key=EMPTY \
  --tasks egonight \
  --batch_size 1
```

## 3) VLMEvalKit integration

### Step 1: Prepare VLMEvalKit

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

### Step 2: Add EgoNight dataset class

Copy `exports/vlmevalkit/egonight_dataset.py` into `VLMEvalKit/vlmeval/dataset/`.

Then edit `VLMEvalKit/vlmeval/dataset/__init__.py`:

1. Add import near other dataset imports:

```python
from .egonight_dataset import EgoNight
```

2. Add `EgoNight` to `CUSTOM_DATASET`:

```python
CUSTOM_DATASET = [
    ...,
    EgoNight,
]
```

### Step 3: Place exported TSV where VLMEvalKit can find it

VLMEvalKit loads custom TSV benchmarks from its `LMUData` root (`LMUDataRoot()`).
Copy the generated TSV there with filename `EgoNight.tsv`:

```bash
cp /path/to/EgoNight.tsv /path/to/LMUData/EgoNight.tsv
```

If needed, print your LMUData root:

```bash
python -c "from vlmeval.smp import LMUDataRoot; print(LMUDataRoot())"
```

### Step 4: Smoke-test dataset loading (verified)

Run this in `VLMEvalKit` to verify registration + prompt construction:

```bash
python -c "from vlmeval.dataset.egonight_dataset import EgoNight; ds=EgoNight(); print('rows', len(ds.data)); msg=ds.build_prompt(0); print('msg_items', len(msg)); print('last_type', msg[-1]['type'])"
```

Expected:
- `rows` should be non-zero
- `msg_items` should include image entries and one final text entry

### Step 5: Run evaluation

Use the standard VLMEvalKit runner:

```bash
python run.py --data EgoNight --model <YOUR_MODEL_NAME>
```

For API-based models, configure VLMEvalKit environment variables/API settings according to its Quickstart docs before running `run.py`.

This EgoNight dataset class consumes:

- `frame_dir`
- `start_frame`
- `end_frame`
- `sample_fps`
- `question`

from each TSV row, and builds a multi-frame prompt for VQA-style inference.
