dir_path="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

subfolders=()
for subdir in "$dir_path"/*/; do
    subfolders+=("$subdir")
done

for subfolder in "${subfolders[@]}"; do
    python "$SCRIPT_DIR/evaluate_qwen7b.py" --dir_path "$subfolder" &
    python "$SCRIPT_DIR/evaluate_gpt.py" --dir_path "$subfolder" &
    python "$SCRIPT_DIR/evaluate_gemini.py" --dir_path "$subfolder"
    python "$SCRIPT_DIR/score_gpt.py" --dir_path "$subfolder"
done