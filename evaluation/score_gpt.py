import json
import os
import re
import ast
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from api_keys import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
MAX_WORKERS = 1



def create_prompt(q, a, pred):
    return f"""role: "system",
content: "You are an intelligent chatbot designed for evaluating the correctness of AI assistant predictions for question-answer pairs.
Your task is to compare the predicted answer with the ground-truth answer and determine if the predicted answer is correct or not. Here's how you can accomplish the task:
-----##INSTRUCTIONS:
- Focus on the correctness and accuracy of the predicted answer with the ground-truth.
- Consider uncertain predictions, such as 'it is impossible to answer the question from the video', as incorrect, unless the ground truth answer also says that."
role: "user",
content: "Please evaluate the following video-based question-answer pair:
Question: {q}
Ground truth correct Answer: {a}
Predicted Answer: {pred}
Provide your evaluation as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness. For question that counting the number of objects, if the predicted answer fells in the range of the ground truth answer, it should be considered as correct.
Please generate the response in the form of a Python dictionary string with keys 'pred', 'score' and 'reason', where value of 'pred' is a string of 'correct' or 'incorrect',
value of 'score' is in INTEGER, not STRING and value of 'reason' should provide the reason behind the decision."
"""


def evaluate_with_gpt(qa_item):
    question = qa_item['Q']
    answer = qa_item['C']
    pred = qa_item['A']

    prompt = create_prompt(question, answer, pred)

    for attempt in range(5):
        try:
            messages = [
                {
                    "role": "user",
                    "content": 
                    [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            response = client.chat.completions.create(model='gpt-4o', messages=messages)
            match = re.search(r'\{.*?\}', response.choices[0].message.content, re.DOTALL)
            if match:
                eval_dict = ast.literal_eval(match.group(0))
                return {
                    "pred": eval_dict.get("pred", ""),
                    "score": int(eval_dict.get("score", 0)),
                    "reason": eval_dict.get("reason", "")
                }
            # If no match, raise to retry
            raise ValueError("No dictionary found in response.")
        except Exception as e:
            print(f"Error evaluating (attempt {attempt+1}/5): {prompt} {e}")
            time.sleep(2 * attempt)
    return None


def evaluate_predictions(input_path, output_path):
    with open(input_path, "r") as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} QA pairs from {input_path}")

    results = []
    for idx, qa_item in enumerate(tqdm(raw_data, total=len(raw_data))):
        result = evaluate_with_gpt(qa_item)
        results.append([result, raw_data[idx]])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluated results to {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, required=True, help='Input directory containing result json files')
    args = parser.parse_args()
    qa_result_dir = os.path.join(args.dir_path, "qa_result")
    input_dir = qa_result_dir
    output_dir = qa_result_dir
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if "result" in file_name and file_name.endswith(".json"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name.replace("results", "scores"))
            # INSERT_YOUR_CODE
            if os.path.exists(output_path):
                try:
                    with open(output_path, "r") as f:
                        existing_results = json.load(f)
                    with open(input_path, "r") as f:
                        input_results = json.load(f)
                    # Check for None items in existing_results
                    if (
                        len(existing_results) == len(input_results)
                        and all(item is not None and item[0] is not None for item in existing_results)
                    ):
                        print(f"Skipping {output_path} as it already exists, matches input length, and contains no None items.")
                        continue
                except Exception as e:
                    print(f"Error checking {output_path}: {e}")
            evaluate_predictions(input_path, output_path)


if __name__ == "__main__":
    main()