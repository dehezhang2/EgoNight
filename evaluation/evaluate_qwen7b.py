import argparse
import base64
import time

import requests

from common import run_evaluation_main

QWEN_API_BASE_URL = "http://localhost:8004"

parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", type=str, default="", help="Path to directory")
parser.add_argument("--use_day", type=bool, default=False, help="Use day images")
parser.add_argument(
    "--api_url",
    type=str,
    default=QWEN_API_BASE_URL,
    help="Qwen API base URL (default: http://localhost:8004)",
)

args = parser.parse_args()


def call_qwen_model(start_idx, end_idx, image_buffers, prompt, api_url=None):
    api_url = (api_url or args.api_url).rstrip("/")
    end_idx = min(end_idx + 1, len(image_buffers))
    for attempt_id in range(5):
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}",
                            },
                        }
                        for buffer in image_buffers[start_idx:end_idx]
                    ]
                    + [{"type": "text", "text": prompt}],
                }
            ]
            response = requests.post(
                f"{api_url}/v1/chat/completions",
                json={
                    "model": "qwen2.5-vl-7b-instruct",
                    "messages": messages,
                },
            )
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error processing {prompt}: {e}")
            time.sleep(5 * attempt_id)


if __name__ == "__main__":
    run_evaluation_main(
        args.dir_path,
        args.use_day,
        f"qwen7b_results_{'day' if args.use_day else 'night'}",
        call_qwen_model,
        api_url=args.api_url,
    )
