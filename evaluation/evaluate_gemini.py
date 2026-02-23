import argparse
import base64
import time

import google.generativeai as genai

from api_keys import GEMINI_API_KEY
from common import run_evaluation_main

parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", type=str, default="", help="Path to directory")
parser.add_argument("--use_day", type=bool, default=False, help="Use day images")

args = parser.parse_args()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-pro")


def call_gemini_model(start_idx, end_idx, image_buffers, prompt):
    end_idx = min(end_idx + 1, len(image_buffers))
    for attempt_id in range(5):
        try:
            messages = [
                {
                    "parts": [
                        {
                            "mime_type": "image/jpeg",
                            "data": base64.b64encode(buffer).decode("utf-8"),
                        }
                        for buffer in image_buffers[start_idx:end_idx]
                    ]
                    + [{"text": prompt}],
                }
            ]
            response = model.generate_content(contents=messages)
            return response.text
        except Exception as e:
            print(f"Error processing {prompt}: {e}")
            time.sleep(5 * attempt_id)


if __name__ == "__main__":
    run_evaluation_main(
        args.dir_path,
        args.use_day,
        f"gemini_results_{'day' if args.use_day else 'night'}",
        call_gemini_model,
    )
