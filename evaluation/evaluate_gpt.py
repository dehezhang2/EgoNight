import argparse
import base64
import time

from openai import OpenAI

from api_keys import OPENAI_API_KEY
from common import run_evaluation_main

parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", type=str, default="", help="Path to directory")
parser.add_argument("--use_day", type=bool, default=False, help="Use day images")

args = parser.parse_args()
client = OpenAI(api_key=OPENAI_API_KEY)


def call_gpt_model(start_idx, end_idx, image_buffers, prompt):
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
            response = client.chat.completions.create(
                model="gpt-4.1", messages=messages, temperature=0, stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error processing {prompt}: {e}")
            time.sleep(5 * attempt_id)


if __name__ == "__main__":
    run_evaluation_main(
        args.dir_path,
        args.use_day,
        f"gpt_results_{'day' if args.use_day else 'night'}",
        call_gpt_model,
        read_images_kwargs={"target_size": (540, 960)},
    )
