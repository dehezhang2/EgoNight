"""
Load API keys from keys.env file or environment variables.
Copy keys.env.example to keys.env and fill in your API keys.
"""
import os

# Path to keys file: same directory as this module
_KEYS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keys.env")


def _load_keys_from_file(path: str) -> dict[str, str]:
    """Parse KEY=VALUE lines from a file. Skips empty lines and comments."""
    result = {}
    if not os.path.isfile(path):
        return result
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key:
                    result[key] = value
    return result


_keys = _load_keys_from_file(_KEYS_FILE)

OPENAI_API_KEY = _keys.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = _keys.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
