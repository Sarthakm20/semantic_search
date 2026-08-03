"""Embedding fetching for your semantic search system.

Two modes, controlled by the LAB3_EMBEDDING_MODE environment variable:

    "offline" (default) - a hashed bag-of-words vector. Fully implemented
                below already - you don't need to write or even fully
                understand this part, it's provided so you can build and
                test your whole pipeline with no API key and no network.
    "api"     - calls NVIDIA NIM's embeddings endpoint. THIS is the part
                you implement - see get_embedding() below.

Both modes return a plain Python list of floats, so the rest of your code
never needs to know which one is in use.
"""

import hashlib
import math
import os
import re

import requests
from dotenv import load_dotenv

load_dotenv()  # picks up NVIDIA_API_KEY and LAB3_EMBEDDING_MODE from a .env file, if present

EMBEDDING_MODE = os.environ.get("LAB3_EMBEDDING_MODE", "offline")  # "api" or "offline"

API_KEY = os.environ.get("NVIDIA_API_KEY")
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
EMBEDDING_URL = "https://integrate.api.nvidia.com/v1/embeddings"

OFFLINE_DIM = 64
_WORD_RE = re.compile(r"[a-z']+")


def _get_embedding_offline(text, dim=OFFLINE_DIM):
    vec = [0.0] * dim
    words = _WORD_RE.findall(text.lower())
    for word in words:
        bucket = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % dim
        vec[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _get_embedding_api(text, input_type="passage"):
    if not API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Copy .env.example to .env and fill in "
            "your key from Lab 1 (or regenerate one at "
            "https://build.nvidia.com/settings/api-keys)."
        )

    if input_type not in ("passage", "query"):
        raise ValueError(f"input_type must be 'passage' or 'query', got {input_type!r}")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "input": [text],
        "model": EMBEDDING_MODEL,
        "input_type": input_type,
    }

    try:
        response = requests.post(EMBEDDING_URL, headers=headers, json=body, timeout=30)
    except requests.exceptions.RequestException as e:
        # Covers connection errors, timeouts, DNS failures, etc. - anything
        # that means the request never got a response at all.
        raise RuntimeError(f"Could not reach the NVIDIA NIM embeddings API: {e}") from e

    if response.status_code == 401:
        raise RuntimeError(
            "NVIDIA API key was rejected (401 Unauthorized). It may be invalid "
            "or expired."
        )
    if response.status_code == 429:
        raise RuntimeError(
            "NVIDIA API rate limit hit (429 Too Many Requests). Wait a bit and "
            "try again, or rely on the embeddings cache for documents already fetched."
        )
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"NVIDIA NIM embeddings API returned an error: {e}") from e

    try:
        data = response.json()
        return data["data"][0]["embedding"]
    except (KeyError, IndexError, ValueError) as e:
        raise RuntimeError(
            f"Unexpected response shape from the embeddings API: {e}. "
            f"Response body: {response.text[:500]}"
        ) from e


_api_unavailable = False


def get_embedding(text, input_type="passage"):
    global _api_unavailable

    if EMBEDDING_MODE == "api" and not _api_unavailable:
        try:
            return _get_embedding_api(text, input_type=input_type)
        except RuntimeError as e:
            _api_unavailable = True
            print(
                f"[warning] API embedding failed, falling back to offline mode "
                f"for the rest of this session: {e}"
            )

    return _get_embedding_offline(text)
