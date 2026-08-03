"""Corpus loading and embedding matrix construction for semantic search."""

import json
import os

import numpy as np

from embeddings import get_embedding, EMBEDDING_MODE

CACHE_PATH = "embeddings_cache.json"


def load_documents(path):
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find documents file at {path!r}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            documents = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path!r} is not valid JSON: {e}") from e

    seen_ids = set()
    for i, doc in enumerate(documents):
        for field in ("id", "topic", "text"):
            if field not in doc or not str(doc[field]).strip():
                raise ValueError(
                    f"Document at index {i} is missing a non-empty {field!r} field: {doc}"
                )
        if doc["id"] in seen_ids:
            raise ValueError(f"Duplicate document id {doc['id']!r} - ids must be unique.")
        seen_ids.add(doc["id"])

    return documents


def _load_cache():
    
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(
                f"[warning] {CACHE_PATH} is corrupted and could not be read - "
                f"ignoring it and rebuilding the cache from scratch."
            )
            return {}


def _save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def build_embedding_matrix(documents):

    cache = _load_cache()
    cache_dirty = False

    vectors = []
    for doc in documents:
        doc_id = doc["id"]
        text = doc["text"]

        cached_entry = cache.get(doc_id)
        cache_hit = (
            cached_entry is not None
            and cached_entry.get("text") == text
            and cached_entry.get("mode") == EMBEDDING_MODE
        )

        if cache_hit:
            vector = cached_entry["embedding"]
        else:
            vector = get_embedding(text, input_type="passage")
            cache[doc_id] = {
                "text": text,
                "mode": EMBEDDING_MODE,
                "embedding": vector,
            }
            cache_dirty = True

        vectors.append(vector)

    if cache_dirty:
        _save_cache(cache)

    return np.array(vectors)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(query, embedding_matrix, documents, top_k=5):

    query_vector = get_embedding(query, input_type="query")

    scores = np.array([
        cosine_similarity(query_vector, doc_vector)
        for doc_vector in embedding_matrix
    ])

    top_indices = np.argsort(-scores)[:top_k]

    return [(documents[i], scores[i]) for i in top_indices]


def pca_via_svd(data, n_components):
    
    mean = np.mean(data, axis=0)
    centered = data - mean
    U, S, Vt = np.linalg.svd(centered)
    components = Vt[:n_components]
    projected = centered @ components.T
    return projected, components, mean