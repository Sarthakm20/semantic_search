# Semantic Search

A small semantic search engine. It takes a corpus of short text documents, turns
each one into an embedding vector, and lets you search the corpus by *meaning*
rather than by exact keyword match, using cosine similarity. It also projects
the embedding space down to 2D with PCA so the topic clusters can be seen
visually.

## What's in this repo

| File | Purpose |
|---|---|
| `embeddings.py` | Fetches an embedding for a piece of text, either from NVIDIA NIM's API or from a local offline fallback. |
| `search.py` | Loads the corpus, builds/caches the embedding matrix, runs cosine-similarity search, and does the PCA projection. |
| `documents.json` | The corpus — 25 short factual documents across 5 topics: astronomy, cooking, geography, music theory, sports. |
| `semantic_search_starter.ipynb` | The notebook that runs the pipeline end to end and produces the search results and the PCA plot. |
| `embeddings_cache.json` | Auto-generated on first run. Caches embeddings so they aren't re-fetched from the API every time. Safe to delete to force a full re-embed. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
pip install -r requirements.txt

```

Make a file `.env` and fill in these details 

```
NVIDIA_API_KEY=nvapi-your-key-here
LAB3_EMBEDDING_MODE=offline
```

## Running it

Open `semantic_search_starter.ipynb` and run all cells. 
Cell 1 loads the corpus and builds the embedding matrix. 
Cell 2 runs a couple of example search queries. 
Cell 3 produces the PCA scatter plot.

### Switching between offline and API mode

This is a one-line change, no code edits needed — just set `LAB3_EMBEDDING_MODE`
in `.env`:

```
LAB3_EMBEDDING_MODE=offline   # no key, no network — for building/testing logic
LAB3_EMBEDDING_MODE=api       # real NVIDIA NIM embeddings — for real results
```

**After changing `.env`, restart the notebook kernel and rerun all cells from
the top.**

- **Offline mode** uses a hashed bag-of-words vector (no network, no key
  required). It's useful for testing that the pipeline runs correctly, but it
  only matches on shared vocabulary, not meaning — search results in this mode
  are not expected to be semantically relevant, and the PCA plot's clusters
  will look far weaker or scrambled compared to API mode. This is expected
  behavior, not a bug.
- **API mode** calls NVIDIA NIM's `nv-embedqa-e5-v5` embedding model for real
  semantic embeddings. This is the mode that produces the actual deliverables
  below.

## Example search results (API mode)

```
Query: 'How do black holes form?'
  [0.261] (astronomy) Neutron stars are so dense that a teaspoon of their material would weigh billions of tons.
  [0.245] (astronomy) Saturn's rings are made mostly of ice particles, with a smaller amount of rocky debris.
  [0.210] (astronomy) A light-year measures distance, not time - it's the distance light travels in one year.

Query: "What's the best way to cook a steak?"
  [0.362] (cooking) Searing meat at high heat creates a browned crust through the Maillard reaction, adding flavor.
  [0.342] (cooking) Resting meat after cooking allows the juices to redistribute instead of spilling out when cut.
  [0.295] (cooking) Blanching vegetables briefly in boiling water then ice water helps them keep a bright color.
```

Both queries return top-3 results entirely from the correct topic, with scores
descending sensibly — confirming the embeddings and ranking are working as
intended, not just running without error.

## PCA visualization

The embedding matrix (documents × 1024 dimensions) is projected down to 2
principal components with a from-scratch SVD-based PCA (reused from Lab 2,
unmodified), then plotted with each point colored by its document's topic.

All five topics form visually distinct, non-overlapping regions in the 2D
projection.

**Is the clustering good?** Yes — the final plot shows all five topics forming
tight, non-overlapping regions, with essentially no points from one topic
sitting inside another topic's group.Clean separation here demonstrates the pipeline is
working correctly, but it doesn't prove the embeddings would separate
topics that are inherently closer together in meaning that would be a harder and
more informative test of the model's discriminative power than this corpus
was designed to be.


## Caching

Embeddings are cached in `embeddings_cache.json`, keyed by document `id`. A
cached entry is only reused if both its stored text and the mode it was
computed in (`offline`/`api`) still match the current document — so editing a
document's text, or switching modes, correctly triggers a re-fetch for just
the affected entries rather than silently returning stale vectors, and
without needing to re-fetch the entire corpus. Delete the cache file at any
time to force a full re-embed.
