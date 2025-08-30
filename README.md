# Code RAG Service

This repository implements a simplified code retrieval-augmented generation service using LlamaIndex and Qdrant.

## Development

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

## Running

```bash
uvicorn rag_service.main:app --reload
```

## Reranking

To re-order retrieval results with a cross-encoder model, enable the reranker in the
configuration:

```json
{
  "llamaindex": {
    "use": true,
    "retrieval": {
      "use_reranker": true
    }
  }
}
```

The reranker uses the ``cross-encoder/ms-marco-MiniLM-L-6-v2`` model from the
``sentence-transformers`` package.

## Retrievers

The service provides two retriever strategies selected via configuration:

- simple: Baseline that queries code chunks, file cards, and optionally directory cards in parallel,
  applies query rewriting/expansions, fuses results, and optionally reranks.
- smart: File-first strategy that retrieves file cards, boosts them with query metadata, focuses code retrieval
  within the selected files, then fuses and optionally reranks. Designed to improve relevance by combining
  file descriptions and code.

Enable the smart retriever:

```json
{
  "llamaindex": {
    "use": true,
    "retrieval": {
      "retriever": "smart"
    }
  }
}
```

### Neighbor expansion

Both retrievers can expand code results with immediate neighbors (prev/next chunks) to provide more complete
context. This is controlled by two parameters in ``llamaindex.retrieval``:

- neighbor_decay: Multiplier applied to the source node score for neighbor chunks (default: 0.9).
- neighbor_limit: Maximum number of neighbors added per source code node (default: 2).

Example configuration:

```json
{
  "llamaindex": {
    "use": true,
    "retrieval": {
      "retriever": "smart",
      "neighbor_decay": 0.9,
      "neighbor_limit": 2
    }
  }
}
```

Note: Neighbor expansion relies on code nodes storing their own ``node_id`` plus ``prev_id``/``next_id`` in metadata.
If you indexed the repository with an older version, re-run indexing to populate these fields in Qdrant.

## API Usage

### Index a repository

Include an optional ``repo_prompt`` to describe the project. This text is used when generating
summaries for files and directories.

```bash
curl -X POST http://localhost:8000/v1/index \
    -H "Content-Type: application/json" \
    -d '{"root_path": ".", "clean": false, "repo_prompt": "Project context"}'
```

### Query indexed data

```bash
curl -X POST http://localhost:8000/v1/query \
    -H "Content-Type: application/json" \
    -d '{"q": "example question", "top_k": 3}'
```
