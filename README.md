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
