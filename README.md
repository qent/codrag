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

## API Usage

### Index a repository

```bash
curl -X POST http://localhost:8000/v1/index \
    -H "Content-Type: application/json" \
    -d '{"root_path": ".", "clean": false}'
```

### Query indexed data

```bash
curl -X POST http://localhost:8000/v1/query \
    -H "Content-Type: application/json" \
    -d '{"q": "example question", "top_k": 3}'
```

