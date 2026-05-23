# LangChain Embedding Support for Wren AI

## Overview

This document describes the changes made to Wren AI's embedding system to support any LangChain-compatible embedding model via the `WREN_EMBEDDING_PROVIDER` and `WREN_EMBEDDING_MODEL` environment variables.

## What Changed

### Before

Wren AI only supported embedding models from **LanceDB's native registry** (sentence-transformers by default). Switching embedders required code changes.

### After

Wren AI now supports:

1. **LanceDB native registry** (unchanged behavior)
2. **Any LangChain `Embeddings` class** — Google Gemini, OpenAI, Cohere, Ollama, HuggingFace, Azure OpenAI, and more

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WREN_EMBEDDING_PROVIDER` | `"sentence-transformers"` or `"langchain"` | `"sentence-transformers"` |
| `WREN_EMBEDDING_MODEL` | Model name or LangChain class path | `"paraphrase-multilingual-MiniLM-L12-v2"` |
| `WREN_EMBEDDING_DIM` | Override vector dimension (optional) | Auto-detected |

### LangChain Model String Format

For LangChain providers, use the format:

```
<module>.<ClassName>|kwarg1=val1,kwarg2=val2
```

Examples:

| Provider | Model String |
|----------|--------------|
| Google Gemini | `langchain_google_genai.GoogleGenerativeAIEmbeddings\|model=models/gemini-embedding-001,output_dimensionality=768` |
| Google Gemini v2 | `langchain_google_genai.GoogleGenerativeAIEmbeddings\|model=models/gemini-embedding-2-preview,output_dimensionality=768` |
| OpenAI | `langchain_openai.OpenAIEmbeddings\|model=text-embedding-3-small` |
| Ollama | `langchain_ollama.OllamaEmbeddings\|model=nomic-embed-text,base_url=http://localhost:11434` |
| Cohere | `langchain_cohere.CohereEmbeddings\|model=embed-english-v3.0` |

## Usage

### 1. Set Environment Variables

```bash
# Example: Using Gemini embeddings
export WREN_EMBEDDING_PROVIDER=langchain
export WREN_EMBEDDING_MODEL=langchain_google_genai.GoogleGenerativeAIEmbeddings|model=models/gemini-embedding-001,output_dimensionality=768
export GOOGLE_API_KEY=your_api_key_here

# Example: Using Ollama (local)
export WREN_EMBEDDING_PROVIDER=langchain
export WREN_EMBEDDING_MODEL=langchain_ollama.OllamaEmbeddings|model=nomic-embed-text,base_url=http://localhost:11434
```

### 2. Index Memory

```bash
wren memory index
```

### 3. Query

```bash
wren memory fetch --query "show me customers"
wren --sql "SELECT * FROM customers LIMIT 10"
```

## Key Implementation Details

### LangChainEmbeddingWrapper

Located in `wren/memory/embeddings.py`, this wrapper:

1. Wraps any LangChain `Embeddings` instance
2. Exposes `compute_source_embeddings(texts)` and `compute_query_embeddings(query)` interface required by LanceDB
3. Normalizes all return shapes to `list[list[float]]`
4. Handles batch fallback when provider returns 1 embedding for N texts

### _to_float_lists Normalization

LangChain embedders return different shapes:
- `embed_query(str)` → `list[float]` or `list[list[float]]`
- `embed_documents` → `list[list[float]]`

This function normalizes all cases to `list[list[float]]` for LanceDB compatibility.

### Batch Fallback Fix

Some embedding providers (e.g., `gemini-embedding-2-preview`) return a single aggregated embedding for batch inputs instead of N embeddings. The wrapper detects this and falls back to individual `embed_query()` calls:

```python
if len(normalized) == 1 and len(texts) > 1:
    normalized = [self._lc.embed_query(t) for t in texts]
```

### _KNOWN_DIMENSIONS Lookup

Pre-populated dimension lookup for 40+ models across 12 providers. Avoids API calls for dimension detection:

```python
_KNOWN_DIMENSIONS = {
    "gemini-text": {
        "models/embedding-001": 768,
        "models/gemini-embedding-2-preview": 768,
    },
    "langchain": {
        "langchain_google_genai.GoogleGenerativeAIEmbeddings|model=models/gemini-embedding-001": 768,
        # ...
    },
    # ...
}
```

## File Changes

| File | Changes |
|------|---------|
| `wren/memory/embeddings.py` | Added `LangChainEmbeddingWrapper`, `_to_float_lists()`, `_parse_langchain_model()`, provider dispatch for langchain |
| `wren/memory/store.py` | Updated imports to use `default_dimension()` instead of hardcoded `_DEFAULT_DIM` |

## Supported Providers

### LanceDB Native Registry
- sentence-transformers (default)
- openai, gemini-text, cohere, ollama, jina, voyageai
- watsonx, huggingface, bedrock-text, gte-text

### LangChain (any Embeddings class)
- `langchain_google_genai` — Google Gemini
- `langchain_openai` — OpenAI
- `langchain_ollama` — Ollama (local)
- `langchain_cohere` — Cohere
- `langchain_huggingface` — HuggingFace
- `langchain_azure_ai` — Azure OpenAI
- And any other LangChain-supported embedder

## Known Issues

### gemini-embedding-2-preview Batch Behavior

This model returns 1 embedding for N texts in batch mode. The wrapper handles this via fallback to individual calls, but indexing is slower (N API calls instead of batch).

**Workaround:** Use `gemini-embedding-001` for faster batch indexing.

## Resetting Memory

If you change the embedding model or dimension, reset the vector index:

```bash
rm -rf .wren/memory
wren memory index
```

## CLI Commands

All Wren memory commands work with LangChain embeddings:

```bash
wren memory index        # Index schema with embeddings
wren memory status      # Check indexed tables
wren memory fetch -q "..."  # Semantic search
wren memory recall -q "..." # Recall past queries
wren memory store --nl "..." --sql "..."  # Store query
```