"""Embedding function abstraction for Wren Memory.

Supports two embedding systems:

1. **LanceDB's native registry** (via get_registry()):
   - sentence-transformers (local, no API key) — default
   - openai, gemini-text, cohere, ollama, jina, voyageai,
     watsonx, huggingface, bedrock-text, and more...
   Set WREN_EMBEDDING_PROVIDER to the registry name.

2. **LangChain Embeddings** (any LangChain-compatible embedder):
   - GoogleGenerativeAIEmbeddings, OllamaEmbeddings, CohereEmbeddings, etc.
   - Set WREN_EMBEDDING_PROVIDER=langchain and
     WREN_EMBEDDING_MODEL=<module>.<ClassName>[|kwarg1=val1,...]
   - e.g. WREN_EMBEDDING_MODEL=langchain_google_genai.GoogleGenerativeAIEmbeddings|model=models/embedding-001

Configuration via environment variables:
  WREN_EMBEDDING_PROVIDER  — "sentence-transformers" | "gemini-text" | "langchain" | ...
  WREN_EMBEDDING_MODEL     — model name or LangChain class path
  WREN_EMBEDDING_DIM        — override vector dimension (optional, auto-detected)
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Union

import numpy as np

_DEFAULT_MODEL = os.getenv("WREN_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
_DEFAULT_PROVIDER = os.getenv("WREN_EMBEDDING_PROVIDER", "sentence-transformers")

# Known dimensions per model (avoids API calls for dimension detection)
# Maps provider → model_name → dimension
_KNOWN_DIMENSIONS: dict[str, dict[str, int]] = {
    "sentence-transformers": {
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
    },
    "openai": {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    },
    "gemini-text": {
        "models/embedding-001": 768,
        "text-embedding-004": 768,
    },
    "cohere": {
        "embed-english-v3.0": 1024,
        "embed-english-v3": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-multilingual-v3": 1024,
    },
    "jina": {
        "jina-clip-v1": 768,
        "jina-embeddings-v2": 768,
        "jina-embeddings-v2-en": 768,
    },
    "voyageai": {
        "voyage-law-2": 1024,
        "voyage-code-2": 1536,
        "voyage-lite-2": 1024,
    },
    "ollama": {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    },
    "gte-text": {
        "gte-Qwen2-7B-instruct": 3584,
        "gte-large-en-v1.5": 1024,
    },
    "bedrock-text": {
        "amazon.titan-embed-text-v1": 1536,
        "amazon.titan-embed-text-v2": 1536,
        "cohere.embed-english-v3": 1024,
        "cohere.embed-multilingual-v3": 1024,
    },
    "huggingface": {
        "sentence-transformers": 768,
    },
    # LangChain providers — add here when verified
    "langchain": {
        "langchain_google_genai.GoogleGenerativeAIEmbeddings|model=models/embedding-001": 768,
        "langchain_ollama.OllamaEmbeddings|model=nomic-embed-text": 768,
        "langchain_openai.OpenAIEmbeddings|model=text-embedding-3-small": 1536,
        "langchain_cohere.CohereEmbeddings|model=embed-english-v3.0": 1024,
    },
}


def _to_float_lists(embeddings: Any) -> list[list[float]]:
    """Normalize embedding output to list[list[float]].

    LangChain embedders return different shapes:
      - embed_query(str)  -> list[float] or list[list[float]]
      - embed_documents   -> list[list[float]]

    LanceDB/Arrow needs list[list[float]]. This normalizes all cases.
    """
    if isinstance(embeddings, np.ndarray):
        embeddings = embeddings.tolist()

    if not isinstance(embeddings, list):
        # unexpected type — wrap single value
        return [[float(v) for v in embeddings]]

    if len(embeddings) == 0:
        return []

    first = embeddings[0]

    # Already 2D list of floats
    if isinstance(first, list):
        return [[float(v) for v in row] for row in embeddings]

    # 1D list of floats (flat list from embed_query) — wrap in outer list
    if isinstance(first, (int, float)):
        return [[float(v) for v in embeddings]]

    # 1D numpy array — wrap in outer list
    if isinstance(first, np.floating):
        return [[float(v) for v in embeddings]]

    # 2D numpy array
    if isinstance(first, np.ndarray):
        return [[float(v) for v in row] for row in embeddings]

    return [[float(v) for v in row] for row in embeddings]


class LangChainEmbeddingWrapper:
    """Wraps any LangChain ``Embeddings`` object to satisfy LanceDB's
    ``EmbeddingFunction`` interface used by Wren MemoryStore.

    Parameters
    ----------
    langchain_embeddings:
        An instance (not a class) of a LangChain Embeddings implementation.
        The instance must already be configured with any required API keys,
        model names, or other init kwargs.

    Example
    -------
    >>> from langchain_google_genai import GoogleGenerativeAIEmbeddings
    >>> lc = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    >>> wrapper = LangChainEmbeddingWrapper(lc)
    >>> wrapper.compute_source_embeddings(["hello world"])
    [[0.123, -0.456, ...]]
    >>> wrapper.compute_query_embeddings("hello")
    [[0.123, -0.456, ...]]
    """

    __slots__ = ("_lc", "_ndims")

    def __init__(self, langchain_embeddings):
        self._lc = langchain_embeddings
        self._ndims: int | None = None

    def compute_source_embeddings(
        self, texts: Union[list[str], np.ndarray]
    ) -> list[list[float]]:
        """Embed a batch of texts using LangChain's embed_documents.

        Handles the case where the LangChain embedder returns a single
        embedding for a batch (e.g. Gemini's embed_content with batched
        inputs returns 1 embedding instead of N).
        """
        raw = self._lc.embed_documents(texts)
        normalized = _to_float_lists(raw)

        # If we got 1 embedding but expected N texts, embed one-by-one
        if len(normalized) == 1 and len(texts) > 1:
            normalized = [self._lc.embed_query(t) for t in texts]
            normalized = _to_float_lists(normalized)

        return normalized

    def compute_query_embeddings(self, query: str) -> list[list[float]]:
        """Embed a single query using LangChain's embed_query.

        Handles three return shape conventions:
          - list[float]  (most providers)
          - list[list[float]]  (some providers return 2D even for 1 item)
          - np.ndarray
        """
        raw = self._lc.embed_query(query)
        return _to_float_lists(raw)

    def ndims(self) -> int:
        """Return the embedding dimension, probing once if unknown."""
        if self._ndims is not None:
            return self._ndims
        try:
            probe = self._lc.embed_query("dimension_probe")
            normalized = _to_float_lists(probe)
            self._ndims = len(normalized[0])
        except Exception:
            self._ndims = 768  # safe default for many embedding models
        return self._ndims


def _parse_langchain_model(model_name: str) -> tuple[str, dict]:
    """Parse model name into (class_path, init_kwargs).

    Format: "module.ClassName|kwarg1=val1,kwarg2=val2"
    The |... suffix is optional. Spaces around '=' and ',' are trimmed.

    Examples
    --------
    >>> _parse_langchain_model("langchain_google_genai.GoogleGenerativeAIEmbeddings|model=models/embedding-001")
    ("langchain_google_genai.GoogleGenerativeAIEmbeddings", {"model": "models/embedding-001"})

    >>> _parse_langchain_model("langchain_openai.OpenAIEmbeddings")
    ("langchain_openai.OpenAIEmbeddings", {})
    """
    kwargs: dict = {}
    config_str = model_name

    if "|" in model_name:
        config_str, kwargs_str = model_name.split("|", 1)
        for pair in kwargs_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            kwargs[k.strip()] = v.strip()

    return config_str, kwargs


def get_embedding_function(model_name: str | None = None, provider: str | None = None):
    """Return an embedding function for the given provider and model.

    Provider ``langchain`` uses LangChain's ``Embeddings`` interface
    instead of LanceDB's native registry. This enables any LangChain-
    compatible embedder (GoogleGenerativeAI, Ollama, Cohere, HuggingFace,
    VoyageAI, etc.).

    For the ``langchain`` provider, ``model_name`` uses the format:
        "<module>.<ClassName>[|kwarg1=val1,kwarg2=val2]"

    e.g. ``langchain_google_genai.GoogleGenerativeAIEmbeddings|model=models/embedding-001``

    For all other providers (LanceDB registry), ``model_name`` is the
    model name string passed to the registry's create() method.

    Parameters
    ----------
    model_name:
        Model identifier. Defaults to WREN_EMBEDDING_MODEL env var.
    provider:
        Provider name. Defaults to WREN_EMBEDDING_PROVIDER env var.
        Supported: "sentence-transformers" (default), "openai",
        "gemini-text", "cohere", "ollama", "jina", "voyageai",
        "watsonx", "bedrock-text", "huggingface", "gte-text",
        "langchain" (LangChain Embeddings interface).

    Returns
    -------
    An object with ``compute_source_embeddings(texts)`` and
    ``compute_query_embeddings(query)`` methods returning list[list[float]].
    """
    import lancedb.embeddings  # noqa: PLC0415

    provider = provider or os.getenv("WREN_EMBEDDING_PROVIDER") or _DEFAULT_PROVIDER
    model_name = model_name or os.getenv("WREN_EMBEDDING_MODEL") or _DEFAULT_MODEL

    if provider == "langchain":
        class_path, kwargs = _parse_langchain_model(model_name)

        # Try each possible import path variation
        errors = []
        for import_path in (class_path, class_path.replace("_google_genai", "_google_genai"),):
            try:
                mod_path, class_name = import_path.rsplit(".", 1)
            except ValueError:
                errors.append(f"Invalid class path: {import_path}")
                continue

            try:
                mod = __import__(mod_path, fromlist=[class_name])
            except ImportError as exc:
                errors.append(
                    f"Could not import '{mod_path}'. "
                    f"Install the required package (e.g. `pip install langchain-google-genai`). "
                    f"Original error: {exc}"
                )
                continue

            lc_cls = getattr(mod, class_name, None)
            if lc_cls is None:
                errors.append(f"Class '{class_name}' not found in module '{mod_path}'")
                continue

            try:
                lc_instance = lc_cls(**kwargs)
            except TypeError as exc:
                errors.append(
                    f"Failed to instantiate {import_path} with kwargs={kwargs}. "
                    f"Check that all required init args are provided via WREN_EMBEDDING_MODEL. "
                    f"Original error: {exc}"
                )
                continue

            return LangChainEmbeddingWrapper(lc_instance)

        # All import paths failed — surface the most useful error
        raise ImportError(
            f"Could not load LangChain embedding class '{class_path}'. "
            f"Please ensure the package is installed. "
            f"Tried errors: {'; '.join(errors)}"
        )

    # All other providers: use LanceDB's native embedding registry
    registry = lancedb.embeddings.get_registry()
    embedder = registry.get(provider).create(name=model_name)
    return embedder


@contextlib.contextmanager
def suppress_stderr():
    """Temporarily redirect stderr to /dev/null.

    Suppresses noisy native output (progress bars, load reports) from
    sentence-transformers / candle during model loading.
    """
    old_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)


def warm_up(embed_fn) -> int:
    """Return the vector dimension of the embedding function.

    For known models (from _KNOWN_DIMENSIONS), returns dimension without
    making an API call. For unknown models, probes the embedder once.
    """
    from lancedb.embeddings.base import EmbeddingFunction

    provider = os.getenv("WREN_EMBEDDING_PROVIDER", _DEFAULT_PROVIDER)
    model_name = os.getenv("WREN_EMBEDDING_MODEL", _DEFAULT_MODEL)

    # 1. Known dimension from lookup table — no API call needed
    if provider in _KNOWN_DIMENSIONS and model_name in _KNOWN_DIMENSIONS[provider]:
        return _KNOWN_DIMENSIONS[provider][model_name]

    # 2. Explicit override via env var
    override = os.getenv("WREN_EMBEDDING_DIM")
    if override:
        return int(override)

    # 3. For LangChain wrapper, use its ndims() (probes once internally)
    if isinstance(embed_fn, LangChainEmbeddingWrapper):
        return embed_fn.ndims()

    # 4. For LanceDB EmbeddingFunction classes with ndims()
    if isinstance(embed_fn, EmbeddingFunction) and hasattr(embed_fn, "ndims"):
        try:
            return int(embed_fn.ndims())
        except Exception:
            pass

    # 5. Fallback: actual inference probe
    with suppress_stderr():
        probe = embed_fn.compute_source_embeddings(["probe"])
    normalized = _to_float_lists(probe)
    return len(normalized[0])


def default_dimension() -> int:
    """Return the default vector dimension for the default model."""
    return _KNOWN_DIMENSIONS.get("sentence-transformers", {}).get(
        "paraphrase-multilingual-MiniLM-L12-v2", 384
    )


def list_providers() -> list[str]:
    """List all registered embedding providers in LanceDB plus 'langchain'."""
    import lancedb.embeddings  # noqa: PLC0415

    registry = lancedb.embeddings.get_registry()
    names = []
    if hasattr(registry, "_functions"):
        names = list(registry._functions.keys())
    names.append("langchain")
    return names


def list_models(provider: str | None = None) -> list[str]:
    """List known model names for a given provider."""
    provider = provider or os.getenv("WREN_EMBEDDING_PROVIDER", _DEFAULT_PROVIDER)

    if provider == "langchain":
        return list(_KNOWN_DIMENSIONS.get("langchain", {}).keys())

    known = _KNOWN_DIMENSIONS.get(provider, {})
    if known:
        return list(known.keys())

    try:
        import lancedb.embeddings  # noqa: PLC0415

        registry = lancedb.embeddings.get_registry()
        cls = registry.get(provider)
        if hasattr(cls, "model_names"):
            return cls.model_names()
    except Exception:
        pass

    return []