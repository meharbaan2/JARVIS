import hashlib
import math
from collections import Counter


DEFAULT_HASH_DIMENSIONS = 96


def stable_tokens(text):
    import re

    return [token.lower() for token in re.findall(r"[a-zA-Z0-9_+#.-]+", text or "") if len(token) > 1]


class EmbeddingProvider:
    name = "base"
    dimensions = 0

    def embed(self, text):
        raise NotImplementedError

    def health(self):
        return {
            "name": self.name,
            "available": True,
            "dimensions": self.dimensions,
        }


class HashEmbeddingProvider(EmbeddingProvider):
    """Dependency-free local embedding fallback using stable hashed lexical features."""

    def __init__(self, dimensions=DEFAULT_HASH_DIMENSIONS):
        self.name = "hash"
        self.dimensions = int(dimensions)

    def embed(self, text):
        tokens = stable_tokens(text)
        if not tokens:
            return [0.0] * self.dimensions

        features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        vector = [0.0] * self.dimensions
        for feature, count in Counter(features).items():
            vector[self._bucket(feature)] += 1.0 + math.log(count)

        return _normalize(vector)

    def _bucket(self, token):
        digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimensions


class TransformersEmbeddingProvider(EmbeddingProvider):
    """Optional local HF/transformers embedding provider with lazy model loading."""

    def __init__(self, model_path, device="auto", max_length=512):
        self.name = f"transformers:{model_path}"
        self.model_path = model_path
        self.device = device
        self.max_length = int(max_length)
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._load_error = None
        self.dimensions = 0

    def embed(self, text):
        if not self._ensure_loaded():
            raise RuntimeError(f"Transformers embedding provider is unavailable: {self._load_error}")

        torch = self._torch
        encoded = self._tokenizer(
            text or "",
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        encoded = {key: value.to(self._model.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self._model(**encoded)
        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = summed / counts
        vector = pooled[0].detach().cpu().float().tolist()
        self.dimensions = len(vector)
        return _normalize(vector)

    def health(self):
        return {
            "name": self.name,
            "available": self._load_error is None,
            "loaded": self._model is not None,
            "dimensions": self.dimensions,
            "model_path": self.model_path,
            "device": self.device,
            "error": self._load_error,
        }

    def _ensure_loaded(self):
        if self._model is not None and self._tokenizer is not None:
            return True
        if self._load_error:
            return False
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self._model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
            self._model.to(device)
            self._model.eval()
            self._torch = torch
            hidden_size = getattr(getattr(self._model, "config", None), "hidden_size", None)
            self.dimensions = int(hidden_size or 0)
            return True
        except Exception as exc:
            self._load_error = str(exc)
            return False


class FallbackEmbeddingProvider(EmbeddingProvider):
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name
        self.dimensions = primary.dimensions or fallback.dimensions
        self.last_provider_name = None

    def embed(self, text):
        try:
            vector = self.primary.embed(text)
            self.last_provider_name = self.primary.name
            self.dimensions = self.primary.dimensions or len(vector)
            return vector
        except Exception:
            vector = self.fallback.embed(text)
            self.last_provider_name = self.fallback.name
            self.name = self.fallback.name
            self.dimensions = self.fallback.dimensions
            return vector

    def health(self):
        primary_health = self.primary.health()
        fallback_health = self.fallback.health()
        active = self.last_provider_name or (self.primary.name if primary_health.get("loaded") else self.fallback.name)
        return {
            "name": active,
            "available": True,
            "dimensions": self.primary.dimensions if active == self.primary.name else self.fallback.dimensions,
            "primary": primary_health,
            "fallback": fallback_health,
        }


def create_embedding_provider(settings):
    provider = str(getattr(settings, "embedding_provider", "hash") or "hash").lower()
    hash_dimensions = int(getattr(settings, "embedding_hash_dimensions", DEFAULT_HASH_DIMENSIONS))
    fallback = HashEmbeddingProvider(hash_dimensions)
    if provider in {"hash", "local_hash"}:
        return fallback
    if provider in {"transformers", "hf", "local_transformers"}:
        model_path = getattr(settings, "embedding_model", "")
        if not model_path:
            return fallback
        primary = TransformersEmbeddingProvider(
            model_path=model_path,
            device=getattr(settings, "embedding_device", "auto"),
            max_length=getattr(settings, "embedding_max_length", 512),
        )
        return FallbackEmbeddingProvider(primary, fallback)
    return fallback


def _normalize(vector):
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [round(value / norm, 6) for value in vector]
