"""Gemini wrapper: text generation, structured (JSON) output and embeddings.

Design notes
- Every call is cached on disk (see cache.py) -> fast, repeatable demos and
  friendlier to free-tier rate limits.
- Retries with exponential backoff on 429 / 5xx errors.
- If the configured model is unavailable, falls back through a list of models.
- If no API key is configured, `available` is False and callers switch to
  their deterministic offline fallbacks - the app never hard-fails.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Type

from pydantic import BaseModel

from . import config
from .cache import DiskCache, key_for
from .guardrails import sanitize_llm_output

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def _sanitize_tree(obj: Any) -> Any:
    if isinstance(obj, str):
        return sanitize_llm_output(obj)
    if isinstance(obj, list):
        return [_sanitize_tree(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_tree(v) for k, v in obj.items()}
    return obj


_NETWORK_HINTS = ("connecterror", "connection refused", "getaddrinfo", "ssl", "certificate", "name resolution", "network is unreachable")
_BUSY_HINTS = ("500", "503", "504", "unavailable", "overloaded", "high demand", "deadline", "timed out", "timeout")
BUSY_COOLDOWN_S = 600  # a model that timed out / was overloaded is skipped for 10 minutes

# Corporate proxies / antivirus often re-sign HTTPS traffic with a certificate that is trusted by
# Windows/macOS but not by Python's bundled CA list -> CERTIFICATE_VERIFY_FAILED. truststore makes
# Python use the operating-system certificate store instead (same thing pip does).
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - optional dependency
    pass


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        # api_key=None -> use configured key; api_key="" -> force offline mode
        self.api_key = (config.GEMINI_API_KEY if api_key is None else api_key or "").strip()
        self.model = model or config.GEMINI_MODEL
        self.embed_model = config.GEMINI_EMBED_MODEL
        self._client = None
        self._gen_cache = DiskCache("llm_responses")
        self._emb_cache = DiskCache("embeddings")
        self.last_error: str | None = None
        self.notices: list[str] = []
        self._dead_models: set[str] = set()
        self._busy_until: dict[str, float] = {}
        if self.api_key:
            try:
                from google import genai

                from google.genai import types

                try:  # 30s timeout; our own loop handles retries/failover, so disable SDK-level retries
                    opts = types.HttpOptions(timeout=30_000, retry_options=types.HttpRetryOptions(attempts=1))
                except Exception:
                    opts = types.HttpOptions(timeout=30_000)
                self._client = genai.Client(api_key=self.api_key, http_options=opts)
            except Exception as exc:  # pragma: no cover - import/env issues
                self.last_error = f"Could not initialise Gemini client: {exc}"
                log.warning(self.last_error)

    # ------------------------------------------------------------------ status
    @property
    def available(self) -> bool:
        return self._client is not None

    def _models_to_try(self) -> list[str]:
        models = [self.model] + [m for m in config.GEMINI_FALLBACK_MODELS if m != self.model]
        return models

    # ------------------------------------------------------------- generation
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        schema: Type[BaseModel] | None = None,
        use_cache: bool = True,
    ) -> Any:
        """Generate text (or a parsed pydantic object when `schema` is given)."""
        if not self.available:
            raise LLMError("Gemini is not configured (no API key).")

        from google.genai import types

        # Model-agnostic key: an answer cached from a fallback model is still reused.
        cache_key = key_for("gen", system, prompt, temperature, schema.__name__ if schema else None)
        if use_cache:
            cached = self._gen_cache.get(cache_key)
            if cached is not None:
                return schema.model_validate(cached) if schema else cached

        cfg_kwargs: dict[str, Any] = {"temperature": temperature}
        if system:
            cfg_kwargs["system_instruction"] = system
        if schema:
            cfg_kwargs["response_mime_type"] = "application/json"
            cfg_kwargs["response_schema"] = schema
        cfg = types.GenerateContentConfig(**cfg_kwargs)

        last_exc: Exception | None = None
        for model in self._models_to_try():
            if model in self._dead_models or self._busy_until.get(model, 0) > time.time():
                continue
            for attempt in range(2):
                try:
                    resp = self._client.models.generate_content(model=model, contents=prompt, config=cfg)
                    if schema:
                        parsed = resp.parsed
                        if parsed is None:
                            parsed = schema.model_validate(json.loads(resp.text))
                        if isinstance(parsed, dict):
                            parsed = schema.model_validate(parsed)
                        # Sanitise every string field (exfiltration links, HTML) before anyone sees it
                        parsed = schema.model_validate(_sanitize_tree(parsed.model_dump()))
                        self._gen_cache.set(cache_key, parsed.model_dump())
                        self.model = model  # remember the model that worked
                        return parsed
                    text = sanitize_llm_output((resp.text or "").strip())
                    if not text:
                        raise LLMError("Empty response from model (possibly blocked by safety filters).")
                    self._gen_cache.set(cache_key, text)
                    self.model = model
                    return text
                except Exception as exc:
                    last_exc = exc
                    msg = str(exc)
                    low = msg.lower()
                    if "404" in msg or "NOT_FOUND" in msg or "no longer available" in low:
                        self._dead_models.add(model)  # retired / not enabled for this key -> never try again
                        break
                    if any(h in low for h in _BUSY_HINTS):
                        self._busy_until[model] = time.time() + BUSY_COOLDOWN_S
                        break  # model busy -> fail over to the next model immediately
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        if attempt == 0:
                            time.sleep(3)  # short back-off once, then try another model (separate quota)
                            continue
                        break
                    if "API key" in msg or "API_KEY_INVALID" in msg or "PERMISSION_DENIED" in msg or "401" in msg or "403" in msg:
                        self.last_error = "Gemini API key was rejected - switched to offline mode. Check the key in .env or the sidebar."
                        self._client = None  # stop hammering the API; everything degrades to offline fallbacks
                        self._notify(self.last_error)
                        raise LLMError(self.last_error) from exc
                    if any(s in low for s in _NETWORK_HINTS):
                        self.last_error = "Could not reach Gemini (network / SSL issue) - showing offline fallback."
                        self._notify(self.last_error)
                        raise LLMError(self.last_error) from exc  # every model would fail the same way
                    if attempt == 0:
                        continue  # malformed JSON / empty / safety-blocked output -> retry once
                    break
        msg = str(last_exc)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            self.last_error = "Gemini free-tier rate limit / daily quota reached - showing offline fallback. Try again in a minute."
        elif any(h in msg.lower() for h in _BUSY_HINTS):
            self.last_error = "All Gemini models are busy right now (high demand) - showing offline fallback. Try again shortly."
        else:
            self.last_error = f"Gemini call failed ({msg[:160]}) - showing offline fallback."
        self._notify(self.last_error)
        raise LLMError(self.last_error) from last_exc

    # ------------------------------------------------------------- notices
    def _notify(self, msg: str) -> None:
        if msg not in self.notices:
            self.notices.append(msg)

    def pop_notices(self) -> list[str]:
        """Return and clear user-facing warnings (shown as toasts in the UI)."""
        out, self.notices = self.notices, []
        return out

    # ------------------------------------------------------------- embeddings
    def embed(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        """Embed texts with Gemini embeddings (cached per text)."""
        if not self.available:
            raise LLMError("Gemini is not configured (no API key).")
        from google.genai import types

        results: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for i, t in enumerate(texts):
            hit = self._emb_cache.get(key_for("emb", self.embed_model, task_type, config.EMBED_DIM, t))
            if hit is not None:
                results[i] = hit
            else:
                missing.append(i)

        batch_size = 50
        for start in range(0, len(missing), batch_size):
            idxs = missing[start : start + batch_size]
            batch = [texts[i] for i in idxs]
            for attempt in range(4):
                try:
                    resp = self._client.models.embed_content(
                        model=self.embed_model,
                        contents=batch,
                        config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=config.EMBED_DIM),
                    )
                    for i, emb in zip(idxs, resp.embeddings):
                        results[i] = list(emb.values)
                        self._emb_cache.set(key_for("emb", self.embed_model, task_type, config.EMBED_DIM, texts[i]), results[i])
                    break
                except Exception as exc:
                    msg = str(exc)
                    if attempt < 3 and any(c in msg for c in ("429", "RESOURCE_EXHAUSTED", "500", "503")):
                        time.sleep(min(2 ** attempt * 2, 20))
                        continue
                    self.last_error = f"Gemini embeddings unavailable ({msg[:120]}) - using offline TF-IDF similarity."
                    self._notify(self.last_error)
                    raise LLMError(self.last_error) from exc
        return results  # type: ignore[return-value]

    def ping(self) -> tuple[bool, str]:
        """Quick connectivity check for the UI."""
        if not self.available:
            return False, "No API key configured - running in offline mode."
        try:
            out = self.generate("Reply with the single word: OK", temperature=0, use_cache=False)
            return True, f"Connected to {self.model} ({out[:20]})"
        except LLMError as exc:
            return False, str(exc)
