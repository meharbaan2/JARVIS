import time


DEFAULT_PROBE_PROMPTS = [
    {
        "category": "routing",
        "prompt": (
            "Classify this request as one of: tool, math, memory, conversation. "
            "Request: run OFDMSim at 18 dB and update Bayesian confidence. "
            "Answer with only the label."
        ),
        "expected_hint": "tool",
    },
    {
        "category": "math",
        "prompt": "Solve exactly: derivative of x^3 + 2*x with respect to x. Keep the answer short.",
        "expected_hint": "3*x^2 + 2",
    },
    {
        "category": "coding",
        "prompt": "In one sentence, explain why subprocess.run should use shell=False for tool adapters.",
        "expected_hint": "Avoid shell injection and quoting issues.",
    },
]


class RuntimeModelServiceFacade:
    """Direct runtime boundary for model status and lifecycle controls."""

    def __init__(self, model_registry, load_handlers=None, unload_handlers=None, probe_handlers=None):
        self.model_registry = model_registry
        self.load_handlers = load_handlers or {}
        self.unload_handlers = unload_handlers or {}
        self.probe_handlers = probe_handlers or {}

    def list_models(self):
        if not self.model_registry:
            return {
                "model_count": 0,
                "available_model_count": 0,
                "models": [],
            }
        return self.model_registry.snapshot()

    def get_model_status(self, model_name):
        model = self._get_model(model_name)
        if not model:
            return {
                "found": False,
                "message": f"Unknown model: {model_name}",
                "model": None,
            }
        return {
            "found": True,
            "message": f"{model.name} is {model.status}.",
            "model": model.to_dict(),
        }

    def load_model(self, model_name):
        normalized = _normalize_model_name(model_name)
        handler = self.load_handlers.get(normalized)
        if not handler:
            status = self.get_model_status(normalized)
            return {
                "success": False,
                "status": "unsupported",
                "message": f"Loading is not supported for {model_name}.",
                "model": status.get("model"),
            }
        message = handler()
        status = self.get_model_status(normalized)
        model = status.get("model")
        success = bool(model and model.get("available"))
        return {
            "success": success,
            "status": model.get("status", "unknown") if model else "unknown",
            "message": message,
            "model": model,
        }

    def unload_model(self, model_name):
        normalized = _normalize_model_name(model_name)
        handler = self.unload_handlers.get(normalized)
        if not handler:
            status = self.get_model_status(normalized)
            return {
                "success": False,
                "status": "unsupported",
                "message": f"Unloading is not supported for {model_name}.",
                "model": status.get("model"),
            }
        message = handler()
        status = self.get_model_status(normalized)
        model = status.get("model")
        return {
            "success": True,
            "status": model.get("status", "unknown") if model else "unknown",
            "message": message,
            "model": model,
        }

    def probe_model(self, model_name, prompts=None):
        normalized = _normalize_model_name(model_name)
        handler = self.probe_handlers.get(normalized)
        model_status = self.get_model_status(normalized)
        model = model_status.get("model")
        if not handler:
            return {
                "success": False,
                "status": "unsupported",
                "message": f"Probing is not supported for {model_name}.",
                "model": model,
                "results": [],
            }

        probe_prompts = _normalize_probe_prompts(prompts)
        results = []
        for item in probe_prompts:
            started = time.perf_counter()
            try:
                response_text = handler(item["prompt"])
                status = "completed"
                warning = _probe_warning(response_text, item.get("expected_hint", ""))
            except Exception as exc:
                response_text = ""
                status = "failed"
                warning = str(exc)
            latency_ms = (time.perf_counter() - started) * 1000
            results.append(
                {
                    "category": item["category"],
                    "prompt": item["prompt"],
                    "response_text": response_text,
                    "expected_hint": item.get("expected_hint", ""),
                    "status": status,
                    "warning": warning,
                    "latency_ms": round(latency_ms, 2),
                }
            )

        failed = [item for item in results if item["status"] != "completed"]
        warned = [item for item in results if item.get("warning")]
        refreshed_status = self.get_model_status(normalized)
        return {
            "success": not failed,
            "status": "completed_with_warnings" if warned and not failed else ("failed" if failed else "completed"),
            "message": _probe_summary(normalized, results),
            "model": refreshed_status.get("model") or model,
            "results": results,
        }

    def _get_model(self, model_name):
        if not self.model_registry:
            return None
        return self.model_registry.get(_normalize_model_name(model_name))


def _normalize_model_name(model_name):
    text = (model_name or "").strip().lower()
    aliases = {
        "phi": "phi3",
        "phi-3": "phi3",
        "phi_3": "phi3",
    }
    return aliases.get(text, text)


def _normalize_probe_prompts(prompts):
    if not prompts:
        return list(DEFAULT_PROBE_PROMPTS)
    normalized = []
    for item in prompts:
        prompt = (item.get("prompt") or "").strip()
        if not prompt:
            continue
        normalized.append(
            {
                "category": (item.get("category") or "custom").strip() or "custom",
                "prompt": prompt,
                "expected_hint": (item.get("expected_hint") or "").strip(),
            }
        )
    return normalized or list(DEFAULT_PROBE_PROMPTS)


def _probe_warning(response_text, expected_hint):
    if not expected_hint:
        return ""
    response = (response_text or "").lower()
    hint_tokens = [
        token
        for token in expected_hint.lower().replace("*", " ").replace("+", " ").split()
        if len(token) > 1
    ]
    if hint_tokens and not any(token in response for token in hint_tokens):
        return f"Expected hint not apparent: {expected_hint}"
    return ""


def _probe_summary(model_name, results):
    completed = sum(1 for item in results if item["status"] == "completed")
    warnings = sum(1 for item in results if item.get("warning"))
    total = len(results)
    return f"{model_name} probe completed: {completed}/{total} prompts answered, {warnings} warnings."
