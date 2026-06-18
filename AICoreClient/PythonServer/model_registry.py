from dataclasses import dataclass, field

from intent_router import IntentRouter


@dataclass
class ModelDescriptor:
    name: str
    provider: str
    kind: str
    purpose: str
    model_name: str
    available: bool = False
    status: str = "configured"
    priority: int = 100
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "name": self.name,
            "provider": self.provider,
            "kind": self.kind,
            "purpose": self.purpose,
            "model_name": self.model_name,
            "available": self.available,
            "status": self.status,
            "priority": self.priority,
            "metadata": self.metadata,
        }


class ModelRegistry:
    def __init__(self):
        self._models = {}
        self.intent_router = IntentRouter()

    def register(self, descriptor):
        self._models[descriptor.name] = descriptor

    def get(self, name):
        return self._models.get(name)

    def list_models(self):
        return sorted(self._models.values(), key=lambda item: item.priority)

    def set_availability(self, name, available, status=None, metadata=None):
        descriptor = self._models.get(name)
        if not descriptor:
            return
        descriptor.available = bool(available)
        if status:
            descriptor.status = status
        if metadata:
            descriptor.metadata.update(metadata)

    def snapshot(self):
        return {
            "model_count": len(self._models),
            "available_model_count": sum(1 for model in self._models.values() if model.available),
            "models": [model.to_dict() for model in self.list_models()],
        }

    def describe(self):
        lines = ["Configured model routes:"]
        for model in self.list_models():
            state = "available" if model.available else model.status
            lines.append(
                f"- {model.name} ({model.provider}, {model.kind}, {state}): "
                f"{model.purpose} [{model.model_name}]"
            )
        return "\n".join(lines)

    def route_query(self, text):
        return self.route_decision(text).target

    def route_decision(self, text):
        return self.intent_router.route(text)


def create_default_model_registry(settings, embedding_provider=None):
    registry = ModelRegistry()
    registry.register(
        ModelDescriptor(
            name="runtime_orchestrator",
            provider="hivemind",
            kind="router",
            purpose="Routes runtime/tool/memory requests before LLM fallback.",
            model_name="rules+registry",
            available=True,
            status="available",
            priority=10,
        )
    )
    registry.register(
        ModelDescriptor(
            name="math_service",
            provider="hivemind",
            kind="deterministic_math",
            purpose="Tool-first arithmetic, algebra, calculus, and equation solving before any math LLM.",
            model_name="sympy+safe_arithmetic",
            available=True,
            status="available",
            priority=20,
        )
    )
    registry.register(
        ModelDescriptor(
            name="wizardmath",
            provider="local_transformers",
            kind="legacy_chat",
            purpose="Legacy math model, disabled by default because deterministic math is preferred.",
            model_name=getattr(settings, "wizardmath_model", "WizardLMTeam/WizardMath-7B-V1.1"),
            available=False,
            status="disabled_by_default",
            priority=60,
        )
    )
    registry.register(
        ModelDescriptor(
            name="phi3",
            provider="local_transformers",
            kind="chat",
            purpose="Short general reasoning and local conversation fallback.",
            model_name=getattr(settings, "phi3_model", "microsoft/Phi-3-mini-4k-instruct"),
            available=False,
            status="configured",
            priority=30,
        )
    )
    registry.register(
        ModelDescriptor(
            name="mistral",
            provider="mistral",
            kind="chat",
            purpose="Cloud fallback for long-form or broad natural-language requests.",
            model_name=getattr(settings, "mistral_model", "mistral-large-latest"),
            available=bool(getattr(settings, "mistral_api_key", "")),
            status="available" if getattr(settings, "mistral_api_key", "") else "missing_api_key",
            priority=40,
        )
    )
    embedding_health = embedding_provider.health() if embedding_provider else {}
    registry.register(
        ModelDescriptor(
            name="embedding",
            provider=embedding_health.get("name", getattr(settings, "embedding_provider", "hash")),
            kind="embedding",
            purpose="Indexes and retrieves project memory/RAG chunks.",
            model_name=getattr(settings, "embedding_model", "") or embedding_health.get("name", "hash"),
            available=bool(embedding_health.get("available", True)),
            status="available" if embedding_health.get("available", True) else "unavailable",
            priority=50,
            metadata=embedding_health,
        )
    )
    return registry
