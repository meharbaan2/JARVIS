from dataclasses import dataclass, field


@dataclass(frozen=True)
class RouteDecision:
    intent: str
    target: str
    confidence: float
    reason: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "intent": self.intent,
            "target": self.target,
            "confidence": self.confidence,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class IntentRouter:
    """Rule-based first-pass router for the local runtime."""

    def route(self, text):
        lowered = (text or "").strip().lower()
        if not lowered:
            return RouteDecision("conversation", "phi3", 0.35, "empty or whitespace input")

        if _is_tool_query(lowered):
            return RouteDecision("tool", "runtime_orchestrator", 0.84, "explicit runtime tool or adapter request")

        if _is_math_query(lowered):
            return RouteDecision("math", "math_service", 0.86, "explicit math or symbolic operation")

        if _is_long_generation_query(lowered):
            return RouteDecision("longform", "mistral", 0.72, "long-form generation is better suited to cloud fallback when configured")

        if _is_code_or_project_query(lowered):
            return RouteDecision("engineering", "phi3", 0.70, "code or project reasoning request")

        if _is_general_question(lowered):
            return RouteDecision("conversation", "phi3", 0.70, "general local reasoning question")

        return RouteDecision("conversation", "phi3", 0.50, "default local assistant route")


def _is_math_query(lowered_text):
    math_phrases = [
        "solve",
        "calculate",
        "evaluate",
        "simplify",
        "factor",
        "expand",
        "integral",
        "integrate",
        "derivative",
        "differentiate",
        "equation",
        "limit",
    ]
    symbolic_markers = [" x=", " y=", "dx", "dy", "^", "**", "="]
    if any(phrase in lowered_text for phrase in math_phrases):
        return True
    if any(marker in lowered_text for marker in symbolic_markers) and any(char.isdigit() for char in lowered_text):
        return True
    return False


def _is_tool_query(lowered_text):
    action_markers = [
        "run",
        "simulate",
        "convert",
        "export",
        "reconstruct",
        "validate",
        "verify",
        "inspect",
        "check",
        "assess",
        "confidence",
        "probability",
        "reliable",
        "list",
        "health",
    ]
    if any(
        marker in lowered_text
        for marker in ["ofdmsim", "cadconverter", "folderguardian", "bayesian", "tool", "adapter", "capability", "runtime"]
    ) and any(action in lowered_text for action in action_markers):
        return True

    if "ofdm" in lowered_text and any(
        marker in lowered_text
        for marker in ["run", "simulate", "simulation", "ber", "snr", " db", "emi", "channel", "modulation", "equalizer"]
    ):
        return True

    cad_markers = ["cad converter", ".step", ".stp", ".glb", ".gltf", "step file", "stp file", "glb file", "gltf file"]
    if any(marker in lowered_text for marker in cad_markers) and any(action in lowered_text for action in action_markers):
        return True

    if "folder guardian" in lowered_text and any(action in lowered_text for action in action_markers):
        return True

    if any(marker in lowered_text for marker in ["output reliable", "result reliable", "tool output", "tool result"]):
        return True

    return False


def _is_code_or_project_query(lowered_text):
    markers = [
        "code",
        "function",
        "class",
        "bug",
        "stack trace",
        "repo",
        "project",
        "architecture",
        "grpc",
        "adapter",
        "runtime",
    ]
    return any(marker in lowered_text for marker in markers)


def _is_long_generation_query(lowered_text):
    markers = [
        "write a",
        "create a",
        "generate",
        "draft",
        "long",
        "detailed",
        "comprehensive",
        "essay",
        "article",
        "story",
    ]
    return any(marker in lowered_text for marker in markers)


def _is_general_question(lowered_text):
    markers = [
        "what is",
        "how",
        "explain",
        "tell me about",
        "why is",
        "who is",
        "when did",
        "where is",
        "can you",
        "could you",
        "would you",
        "should i",
        "what are",
        "how does",
        "help me",
        "whats",
    ]
    return any(marker in lowered_text for marker in markers)
