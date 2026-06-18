import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RuntimePlanStep:
    action: str
    target: str
    details: str = ""
    status: str = "pending"
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "action": self.action,
            "target": self.target,
            "details": self.details,
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclass
class RuntimeExecutionPlan:
    intent: str
    source: str
    steps: list
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "planned"
    created_at: str = field(default_factory=_now)
    finished_at: str = ""
    result_summary: str = ""
    error: str = ""

    def start(self):
        self.status = "running"
        for step in self.steps:
            if step.status == "pending":
                step.status = "running"

    def complete(self, result_summary=""):
        self.status = "completed"
        self.finished_at = _now()
        self.result_summary = result_summary
        for step in self.steps:
            if step.status in {"pending", "running"}:
                step.status = "completed"

    def fail(self, error):
        self.status = "failed"
        self.finished_at = _now()
        self.error = str(error)
        failed_one = False
        for step in self.steps:
            if step.status == "running" and not failed_one:
                step.status = "failed"
                failed_one = True
            elif step.status == "running":
                step.status = "skipped"
            elif step.status == "pending":
                step.status = "skipped"

    def to_dict(self):
        return {
            "id": self.id,
            "intent": self.intent,
            "source": self.source,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "result_summary": self.result_summary,
            "error": self.error,
            "steps": [step.to_dict() for step in self.steps],
        }

    def summary_line(self):
        return f"{self.id}: {self.intent} [{self.status}] via {self.source}"


def make_plan(intent, source, steps):
    return RuntimeExecutionPlan(
        intent=intent,
        source=source,
        steps=[
            step if isinstance(step, RuntimePlanStep) else RuntimePlanStep(**step)
            for step in steps
        ],
    )


def summarize_result(text, max_chars=240):
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."
