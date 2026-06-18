import json
from dataclasses import dataclass


REQUIRED_PARAMS = {
    ("cadconverter", "validate_glb"): {"input_path"},
    ("cadconverter", "convert_step_to_glb"): {"input_path"},
    ("cadconverter", "convert_glb_to_step"): {"input_path"},
    ("folderguardian", "encrypt_folder"): {"folder_path"},
    ("folderguardian", "decrypt_folder"): {"folder_path"},
    ("security_suite", "assess_path"): {"target_path"},
    ("security_suite", "audit_project"): {"target_path"},
    ("security_suite", "defender_scan"): {"target_path"},
    ("security_suite", "quarantine_plan"): {"target_path"},
    ("security_suite", "quarantine_path"): {"target_path"},
    ("security_suite", "restore_quarantine"): set(),
    ("security_suite", "rule_scan"): {"target_path"},
    ("bayesian_engine", "add_tool_run_evidence"): {"artifact_path"},
    ("bayesian_engine", "predict_scenario"): {"model_path"},
}

BASIC_TYPES = {"path", "json", "string", "integer", "number", "boolean", "list"}


@dataclass
class RuntimeServiceFacade:
    """Service-boundary facade shared by compatibility routes and future gRPC servicers."""

    memory: object
    tool_registry: object
    model_registry: object = None

    def list_capabilities(self):
        capabilities = []
        for tool in self.tool_registry.list_tools():
            for capability in tool.capabilities:
                capabilities.append(
                    {
                        "tool_name": tool.name,
                        "capability_name": capability.name,
                        "description": capability.description,
                        "input_schema": capability.input_schema,
                        "safety_level": capability.safety_level,
                        "supports_progress": capability.supports_progress,
                        "supports_cancel": capability.supports_cancel,
                        "examples": capability.examples,
                        "available": tool.root.exists(),
                        "runtime": tool.runtime,
                        "root": str(tool.root),
                        "tool_description": tool.description,
                    }
                )
        return capabilities

    def get_capability_schema(self, tool_name, capability_name):
        tool = self._get_tool(tool_name)
        capability = self._get_capability(tool, capability_name)
        return {
            "tool_name": tool.name,
            "capability_name": capability.name,
            "description": capability.description,
            "input_schema": capability.input_schema,
            "safety_level": capability.safety_level,
            "supports_progress": capability.supports_progress,
            "supports_cancel": capability.supports_cancel,
            "examples": capability.examples,
        }

    def run_capability(self, tool_name, capability_name, params=None):
        params = params or {}
        tool = self._get_tool(tool_name)
        capability = self._get_capability(tool, capability_name)
        validation = validate_capability_params(tool_name, capability_name, capability.input_schema, params)
        if validation["errors"]:
            message = (
                f"Invalid {tool_name}.{capability_name} parameters:\n- "
                + "\n- ".join(validation["errors"])
            )
            return {
                "status": "failed",
                "tool_name": tool_name,
                "capability_name": capability_name,
                "output_text": message,
                "output_json": json.dumps({"text": message, "validation": validation}),
                "error_message": message,
                "validation": validation,
            }

        output = self.tool_registry.run_capability(
            tool_name,
            capability_name,
            params,
            memory=self.memory,
        )
        return {
            "status": _status_from_output(output),
            "tool_name": tool_name,
            "capability_name": capability_name,
            "output_text": output,
            "output_json": json.dumps({"text": output, "validation": validation}),
            "error_message": "" if not _looks_like_failure(output) else output,
            "validation": validation,
        }

    def search_memory(self, query, project=None, limit=5, source_kind=None, max_per_source=2):
        return self.memory.search(
            query,
            project=project or None,
            limit=limit or 5,
            source_kind=source_kind or None,
            max_per_source=max_per_source or 2,
        )

    def index_project_memory(self, project_names=None, max_files_per_project=24, include_source_summary=True):
        requested = {name.lower() for name in project_names or []}
        results = []
        total = 0
        for tool in self.tool_registry.list_tools():
            if requested and tool.name.lower() not in requested:
                continue
            document_count = self.memory.index_project_documents(
                tool.name,
                tool.root,
                max_files=max_files_per_project or 24,
            )
            source_summary_count = 0
            if include_source_summary and hasattr(self.memory, "index_project_source_summary"):
                source_summary_count = self.memory.index_project_source_summary(tool.name, tool.root)
            overview_count = 0
            if hasattr(self.memory, "index_project_overview"):
                overview_count = self.memory.index_project_overview(tool.name, tool.root)
            artifact_count = 0
            artifact_base = getattr(self.tool_registry, "artifact_dir", None)
            if artifact_base and hasattr(self.memory, "index_tool_artifacts"):
                artifact_count = self.memory.index_tool_artifacts(
                    tool.name,
                    f"{artifact_base}/{tool.name}",
                    max_files=max_files_per_project or 24,
                )
            count = document_count + source_summary_count + overview_count + artifact_count
            status = "indexed" if count else ("missing" if not tool.root.exists() else "no_documents")
            total += count
            results.append(
                {
                    "project": tool.name,
                    "root": str(tool.root),
                    "indexed_document_count": count,
                    "indexed_regular_document_count": document_count,
                    "indexed_source_summary_count": source_summary_count,
                    "indexed_overview_count": overview_count,
                    "indexed_artifact_count": artifact_count,
                    "status": status,
                }
            )
        return {
            "results": results,
            "total_indexed_document_count": total,
        }

    def recent_tool_runs(self, limit=8):
        return self.memory.recent_tool_runs(limit=limit or 8)

    def recent_execution_plans(self, limit=8):
        return self.memory.recent_execution_plans(limit=limit or 8)

    def memory_stats(self):
        return self.memory.stats()

    def runtime_snapshot(self):
        tools = []
        health_by_name = {item["name"]: item for item in self.tool_registry.health()}
        for tool in self.tool_registry.list_tools():
            health = health_by_name.get(tool.name, {})
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "root": str(tool.root),
                    "runtime": tool.runtime,
                    "available": bool(health.get("available")),
                    "capabilities": [
                        {
                            "name": cap.name,
                            "description": cap.description,
                            "input_schema": cap.input_schema,
                            "safety_level": cap.safety_level,
                            "supports_progress": cap.supports_progress,
                            "supports_cancel": cap.supports_cancel,
                            "examples": cap.examples,
                        }
                        for cap in tool.capabilities
                    ],
                }
            )
        snapshot = {
            "tool_count": len(tools),
            "available_tool_count": sum(1 for tool in tools if tool["available"]),
            "tools": tools,
            "memory": self.memory_stats(),
            "recent_tool_runs": self.recent_tool_runs(limit=8),
            "recent_execution_plans": self.recent_execution_plans(limit=8),
        }
        if self.model_registry:
            snapshot["models"] = self.model_registry.snapshot()
        return snapshot

    def _get_tool(self, tool_name):
        for tool in self.tool_registry.list_tools():
            if tool.name == tool_name:
                return tool
        raise KeyError(f"Unknown tool: {tool_name}")

    @staticmethod
    def _get_capability(tool, capability_name):
        for capability in tool.capabilities:
            if capability.name == capability_name:
                return capability
        raise KeyError(f"Unknown capability for {tool.name}: {capability_name}")


def _looks_like_failure(output):
    lowered = (output or "").lower()
    return any(
        marker in lowered
        for marker in (
            " failed",
            "invalid ",
            "missing",
            "not configured",
            "unknown capability",
            "not registered",
        )
    )


def _status_from_output(output):
    return "failed" if _looks_like_failure(output) else "completed"


def validate_capability_params(tool_name, capability_name, input_schema, params):
    input_schema = input_schema or {}
    params = params or {}
    errors = []
    warnings = []

    required = REQUIRED_PARAMS.get((tool_name, capability_name), set())
    for name in sorted(required):
        if _missing(params.get(name)):
            errors.append(f"{name} is required")

    allowed = set(input_schema.keys())
    for name in sorted(set(params.keys()) - allowed):
        errors.append(f"{name} is not a supported parameter")

    for name, spec in input_schema.items():
        if name not in params or _missing(params.get(name)):
            continue
        error = _validate_value(name, params[name], str(spec))
        if error:
            errors.append(error)

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def _missing(value):
    return value is None or (isinstance(value, str) and not value.strip())


def _validate_value(name, value, spec):
    spec = (spec or "").strip()
    choices = [part.strip() for part in spec.split("|") if part.strip()]
    lower_choices = {choice.lower() for choice in choices}

    if choices and not lower_choices <= BASIC_TYPES:
        if str(value).strip().lower() not in lower_choices and not isinstance(value, int):
            return f"{name} must be one of: {', '.join(choices)}"
        return None

    if "|" in spec:
        return _validate_union(name, value, choices)

    return _validate_basic(name, value, spec.lower())


def _validate_union(name, value, choices):
    errors = []
    for choice in choices:
        error = _validate_basic(name, value, choice.lower())
        if not error:
            return None
        errors.append(error)
    return f"{name} must match one of: {', '.join(choices)}"


def _validate_basic(name, value, kind):
    if kind in {"path", "string"}:
        if isinstance(value, (str, bytes)) and str(value).strip():
            return None
        return f"{name} must be a non-empty {kind}"
    if kind == "json":
        if isinstance(value, (dict, list, int, float, bool)) or value is None:
            return None
        if isinstance(value, str):
            try:
                json.loads(value)
                return None
            except json.JSONDecodeError:
                return f"{name} must be valid JSON"
        return f"{name} must be JSON-compatible"
    if kind == "number":
        if isinstance(value, bool):
            return f"{name} must be a number"
        try:
            float(value)
            return None
        except (TypeError, ValueError):
            return f"{name} must be a number"
    if kind == "integer":
        if isinstance(value, bool):
            return f"{name} must be an integer"
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return f"{name} must be an integer"
        if parsed.is_integer():
            return None
        return f"{name} must be an integer"
    if kind == "boolean":
        if isinstance(value, bool):
            return None
        if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0", "yes", "no"}:
            return None
        return f"{name} must be a boolean"
    if kind == "list":
        if isinstance(value, list):
            return None
        return f"{name} must be a list"
    return None
