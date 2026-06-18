import json
import re
from datetime import datetime, timezone
from pathlib import Path


def write_tool_run_artifact(
    base_dir,
    tool_name,
    capability,
    params,
    status,
    output_text="",
    error_text="",
    metrics=None,
    command=None,
    evidence_status=None,
):
    base = Path(base_dir)
    tool_dir = base / _slug(tool_name)
    tool_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    artifact = {
        "created_at": created_at,
        "source_tool": tool_name,
        "capability": capability,
        "params": params or {},
        "status": status,
        "stdout": output_text or "",
        "stderr": error_text or "",
        "metrics": metrics or {},
        "command": command or [],
    }
    if evidence_status:
        artifact["evidence_status"] = evidence_status
    path = tool_dir / f"{stamp}_{_slug(capability)}.json"
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return path


def update_tool_run_artifact(path, **updates):
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    data = json.loads(target.read_text(encoding="utf-8"))
    data.update({key: value for key, value in updates.items() if value is not None})
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return target


def _slug(value):
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_").lower()
    return text or "artifact"
