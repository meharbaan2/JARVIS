import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from runtime_artifacts import update_tool_run_artifact, write_tool_run_artifact


EXECUTABLE_EXTENSIONS = {".exe", ".dll", ".sys", ".scr", ".com", ".msi"}
SHORTCUT_EXTENSIONS = {".lnk", ".url"}
SCRIPT_EXTENSIONS = {
    ".ps1",
    ".psm1",
    ".bat",
    ".cmd",
    ".vbs",
    ".vbe",
    ".js",
    ".jse",
    ".wsf",
    ".hta",
    ".py",
    ".sh",
}
ARCHIVE_EXTENSIONS = {".zip", ".7z", ".rar", ".gz", ".bz2", ".xz", ".tar"}
MACRO_DOCUMENT_EXTENSIONS = {".docm", ".xlsm", ".pptm"}
DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    *MACRO_DOCUMENT_EXTENSIONS,
    ".txt",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
}
RISKY_FILE_EXTENSIONS = EXECUTABLE_EXTENSIONS | SCRIPT_EXTENSIONS | SHORTCUT_EXTENSIONS
ZIP_COMPATIBLE_EXTENSIONS = ARCHIVE_EXTENSIONS | {
    ".docx",
    ".xlsx",
    ".pptx",
    ".jar",
    ".war",
    ".apk",
    ".vsix",
}
PE_EXTENSIONS = EXECUTABLE_EXTENSIONS | {".ocx", ".cpl"}
TEXT_SCAN_EXTENSIONS = SCRIPT_EXTENSIONS | {
    ".txt",
    ".log",
    ".ini",
    ".cfg",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
}
SEVERITY_SCORES = {"info": 0, "low": 8, "medium": 18, "high": 35, "critical": 55}
MAX_TEXT_SCAN_BYTES = 256 * 1024
MAX_ENTROPY_SAMPLE_BYTES = 512 * 1024
PROJECT_SECRET_PATTERNS = [
    ("high", "aws_access_key", r"\bAKIA[0-9A-Z]{16}\b", "Possible AWS access key found."),
    ("high", "private_key", r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----", "Private key material found."),
    ("medium", "generic_api_key", r"\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,}", "Possible hardcoded secret or token found."),
    ("medium", "connection_string", r"\b(?:mongodb|postgres|mysql|sqlserver)://[^\s\"']+", "Possible database connection string found."),
]
PROJECT_MANIFEST_NAMES = {
    "package.json",
    "requirements.txt",
    "pyproject.toml",
    "poetry.lock",
    "Pipfile",
    "Pipfile.lock",
    "Cargo.toml",
    "go.mod",
    "composer.json",
    "packages.config",
}

SCRIPT_PATTERNS = [
    (
        "critical",
        "defender_tampering",
        r"\b(?:set-mppreference|add-mppreference)\b[^\r\n]{0,180}\b(?:disablerealtimemonitoring|exclusionpath|exclusionprocess)\b",
        "Script appears to change Microsoft Defender settings or exclusions.",
    ),
    (
        "high",
        "encoded_powershell",
        r"\b(?:powershell(?:\.exe)?|pwsh(?:\.exe)?)\b[^\r\n]{0,120}(?:^|\s)(?:-enc|-encodedcommand)\b",
        "Script launches PowerShell with an encoded command.",
    ),
    (
        "high",
        "download_and_execute",
        r"\b(?:invoke-webrequest|iwr|curl|wget)\b[\s\S]{0,260}\b(?:iex|invoke-expression|start-process|cmd\s*/c|powershell)\b",
        "Script combines network download behavior with immediate execution.",
    ),
    (
        "high",
        "certutil_download",
        r"\bcertutil(?:\.exe)?\b[^\r\n]{0,120}\b(?:-urlcache|-decode)\b",
        "Script uses certutil for download or decoding behavior.",
    ),
    (
        "high",
        "shadow_copy_deletion",
        r"\bvssadmin(?:\.exe)?\b[^\r\n]{0,120}\bdelete\s+shadows\b",
        "Script attempts to delete Windows shadow copies.",
    ),
    (
        "medium",
        "autorun_persistence_hint",
        r"\breg(?:\.exe)?\b[^\r\n]{0,120}\badd\b[^\r\n]{0,160}\\run\b",
        "Script writes to a Windows autorun registry location.",
    ),
    (
        "medium",
        "scheduled_task_creation",
        r"\bschtasks(?:\.exe)?\b[^\r\n]{0,160}\b/create\b",
        "Script creates a scheduled task.",
    ),
    (
        "medium",
        "base64_decode_hint",
        r"\bfrombase64string\b|\bbase64\s+-d\b",
        "Script contains base64 decoding behavior.",
    ),
    (
        "medium",
        "long_base64_blob",
        r"[A-Za-z0-9+/]{160,}={0,2}",
        "Script contains a long base64-like blob.",
    ),
]

BINARY_INDICATOR_PATTERNS = [
    ("high", "credential_access_hint", b"lsass.exe", "Binary text references LSASS, which is often targeted for credential theft."),
    ("medium", "remote_thread_hint", b"createremotethread", "Binary text references remote thread creation."),
    ("medium", "process_injection_hint", b"writeprocessmemory", "Binary text references process memory writing."),
    ("medium", "dynamic_code_hint", b"virtualalloc", "Binary text references dynamic executable memory allocation."),
    ("medium", "shell_execution_hint", b"shellexecute", "Binary text references shell execution APIs."),
    ("low", "network_stack_hint", b"winhttp", "Binary text references Windows HTTP APIs."),
]


def run_security_suite_capability(capability_name, params, memory=None, artifact_dir=None, bayesian_adapter=None):
    params = params or {}
    if capability_name == "assess_path":
        return assess_path(params, memory=memory, artifact_dir=artifact_dir, bayesian_adapter=bayesian_adapter)
    if capability_name == "audit_project":
        return audit_project(params, memory=memory, artifact_dir=artifact_dir)
    if capability_name == "defender_scan":
        return defender_scan(params, memory=memory, artifact_dir=artifact_dir)
    if capability_name == "quarantine_plan":
        return quarantine_plan(params, memory=memory, artifact_dir=artifact_dir)
    if capability_name == "quarantine_path":
        return quarantine_path(params, memory=memory, artifact_dir=artifact_dir)
    if capability_name == "restore_quarantine":
        return restore_quarantine(params, memory=memory, artifact_dir=artifact_dir)
    if capability_name == "defender_history":
        return defender_history(params, memory=memory, artifact_dir=artifact_dir)
    if capability_name == "rule_scan":
        return rule_scan(params, memory=memory, artifact_dir=artifact_dir)
    if capability_name == "security_history":
        return security_history(params, artifact_dir=artifact_dir)
    if capability_name == "explain_assessment":
        return explain_assessment(params, artifact_dir=artifact_dir)
    return f"Unknown security_suite capability: {capability_name}"


def assess_path(params, memory=None, artifact_dir=None, bayesian_adapter=None):
    try:
        target_path = _resolve_target_path(params.get("target_path") or params.get("path"))
    except ValueError as exc:
        return f"Invalid security suite assessment: {exc}"

    max_files = _bounded_int(params.get("max_files"), default=500, minimum=1, maximum=5000)
    include_defender = _bool_param(params.get("include_defender_status"), default=True)
    assessment = _build_assessment(target_path, max_files=max_files, include_defender=include_defender)
    artifact_path = _write_assessment_artifact(artifact_dir, params, assessment)
    if artifact_path:
        assessment["artifact_path"] = str(artifact_path)

    evidence_status, bayesian_summary = _maybe_update_bayesian(artifact_path, params, bayesian_adapter)
    output_text = _compact_assessment_json(assessment)
    if memory:
        memory.record_tool_run(
            "security_suite",
            "assess_path",
            json.dumps(params, sort_keys=True),
            "success",
            output_text,
            "",
            artifact_path=artifact_path,
            metrics_json=json.dumps(assessment, sort_keys=True),
            evidence_status=evidence_status,
        )
    return _format_assessment_response(assessment, artifact_path) + bayesian_summary


def audit_project(params, memory=None, artifact_dir=None):
    try:
        target_path = _resolve_target_path(params.get("target_path") or params.get("path"))
    except ValueError as exc:
        return f"Invalid security suite project audit: {exc}"
    if not target_path.is_dir():
        return f"Invalid security suite project audit: target_path must be a project folder: {target_path}"

    max_files = _bounded_int(params.get("max_files"), default=800, minimum=1, maximum=8000)
    audit = _build_project_audit(target_path, max_files=max_files)
    artifact_path = _write_security_artifact(artifact_dir, "audit_project", params, audit, "success")
    output_text = json.dumps(_compact_project_audit(audit), indent=2)
    if memory:
        memory.record_tool_run(
            "security_suite",
            "audit_project",
            json.dumps(params, sort_keys=True),
            "success",
            output_text,
            "",
            artifact_path=artifact_path,
            metrics_json=json.dumps(audit, sort_keys=True),
            evidence_status="not_applicable",
        )
    return _format_project_audit_response(audit, artifact_path)


def defender_scan(params, memory=None, artifact_dir=None):
    try:
        target_path = _resolve_target_path(params.get("target_path") or params.get("path"))
    except ValueError as exc:
        return f"Invalid Defender scan request: {exc}"

    confirm_scan = _bool_param(params.get("confirm_scan"), default=False)
    if not confirm_scan:
        plan = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target_path": str(target_path),
            "will_execute": False,
            "reason": "Defender scan requires confirm_scan=true because local Defender policy may remediate detected threats.",
            "command_preview": "Start-MpScan -ScanType CustomScan -ScanPath <target>",
            "recommended_next_step": "Set confirm_scan=true only when you want Windows Defender to run against this target.",
        }
        artifact_path = _write_security_artifact(artifact_dir, "defender_scan", params, plan, "planned")
        if memory:
            memory.record_tool_run(
                "security_suite",
                "defender_scan",
                json.dumps(params, sort_keys=True),
                "planned",
                json.dumps(plan, indent=2),
                "",
                artifact_path=artifact_path,
                metrics_json=json.dumps(plan, sort_keys=True),
                evidence_status="not_applicable",
            )
        artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
        return (
            "Windows Defender custom scan prepared, not executed.\n"
            f"Target: {target_path}\n"
            "Reason: confirm_scan=true is required because Defender may remediate threats depending on local policy."
            f"{artifact_line}"
        )

    result = _run_defender_custom_scan(target_path, timeout_seconds=_bounded_int(params.get("timeout_seconds"), 600, 15, 7200))
    status = "success" if result.get("returncode") == 0 else "failed"
    artifact_path = _write_security_artifact(artifact_dir, "defender_scan", params, result, status)
    if memory:
        memory.record_tool_run(
            "security_suite",
            "defender_scan",
            json.dumps(params, sort_keys=True),
            status,
            json.dumps(result, indent=2),
            result.get("stderr") or "",
            artifact_path=artifact_path,
            metrics_json=json.dumps(result, sort_keys=True),
            evidence_status="not_applicable",
        )
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    if status == "success":
        return f"Windows Defender custom scan completed.\nTarget: {target_path}{artifact_line}"
    return f"Windows Defender custom scan failed.\nTarget: {target_path}\nError: {result.get('stderr') or result.get('stdout') or result.get('error')}{artifact_line}"


def quarantine_plan(params, memory=None, artifact_dir=None):
    try:
        target_path = _resolve_target_path(params.get("target_path") or params.get("path"))
    except ValueError as exc:
        return f"Invalid quarantine plan request: {exc}"

    plan = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_path": str(target_path),
        "will_execute_live_action": False,
        "reason": "V2 only prepares quarantine planning; live quarantine needs explicit confirmation and restore metadata.",
        "steps": [
            "Create an isolated quarantine directory under runtime_state.",
            "Copy or move the target only after explicit confirmation.",
            "Record original path, SHA-256, file size, timestamp, and restore instructions.",
            "Never delete the original without a separate explicit delete confirmation.",
        ],
    }
    artifact_path = _write_security_artifact(artifact_dir, "quarantine_plan", params, plan, "success")
    if memory:
        memory.record_tool_run(
            "security_suite",
            "quarantine_plan",
            json.dumps(params, sort_keys=True),
            "success",
            json.dumps(plan, indent=2),
            "",
            artifact_path=artifact_path,
            metrics_json=json.dumps(plan, sort_keys=True),
            evidence_status="not_applicable",
        )
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    return (
        "Security quarantine plan prepared. No files were moved or deleted.\n"
        f"Target: {target_path}\n"
        "Live quarantine remains disabled until explicit confirmation and restore metadata are implemented."
        f"{artifact_line}"
    )


def quarantine_path(params, memory=None, artifact_dir=None):
    try:
        target_path = _resolve_target_path(params.get("target_path") or params.get("path"))
    except ValueError as exc:
        return f"Invalid quarantine request: {exc}"

    confirm = _bool_param(params.get("confirm_quarantine"), default=False)
    if not confirm:
        plan = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target_path": str(target_path),
            "will_execute_live_action": False,
            "reason": "Set confirm_quarantine=true to move the target into local Hivemind quarantine storage.",
            "restore_metadata": "A metadata record will preserve original path, quarantine path, hashes, and restore status.",
        }
        artifact_path = _write_security_artifact(artifact_dir, "quarantine_path", params, plan, "planned")
        return (
            "Security quarantine prepared, not executed.\n"
            f"Target: {target_path}\n"
            "Reason: confirm_quarantine=true is required before moving files."
            f"{_artifact_line(artifact_path)}"
        )

    state_dir = _security_state_dir(artifact_dir)
    quarantine_dir = state_dir / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    quarantine_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    destination = quarantine_dir / f"{quarantine_id}_{_safe_quarantine_name(target_path.name)}"
    metadata_path = quarantine_dir / f"{quarantine_id}.metadata.json"
    inventory = _target_inventory(target_path)
    metadata = {
        "quarantine_id": quarantine_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "original_path": str(target_path),
        "quarantine_path": str(destination),
        "target_kind": "folder" if target_path.is_dir() else "file",
        "reason": params.get("reason") or "User-confirmed local defensive quarantine.",
        "inventory": inventory,
        "restore_status": "quarantined",
        "local_only": True,
    }

    try:
        shutil.move(str(target_path), str(destination))
        metadata["moved"] = True
        metadata["metadata_path"] = str(metadata_path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        status = "success"
        error_text = ""
    except Exception as exc:
        metadata["moved"] = False
        metadata["error"] = str(exc)
        status = "failed"
        error_text = str(exc)

    artifact_path = _write_security_artifact(artifact_dir, "quarantine_path", params, metadata, status)
    if memory:
        memory.record_tool_run(
            "security_suite",
            "quarantine_path",
            json.dumps(params, sort_keys=True),
            status,
            json.dumps(metadata, indent=2),
            error_text,
            artifact_path=artifact_path,
            metrics_json=json.dumps(metadata, sort_keys=True),
            evidence_status="not_applicable",
        )
    if status != "success":
        return f"Security quarantine failed.\nTarget: {target_path}\nError: {error_text}{_artifact_line(artifact_path)}"
    return (
        "Security quarantine complete.\n"
        f"Quarantine ID: {quarantine_id}\n"
        f"Original: {target_path}\n"
        f"Quarantined: {destination}\n"
        f"Restore metadata: {metadata_path}"
        f"{_artifact_line(artifact_path)}"
    )


def restore_quarantine(params, memory=None, artifact_dir=None):
    confirm = _bool_param(params.get("confirm_restore"), default=False)
    allow_overwrite = _bool_param(params.get("allow_overwrite"), default=False)
    try:
        metadata_path = _resolve_quarantine_metadata(params, artifact_dir)
    except ValueError as exc:
        return f"Invalid quarantine restore request: {exc}"
    metadata = _read_json_file(metadata_path)
    if not metadata:
        return f"Invalid quarantine restore request: metadata could not be read: {metadata_path}"
    quarantine_path = Path(metadata.get("quarantine_path") or "")
    restore_path = Path(params.get("restore_path") or metadata.get("original_path") or "").expanduser()
    if not restore_path.is_absolute():
        restore_path = (Path.cwd() / restore_path).resolve()

    if not confirm:
        plan = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "quarantine_id": metadata.get("quarantine_id"),
            "metadata_path": str(metadata_path),
            "quarantine_path": str(quarantine_path),
            "restore_path": str(restore_path),
            "will_execute_live_action": False,
            "reason": "Set confirm_restore=true to move the quarantined target back.",
        }
        artifact_path = _write_security_artifact(artifact_dir, "restore_quarantine", params, plan, "planned")
        return (
            "Security quarantine restore prepared, not executed.\n"
            f"Quarantine ID: {metadata.get('quarantine_id', 'unknown')}\n"
            f"Restore target: {restore_path}\n"
            "Reason: confirm_restore=true is required before moving files."
            f"{_artifact_line(artifact_path)}"
        )

    result = dict(metadata)
    result.update(
        {
            "restore_requested_at": datetime.now(timezone.utc).isoformat(),
            "restore_path": str(restore_path),
            "allow_overwrite": allow_overwrite,
        }
    )
    try:
        if not quarantine_path.exists():
            raise FileNotFoundError(f"Quarantined target is missing: {quarantine_path}")
        if restore_path.exists() and not allow_overwrite:
            raise FileExistsError(f"Restore path already exists: {restore_path}")
        restore_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(quarantine_path), str(restore_path))
        result["restore_status"] = "restored"
        result["restored"] = True
        result["restored_at"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        status = "success"
        error_text = ""
    except Exception as exc:
        result["restore_status"] = "failed"
        result["restored"] = False
        result["error"] = str(exc)
        status = "failed"
        error_text = str(exc)

    artifact_path = _write_security_artifact(artifact_dir, "restore_quarantine", params, result, status)
    if memory:
        memory.record_tool_run(
            "security_suite",
            "restore_quarantine",
            json.dumps(params, sort_keys=True),
            status,
            json.dumps(result, indent=2),
            error_text,
            artifact_path=artifact_path,
            metrics_json=json.dumps(result, sort_keys=True),
            evidence_status="not_applicable",
        )
    if status != "success":
        return f"Security quarantine restore failed.\nMetadata: {metadata_path}\nError: {error_text}{_artifact_line(artifact_path)}"
    return (
        "Security quarantine restore complete.\n"
        f"Quarantine ID: {result.get('quarantine_id')}\n"
        f"Restored to: {restore_path}"
        f"{_artifact_line(artifact_path)}"
    )


def defender_history(params, memory=None, artifact_dir=None):
    limit = _bounded_int((params or {}).get("limit"), default=10, minimum=1, maximum=50)
    result = _read_windows_defender_history(limit)
    status = "success" if result.get("status") == "completed" else "skipped"
    artifact_path = _write_security_artifact(artifact_dir, "defender_history", params or {}, result, status)
    if memory:
        memory.record_tool_run(
            "security_suite",
            "defender_history",
            json.dumps(params or {}, sort_keys=True),
            status,
            json.dumps(result, indent=2),
            result.get("reason", ""),
            artifact_path=artifact_path,
            metrics_json=json.dumps(result, sort_keys=True),
            evidence_status="not_applicable",
        )
    if status != "success":
        return f"Windows Defender threat history unavailable: {result.get('reason', 'unknown')}{_artifact_line(artifact_path)}"
    detections = result.get("detections") or []
    lines = [f"Windows Defender threat history read complete. Detections: {len(detections)}."]
    for item in detections[:6]:
        lines.append(
            f"- {item.get('ThreatName') or item.get('ThreatID') or 'threat'}: "
            f"{item.get('ActionSuccess')}; {item.get('Resources') or item.get('InitialDetectionTime') or ''}"
        )
    lines.append(_artifact_line(artifact_path).strip())
    return "\n".join(line for line in lines if line)


def rule_scan(params, memory=None, artifact_dir=None):
    try:
        target_path = _resolve_target_path(params.get("target_path") or params.get("path"))
    except ValueError as exc:
        return f"Invalid rule scan request: {exc}"
    max_files = _bounded_int(params.get("max_files"), default=500, minimum=1, maximum=5000)
    rules = _load_rule_pack(params.get("rules_path"))
    files, collection = _collect_files(target_path, max_files=max_files)
    matches = []
    for path in files:
        sample = _read_binary_sample(path, MAX_TEXT_SCAN_BYTES)
        text = sample.decode("utf-8", errors="ignore")
        relative = _relative_label(path, target_path)
        for rule in rules:
            extensions = {str(item).lower() for item in rule.get("extensions", [])}
            if extensions and path.suffix.lower() not in extensions:
                continue
            haystack = text if rule.get("mode", "text") == "text" else sample.decode("latin1", errors="ignore")
            if re.search(rule["pattern"], haystack, flags=re.IGNORECASE | re.MULTILINE):
                matches.append(
                    {
                        "rule_id": rule["id"],
                        "severity": rule.get("severity", "medium"),
                        "path": relative,
                        "message": rule.get("message", rule["id"]),
                    }
                )
    score = _risk_score([_finding(item["severity"], "rule_match", item["path"], item["message"], item["rule_id"]) for item in matches])
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_path": str(target_path),
        "rules_loaded": len(rules),
        "scanned_file_count": len(files),
        "partial": bool(collection.get("max_files_reached")),
        "matches": matches,
        "risk_score": score,
        "risk_level": _risk_level(score),
        "local_only": True,
    }
    artifact_path = _write_security_artifact(artifact_dir, "rule_scan", params, result, "success")
    if memory:
        memory.record_tool_run(
            "security_suite",
            "rule_scan",
            json.dumps(params, sort_keys=True),
            "success",
            json.dumps(result, indent=2),
            "",
            artifact_path=artifact_path,
            metrics_json=json.dumps(result, sort_keys=True),
            evidence_status="not_applicable",
        )
    lines = [
        "security_suite local rule scan complete.",
        f"Target: {target_path}",
        f"Rules: {len(rules)}; files scanned: {len(files)}; matches: {len(matches)}.",
        f"Verdict: {result['risk_level']} risk, score {score}/100.",
    ]
    if matches:
        lines.append("Matches:")
        for item in matches[:8]:
            lines.append(f"- {item['severity']}: {item['message']} ({item['path']})")
    if result["partial"]:
        lines.append(f"Note: rule scan was partial after {max_files} files.")
    lines.append(_artifact_line(artifact_path).strip())
    return "\n".join(line for line in lines if line)


def security_history(params, artifact_dir=None):
    limit = _bounded_int((params or {}).get("limit"), default=5, minimum=1, maximum=25)
    artifacts = _recent_security_artifacts(artifact_dir, limit=limit)
    if artifacts is None:
        return "Security history is unavailable because runtime artifact storage is not configured."
    if not artifacts:
        return "No security suite assessments have been recorded yet."

    lines = ["Recent local security assessments:"]
    for artifact in artifacts:
        assessment = _assessment_from_artifact(artifact)
        lines.append(
            f"- {assessment.get('risk_level', 'unknown')} risk, score {assessment.get('risk_score', 'unknown')}: "
            f"{assessment.get('target_path', 'unknown target')} ({artifact})"
        )
    return "\n".join(lines)


def explain_assessment(params, artifact_dir=None):
    params = params or {}
    artifact_path = params.get("artifact_path") or params.get("path")
    if artifact_path:
        artifact = Path(str(artifact_path).strip().strip('"').strip("'")).expanduser()
        if not artifact.is_absolute():
            artifact = Path.cwd() / artifact
        artifact = artifact.resolve()
    else:
        artifacts = _recent_security_artifacts(artifact_dir, limit=1)
        if artifacts is None:
            return "Security assessment explanation is unavailable because runtime artifact storage is not configured."
        if not artifacts:
            return "No security suite assessment is available to explain yet."
        artifact = artifacts[0]

    if not artifact.exists():
        return f"Security assessment artifact was not found: {artifact}"
    assessment = _assessment_from_artifact(artifact)
    findings = assessment.get("findings") or []
    recommendations = assessment.get("recommended_actions") or []
    lines = [
        "Local security assessment explanation:",
        f"Target: {assessment.get('target_path', 'unknown')}",
        f"Verdict: {assessment.get('risk_level', 'unknown')} risk, score {assessment.get('risk_score', 'unknown')}.",
        assessment.get("verdict", "No verdict text was recorded."),
    ]
    if findings:
        lines.append("Top evidence:")
        for finding in findings[:6]:
            lines.append(f"- {finding.get('severity', 'info')}: {finding.get('message', 'finding')}")
    else:
        lines.append("No suspicious local indicators were recorded in the scanned files.")
    if recommendations:
        lines.append("Recommended actions:")
        lines.extend(f"- {item}" for item in recommendations[:5])
    lines.append(f"Artifact: {artifact}")
    return "\n".join(lines)


def _build_assessment(target_path, max_files, include_defender):
    files, collection = _collect_files(target_path, max_files=max_files)
    findings = []
    file_hashes = []
    detectors = []
    skipped_checks = []
    summary = {
        "target_is_file": target_path.is_file(),
        "scanned_file_count": len(files),
        "max_files": max_files,
        "executable_count": 0,
        "script_count": 0,
        "recent_24h_count": 0,
        "recent_7d_count": 0,
        "total_size_bytes": 0,
    }

    now_ts = datetime.now(timezone.utc).timestamp()
    for path in files:
        file_result = _analyze_file(path, target_path)
        file_hashes.append(file_result["hash_record"])
        findings.extend(file_result["findings"])
        stat = file_result["stat"]
        suffix = path.suffix.lower()
        summary["total_size_bytes"] += stat.get("size_bytes", 0)
        if suffix in EXECUTABLE_EXTENSIONS:
            summary["executable_count"] += 1
        if suffix in SCRIPT_EXTENSIONS:
            summary["script_count"] += 1
        age_seconds = max(0, now_ts - stat.get("modified_ts", now_ts))
        if age_seconds <= 24 * 60 * 60:
            summary["recent_24h_count"] += 1
        if age_seconds <= 7 * 24 * 60 * 60:
            summary["recent_7d_count"] += 1

    detectors.extend(
        [
            {"name": "hash_inventory", "status": "completed", "file_count": len(file_hashes)},
            {"name": "extension_checks", "status": "completed", "finding_count": _count_findings(findings, {"double_extension", "extension_mismatch"})},
            {"name": "script_pattern_checks", "status": "completed", "finding_count": _count_findings(findings, {"script_pattern"})},
            {"name": "entropy_checks", "status": "completed", "finding_count": _count_findings(findings, {"entropy"})},
            {"name": "archive_inventory", "status": "completed", "finding_count": _count_findings(findings, {"archive_contents"})},
            {"name": "download_zone_checks", "status": "completed", "finding_count": _count_findings(findings, {"download_zone"})},
            {"name": "binary_indicator_checks", "status": "completed", "finding_count": _count_findings(findings, {"binary_indicator", "network_indicator"})},
        ]
    )
    if collection.get("max_files_reached"):
        skipped_checks.append(
            {
                "name": "file_collection",
                "reason": f"Stopped after max_files={max_files}; scan is partial.",
            }
        )
        findings.append(
            _finding(
                "low",
                "partial_scan",
                str(target_path),
                f"Scan stopped after {max_files} files; remaining files were not assessed.",
                "Increase max_files for a broader scan.",
            )
        )

    if summary["executable_count"] >= 25:
        findings.append(
            _finding(
                "low",
                "executable_inventory",
                str(target_path),
                f"Folder contains {summary['executable_count']} executable files.",
                "Large executable inventories deserve manual review if the source is unknown.",
            )
        )
    if summary["script_count"] >= 25:
        findings.append(
            _finding(
                "low",
                "script_inventory",
                str(target_path),
                f"Folder contains {summary['script_count']} script files.",
                "Large script inventories deserve manual review if the source is unknown.",
            )
        )

    log_result = _folderguardian_log_correlation(target_path)
    detectors.append(log_result["detector"])
    skipped_checks.extend(log_result["skipped_checks"])
    findings.extend(log_result["findings"])

    if include_defender:
        defender_result = _read_windows_defender_status()
        detectors.append(defender_result["detector"])
        skipped_checks.extend(defender_result["skipped_checks"])
        findings.extend(defender_result["findings"])
    else:
        detectors.append({"name": "windows_defender_status", "status": "skipped", "reason": "Disabled by request."})
        skipped_checks.append({"name": "windows_defender_status", "reason": "Disabled by request."})

    risk_score = _risk_score(findings)
    risk_level = _risk_level(risk_score)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_path": str(target_path),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "verdict": _verdict(risk_level),
        "findings": findings,
        "detectors": detectors,
        "file_hashes": file_hashes,
        "recommended_actions": _recommendations(risk_level, findings),
        "skipped_checks": skipped_checks,
        "summary": summary,
        "local_only": True,
        "assessment_kind": "local risk assessment",
        "guarantee": "This is not proof that a file is safe or malicious; it is a local indicator review.",
    }


def _analyze_file(path, target_path):
    stat = _safe_stat(path)
    sample = _read_binary_sample(path, MAX_ENTROPY_SAMPLE_BYTES)
    relative = _relative_label(path, target_path)
    suffix = path.suffix.lower()
    findings = []
    hash_record = {
        "path": relative,
        "sha256": _sha256(path),
        "size_bytes": stat.get("size_bytes", 0),
        "extension": suffix,
        "modified_at": stat.get("modified_at"),
        "is_executable": suffix in EXECUTABLE_EXTENSIONS,
        "is_script": suffix in SCRIPT_EXTENSIONS,
        "is_shortcut": suffix in SHORTCUT_EXTENSIONS,
        "is_macro_document": suffix in MACRO_DOCUMENT_EXTENSIONS,
    }
    findings.extend(_file_name_risk_findings(path, relative))
    findings.extend(_double_extension_findings(path, relative))
    findings.extend(_extension_mismatch_findings(path, relative, sample))
    findings.extend(_script_pattern_findings(path, relative, sample))
    findings.extend(_entropy_findings(path, relative, sample, stat.get("size_bytes", 0)))
    findings.extend(_archive_content_findings(path, relative))
    findings.extend(_zone_identifier_findings(path, relative))
    findings.extend(_binary_indicator_findings(path, relative, sample))
    return {"hash_record": hash_record, "findings": findings, "stat": stat}


def _file_name_risk_findings(path, relative):
    suffix = path.suffix.lower()
    findings = []
    if suffix in MACRO_DOCUMENT_EXTENSIONS:
        findings.append(
            _finding(
                "medium",
                "macro_document",
                relative,
                f"Macro-enabled Office document found: {path.name}",
                "Macro-capable documents can execute embedded code when opened.",
            )
        )
    if suffix in SHORTCUT_EXTENSIONS:
        findings.append(
            _finding(
                "medium",
                "shortcut_file",
                relative,
                f"Shortcut-style file found: {path.name}",
                "Shortcuts can hide command targets and should be checked before opening.",
            )
        )
    return findings


def _double_extension_findings(path, relative):
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if len(suffixes) >= 2 and suffixes[-2] in DOCUMENT_EXTENSIONS and suffixes[-1] in EXECUTABLE_EXTENSIONS | SCRIPT_EXTENSIONS:
        return [
            _finding(
                "high",
                "double_extension",
                relative,
                f"Suspicious double extension: {path.name}",
                "Document-looking name ends in an executable or script extension.",
            )
        ]
    return []


def _extension_mismatch_findings(path, relative, sample):
    suffix = path.suffix.lower()
    findings = []
    if sample.startswith(b"MZ") and suffix not in PE_EXTENSIONS:
        findings.append(
            _finding(
                "high",
                "extension_mismatch",
                relative,
                "File has a Windows executable header but not an executable extension.",
                "MZ/PE header mismatch.",
            )
        )
    elif suffix in PE_EXTENSIONS and sample and not sample.startswith(b"MZ"):
        findings.append(
            _finding(
                "medium",
                "extension_mismatch",
                relative,
                "Executable extension does not match the expected Windows executable header.",
                "Executable extension without MZ header.",
            )
        )
    if sample.startswith(b"%PDF") and suffix != ".pdf":
        findings.append(
            _finding(
                "low",
                "extension_mismatch",
                relative,
                "File has a PDF header but a different extension.",
                "PDF header mismatch.",
            )
        )
    if sample.startswith(b"PK\x03\x04") and suffix not in ZIP_COMPATIBLE_EXTENSIONS:
        findings.append(
            _finding(
                "low",
                "extension_mismatch",
                relative,
                "File has a ZIP-like header but an unexpected extension.",
                "ZIP header mismatch.",
            )
        )
    return findings


def _script_pattern_findings(path, relative, sample):
    suffix = path.suffix.lower()
    if suffix not in TEXT_SCAN_EXTENSIONS and not _looks_text(sample):
        return []
    text = _read_text_prefix(path, MAX_TEXT_SCAN_BYTES).lower()
    findings = []
    for severity, pattern_id, pattern, message in SCRIPT_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            findings.append(_finding(severity, "script_pattern", relative, message, pattern_id))
    return findings


def _entropy_findings(path, relative, sample, size_bytes):
    suffix = path.suffix.lower()
    if size_bytes < 4096 or not sample or suffix in ARCHIVE_EXTENSIONS:
        return []
    entropy = _entropy(sample)
    if entropy < 7.35:
        return []
    severity = "medium" if suffix in EXECUTABLE_EXTENSIONS or suffix not in TEXT_SCAN_EXTENSIONS else "low"
    return [
        _finding(
            severity,
            "entropy",
            relative,
            f"High entropy content detected ({entropy:.2f}); this can indicate packed or encrypted content.",
            "High entropy is not malicious by itself, but it is worth reviewing for unknown executables.",
        )
    ]


def _archive_content_findings(path, relative):
    suffix = path.suffix.lower()
    if suffix not in ZIP_COMPATIBLE_EXTENSIONS or not zipfile.is_zipfile(path):
        return []
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return []

    risky_names = []
    double_extension_names = []
    for name in names[:2000]:
        archive_path = Path(name)
        suffixes = [item.lower() for item in archive_path.suffixes]
        if suffixes and suffixes[-1] in RISKY_FILE_EXTENSIONS:
            risky_names.append(name)
        if len(suffixes) >= 2 and suffixes[-2] in DOCUMENT_EXTENSIONS and suffixes[-1] in RISKY_FILE_EXTENSIONS:
            double_extension_names.append(name)

    findings = []
    if risky_names:
        examples = ", ".join(risky_names[:5])
        severity = "high" if any(Path(name).suffix.lower() in EXECUTABLE_EXTENSIONS for name in risky_names) else "medium"
        findings.append(
            _finding(
                severity,
                "archive_contents",
                relative,
                f"Archive contains {len(risky_names)} executable, script, or shortcut-style entries.",
                examples,
            )
        )
    if double_extension_names:
        findings.append(
            _finding(
                "high",
                "archive_contents",
                relative,
                "Archive contains document-looking entries ending in executable or script extensions.",
                ", ".join(double_extension_names[:5]),
            )
        )
    if len(names) > 2000:
        findings.append(
            _finding(
                "low",
                "archive_contents",
                relative,
                "Archive inventory was truncated after 2000 entries.",
                "Large archive; not extracted.",
            )
        )
    return findings


def _zone_identifier_findings(path, relative):
    if os.name != "nt" or not Path(path).is_file():
        return []
    zone_path = f"{path}:Zone.Identifier"
    try:
        zone_text = Path(zone_path).read_text(encoding="utf-8", errors="replace")[:4096]
    except OSError:
        return []
    zone_match = re.search(r"ZoneId\s*=\s*(\d+)", zone_text, flags=re.IGNORECASE)
    if not zone_match:
        return []
    zone_id = int(zone_match.group(1))
    if zone_id < 3:
        return []
    severity = "medium" if Path(path).suffix.lower() in RISKY_FILE_EXTENSIONS else "low"
    return [
        _finding(
            severity,
            "download_zone",
            relative,
            "Windows Mark-of-the-Web indicates this file came from the internet or a restricted zone.",
            f"ZoneId={zone_id}",
        )
    ]


def _binary_indicator_findings(path, relative, sample):
    if not sample:
        return []
    lowered = sample.lower()
    findings = []
    for severity, pattern_id, needle, message in BINARY_INDICATOR_PATTERNS:
        if needle in lowered:
            findings.append(_finding(severity, "binary_indicator", relative, message, pattern_id))
    decoded = sample.decode("utf-8", errors="ignore")
    urls = re.findall(r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]{8,}", decoded)
    if urls:
        findings.append(
            _finding(
                "low",
                "network_indicator",
                relative,
                f"Embedded URL-like strings found ({len(urls)}).",
                ", ".join(urls[:3]),
            )
        )
    return findings


def _folderguardian_log_correlation(target_path):
    folder = target_path if target_path.is_dir() else target_path.parent
    log_path = folder / "SecurityLog.txt"
    detector = {"name": "folderguardian_log_correlation", "status": "completed", "log_path": str(log_path), "event_count": 0}
    skipped_checks = []
    findings = []
    if not log_path.exists():
        detector["status"] = "skipped"
        detector["reason"] = "No SecurityLog.txt found next to the target."
        skipped_checks.append({"name": "folderguardian_log_correlation", "reason": detector["reason"]})
        return {"detector": detector, "skipped_checks": skipped_checks, "findings": findings}
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
    except OSError as exc:
        detector["status"] = "skipped"
        detector["reason"] = str(exc)
        skipped_checks.append({"name": "folderguardian_log_correlation", "reason": str(exc)})
        return {"detector": detector, "skipped_checks": skipped_checks, "findings": findings}

    suspicious = [
        line
        for line in lines
        if any(marker in line.lower() for marker in ["new executable", "script created", "modified:", "deleted:"])
    ]
    detector["event_count"] = len(suspicious)
    if suspicious:
        findings.append(
            _finding(
                "medium",
                "folderguardian_log",
                str(log_path),
                f"FolderGuardian log contains {len(suspicious)} recent security-relevant events.",
                suspicious[-1][:240],
            )
        )
    return {"detector": detector, "skipped_checks": skipped_checks, "findings": findings}


def _read_windows_defender_status():
    detector = {"name": "windows_defender_status", "status": "skipped"}
    skipped_checks = []
    findings = []
    if os.name != "nt":
        detector["reason"] = "Microsoft Defender status is only available on Windows."
        skipped_checks.append({"name": "windows_defender_status", "reason": detector["reason"]})
        return {"detector": detector, "skipped_checks": skipped_checks, "findings": findings}

    command = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        "Get-MpComputerStatus | Select-Object AMServiceEnabled,AntivirusEnabled,RealTimeProtectionEnabled,AntispywareEnabled | ConvertTo-Json -Compress",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=6, shell=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        detector["reason"] = f"Defender status unavailable: {exc}"
        skipped_checks.append({"name": "windows_defender_status", "reason": detector["reason"]})
        return {"detector": detector, "skipped_checks": skipped_checks, "findings": findings}

    if result.returncode != 0:
        detector["reason"] = (result.stderr or result.stdout or "Defender status command failed.").strip()
        skipped_checks.append({"name": "windows_defender_status", "reason": detector["reason"]})
        return {"detector": detector, "skipped_checks": skipped_checks, "findings": findings}

    try:
        status = json.loads((result.stdout or "{}").strip() or "{}")
    except json.JSONDecodeError as exc:
        detector["reason"] = f"Defender status JSON parse failed: {exc}"
        skipped_checks.append({"name": "windows_defender_status", "reason": detector["reason"]})
        return {"detector": detector, "skipped_checks": skipped_checks, "findings": findings}

    detector.update({"status": "completed", "status_json": status})
    if status and status.get("RealTimeProtectionEnabled") is False:
        findings.append(
            _finding(
                "medium",
                "defender_status",
                "Microsoft Defender",
                "Microsoft Defender real-time protection appears disabled.",
                "Read-only status check.",
            )
        )
    if status and status.get("AntivirusEnabled") is False:
        findings.append(
            _finding(
                "medium",
                "defender_status",
                "Microsoft Defender",
                "Microsoft Defender antivirus appears disabled.",
                "Read-only status check.",
            )
        )
    return {"detector": detector, "skipped_checks": skipped_checks, "findings": findings}


def _read_windows_defender_history(limit):
    result = {"name": "windows_defender_history", "status": "skipped", "detections": []}
    if os.name != "nt":
        result["reason"] = "Microsoft Defender threat history is only available on Windows."
        return result
    command = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        (
            f"Get-MpThreatDetection | Select-Object -First {int(limit)} "
            "ThreatID,ThreatName,ActionSuccess,InitialDetectionTime,LastThreatStatusChangeTime,Resources "
            "| ConvertTo-Json -Compress"
        ),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=8, shell=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["reason"] = f"Defender threat history unavailable: {exc}"
        return result
    if completed.returncode != 0:
        result["reason"] = (completed.stderr or completed.stdout or "Defender history command failed.").strip()
        return result
    try:
        payload = json.loads((completed.stdout or "[]").strip() or "[]")
    except json.JSONDecodeError as exc:
        result["reason"] = f"Defender history JSON parse failed: {exc}"
        return result
    if isinstance(payload, dict):
        payload = [payload]
    result.update(
        {
            "status": "completed",
            "detections": payload if isinstance(payload, list) else [],
            "limit": limit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    return result


def _build_project_audit(target_path, max_files):
    files, collection = _collect_files(target_path, max_files=max_files)
    findings = []
    manifests = []
    dependency_counts = {}
    env_files = []
    scanned_text_files = 0
    ignored_dirs = {".git", ".vs", ".venv", "venv", "node_modules", "bin", "obj", "dist", "build", "__pycache__"}
    for path in files:
        try:
            relative_parts = set(path.relative_to(target_path).parts[:-1])
        except ValueError:
            relative_parts = set()
        if relative_parts & ignored_dirs:
            continue
        relative = _relative_label(path, target_path)
        if path.name in PROJECT_MANIFEST_NAMES or path.suffix.lower() in {".csproj", ".sln", ".vcxproj"}:
            manifest = _manifest_summary(path, target_path)
            manifests.append(manifest)
            if manifest.get("dependency_count") is not None:
                dependency_counts[manifest["kind"]] = dependency_counts.get(manifest["kind"], 0) + manifest["dependency_count"]
        if path.name.lower().startswith(".env") or path.name.lower() in {"secrets.json", "appsettings.secrets.json"}:
            env_files.append(relative)
            findings.append(
                _finding(
                    "medium",
                    "sensitive_config_file",
                    relative,
                    f"Sensitive configuration-style file found: {path.name}",
                    "Review before committing or sharing this project.",
                )
            )
        if path.suffix.lower() in TEXT_SCAN_EXTENSIONS or path.name in PROJECT_MANIFEST_NAMES or path.name.lower().startswith(".env"):
            scanned_text_files += 1
            findings.extend(_project_secret_findings(path, relative))

    if collection.get("max_files_reached"):
        findings.append(
            _finding(
                "low",
                "partial_project_audit",
                str(target_path),
                f"Project audit stopped after {max_files} files.",
                "Increase max_files for a broader audit.",
            )
        )

    risk_score = _risk_score(findings)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_path": str(target_path),
        "risk_score": risk_score,
        "risk_level": _risk_level(risk_score),
        "verdict": _verdict(_risk_level(risk_score)),
        "findings": findings,
        "manifest_count": len(manifests),
        "manifests": manifests[:80],
        "dependency_counts": dependency_counts,
        "env_files": env_files[:80],
        "scanned_text_file_count": scanned_text_files,
        "max_files": max_files,
        "partial": bool(collection.get("max_files_reached")),
        "local_only": True,
        "audit_kind": "local project security audit",
        "recommended_actions": _recommendations(_risk_level(risk_score), findings),
    }


def _manifest_summary(path, root):
    relative = _relative_label(path, root)
    name = path.name
    suffix = path.suffix.lower()
    summary = {"path": relative, "kind": name if name in PROJECT_MANIFEST_NAMES else suffix.lstrip("."), "dependency_count": None}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:256_000]
    except OSError:
        return summary
    lowered_name = name.lower()
    if lowered_name == "package.json":
        try:
            data = json.loads(text)
            deps = {}
            for key in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
                deps.update(data.get(key) or {})
            summary["dependency_count"] = len(deps)
            summary["dependencies"] = sorted(list(deps))[:40]
        except json.JSONDecodeError:
            summary["parse_error"] = True
    elif lowered_name == "requirements.txt":
        deps = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith(("#", "-r", "--"))
        ]
        summary["dependency_count"] = len(deps)
        summary["dependencies"] = deps[:40]
    elif lowered_name in {"pyproject.toml", "poetry.lock", "pipfile", "pipfile.lock", "cargo.toml", "go.mod", "composer.json", "packages.config"} or suffix in {".csproj", ".vcxproj"}:
        candidates = re.findall(r"(?:PackageReference\s+Include=|name\s*=|^require\s+|^\s*[\w.-]+\s*=|^\s*[\w./-]+)\s*[\"']?([A-Za-z0-9_.@/+:-]+)", text, flags=re.IGNORECASE | re.MULTILINE)
        summary["dependency_count"] = len(set(candidates))
        summary["dependencies"] = sorted(set(candidates))[:40]
    return summary


def _project_secret_findings(path, relative):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:MAX_TEXT_SCAN_BYTES]
    except OSError:
        return []
    findings = []
    for severity, pattern_id, pattern, message in PROJECT_SECRET_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            findings.append(_finding(severity, "secret_pattern", relative, message, pattern_id))
    return findings


def _compact_project_audit(audit):
    return {
        "target_path": audit.get("target_path"),
        "risk_score": audit.get("risk_score"),
        "risk_level": audit.get("risk_level"),
        "finding_count": len(audit.get("findings") or []),
        "manifest_count": audit.get("manifest_count"),
        "dependency_counts": audit.get("dependency_counts") or {},
        "env_files": audit.get("env_files") or [],
        "top_findings": (audit.get("findings") or [])[:8],
        "partial": audit.get("partial"),
        "local_only": True,
    }


def _format_project_audit_response(audit, artifact_path):
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    findings = audit.get("findings") or []
    lines = [
        "security_suite project audit complete.",
        f"Target: {audit['target_path']}",
        f"Verdict: {audit['risk_level']} risk, score {audit['risk_score']}/100.",
        f"Manifests: {audit['manifest_count']}; scanned text files: {audit['scanned_text_file_count']}.",
    ]
    if audit.get("dependency_counts"):
        deps = ", ".join(f"{key}: {value}" for key, value in sorted(audit["dependency_counts"].items()))
        lines.append(f"Dependencies found: {deps}.")
    if findings:
        lines.append("Evidence:")
        for finding in findings[:6]:
            lines.append(f"- {finding['severity']}: {finding['message']} ({finding['path']})")
    else:
        lines.append("Evidence: no obvious local project security issues were found.")
    if audit.get("partial"):
        lines.append(f"Note: audit was partial after {audit['max_files']} files.")
    lines.append("Note: this is a local static review, not a vulnerability database scan.")
    lines.append(artifact_line.strip())
    return "\n".join(line for line in lines if line)


def _run_defender_custom_scan(target_path, timeout_seconds):
    command = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        f"Start-MpScan -ScanType CustomScan -ScanPath '{_escape_powershell_single_quoted(target_path)}'",
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, shell=False)
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "started_at": started_at,
            "target_path": str(target_path),
            "returncode": completed.returncode,
            "stdout": (completed.stdout or "").strip(),
            "stderr": (completed.stderr or "").strip(),
            "command": ["powershell.exe", "-NoProfile", "-Command", "Start-MpScan -ScanType CustomScan -ScanPath <target>"],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "started_at": started_at,
            "target_path": str(target_path),
            "returncode": -1,
            "error": str(exc),
            "stdout": "",
            "stderr": str(exc),
        }


def _escape_powershell_single_quoted(value):
    return str(value).replace("'", "''")


def _write_assessment_artifact(artifact_dir, params, assessment):
    if not artifact_dir:
        return None
    path = write_tool_run_artifact(
        artifact_dir,
        "security_suite",
        "assess_path",
        params,
        "success",
        output_text=_compact_assessment_json(assessment),
        error_text="",
        metrics=assessment,
        command=[],
        evidence_status="not_applicable",
    )
    update_tool_run_artifact(
        path,
        target_path=assessment["target_path"],
        risk_score=assessment["risk_score"],
        risk_level=assessment["risk_level"],
        verdict=assessment["verdict"],
        findings=assessment["findings"],
        detectors=assessment["detectors"],
        file_hashes=assessment["file_hashes"],
        recommended_actions=assessment["recommended_actions"],
        timestamp=assessment["timestamp"],
        skipped_checks=assessment["skipped_checks"],
        summary=assessment["summary"],
    )
    return path


def _write_security_artifact(artifact_dir, capability, params, metrics, status):
    if not artifact_dir:
        return None
    return write_tool_run_artifact(
        artifact_dir,
        "security_suite",
        capability,
        params,
        status,
        output_text=json.dumps(metrics, indent=2),
        error_text=str(metrics.get("stderr") or metrics.get("error") or "") if isinstance(metrics, dict) else "",
        metrics=metrics,
        command=[],
        evidence_status="not_applicable",
    )


def _artifact_line(artifact_path):
    return f"\nArtifact: {artifact_path}" if artifact_path else ""


def _security_state_dir(artifact_dir):
    if artifact_dir:
        return Path(artifact_dir).parent / "security_suite"
    return Path.cwd() / "runtime_state" / "security_suite"


def _safe_quarantine_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "target")).strip("._")
    return cleaned[:80] or "target"


def _target_inventory(target_path, max_files=500):
    target_path = Path(target_path)
    if target_path.is_file():
        stat = _safe_stat(target_path)
        return {
            "kind": "file",
            "sha256": _sha256(target_path),
            "size_bytes": stat.get("size_bytes", 0),
            "modified_at": stat.get("modified_at"),
        }
    files, collection = _collect_files(target_path, max_files=max_files)
    records = []
    digest = hashlib.sha256()
    for path in files:
        relative = _relative_label(path, target_path)
        sha = _sha256(path)
        stat = _safe_stat(path)
        records.append({"path": relative, "sha256": sha, "size_bytes": stat.get("size_bytes", 0)})
        digest.update(relative.encode("utf-8", errors="ignore"))
        digest.update(sha.encode("ascii", errors="ignore"))
    return {
        "kind": "folder",
        "file_count": len(records),
        "partial": bool(collection.get("max_files_reached")),
        "manifest_sha256": digest.hexdigest(),
        "files": records[:100],
    }


def _resolve_quarantine_metadata(params, artifact_dir):
    params = params or {}
    explicit = params.get("metadata_path")
    if explicit:
        path = Path(str(explicit).strip().strip('"').strip("'")).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise ValueError(f"metadata_path does not exist: {path}")
        return path

    artifact_path = params.get("artifact_path")
    if artifact_path:
        path = Path(str(artifact_path).strip().strip('"').strip("'")).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        artifact = _read_json_file(path)
        metrics = artifact.get("metrics") if isinstance(artifact.get("metrics"), dict) else artifact
        metadata_path = metrics.get("metadata_path")
        if metadata_path:
            metadata = Path(metadata_path)
            if metadata.exists():
                return metadata
        raise ValueError(f"artifact does not contain readable quarantine metadata: {path}")

    quarantine_id = params.get("quarantine_id")
    if quarantine_id:
        candidate = _security_state_dir(artifact_dir) / "quarantine" / f"{_safe_quarantine_name(quarantine_id)}.metadata.json"
        if candidate.exists():
            return candidate
        raise ValueError(f"quarantine_id metadata was not found: {quarantine_id}")
    raise ValueError("quarantine_id, metadata_path, or artifact_path is required")


def _load_rule_pack(rules_path=None):
    built_in = [
        {
            "id": "encoded_powershell",
            "severity": "high",
            "extensions": [".ps1", ".bat", ".cmd", ".txt", ".log"],
            "pattern": r"\b(?:powershell|pwsh)(?:\.exe)?\b[^\r\n]{0,120}(?:-enc|-encodedcommand)\b",
            "message": "PowerShell encoded command pattern.",
        },
        {
            "id": "private_key_material",
            "severity": "high",
            "extensions": [".txt", ".pem", ".key", ".env", ".config", ".json"],
            "pattern": r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----",
            "message": "Private key material pattern.",
        },
        {
            "id": "download_execute_chain",
            "severity": "high",
            "extensions": [".ps1", ".bat", ".cmd", ".vbs", ".js", ".txt"],
            "pattern": r"\b(?:invoke-webrequest|iwr|curl|wget)\b[\s\S]{0,260}\b(?:iex|invoke-expression|start-process|cmd\s*/c|powershell)\b",
            "message": "Download-and-execute script pattern.",
        },
    ]
    if not rules_path:
        return built_in
    path = Path(str(rules_path).strip().strip('"').strip("'")).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return built_in
    rules = payload.get("rules") if isinstance(payload, dict) else payload
    if not isinstance(rules, list):
        return built_in
    normalized = []
    for item in rules:
        if not isinstance(item, dict) or not item.get("id") or not item.get("pattern"):
            continue
        normalized.append(
            {
                "id": str(item["id"]),
                "severity": str(item.get("severity", "medium")).lower(),
                "extensions": item.get("extensions") or [],
                "pattern": str(item["pattern"]),
                "message": str(item.get("message") or item["id"]),
                "mode": str(item.get("mode", "text")),
            }
        )
    return normalized or built_in


def _read_json_file(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _maybe_update_bayesian(artifact_path, params, bayesian_adapter):
    if not params.get("update_bayesian"):
        return "not_requested", ""
    if not bayesian_adapter or not artifact_path:
        if artifact_path:
            update_tool_run_artifact(artifact_path, evidence_status="skipped: missing adapter or artifact")
        return "skipped: missing adapter or artifact", "\nBayesian evidence update skipped because the adapter or artifact path is missing."
    try:
        evidence = bayesian_adapter.add_tool_run_evidence("security_suite", artifact_path)
        update_tool_run_artifact(
            artifact_path,
            evidence_status="updated",
            bayesian_assessment=_compact_bayesian_assessment(evidence.get("assessment", {})),
            bayesian_evidence=evidence.get("evidence", []),
            bayesian_config=evidence.get("config", {}),
        )
        assessment = evidence.get("assessment") or {}
        probability = assessment.get("probability")
        confidence = assessment.get("confidence")
        return "updated", _format_bayesian_update_summary(evidence.get("evidence") or [], probability, confidence)
    except Exception as exc:
        if artifact_path:
            update_tool_run_artifact(artifact_path, evidence_status=f"failed: {exc}")
        return f"failed: {exc}", f"\nBayesian evidence update failed: {exc}"


def _format_bayesian_update_summary(evidence_items, probability, confidence):
    added = [item for item in evidence_items if item.get("status") == "added"]
    skipped = [item for item in evidence_items if item.get("status") == "skipped"]
    failed_checks = [item for item in evidence_items if item.get("passed") is False]
    lines = ["", "Bayesian security evidence updated."]
    if probability is not None:
        confidence_text = f"; model confidence {confidence * 100:.1f}%" if confidence is not None else ""
        lines.append(f"Reliability estimate: {probability * 100:.1f}%{confidence_text}.")
    lines.append(f"Evidence added: {len(added)}; skipped: {len(skipped)}.")
    if skipped:
        reasons = {}
        for item in skipped:
            reason = item.get("reason") or "unspecified"
            reasons[reason] = reasons.get(reason, 0) + 1
        lines.append("Skipped evidence: " + ", ".join(f"{reason} ({count})" for reason, count in sorted(reasons.items())) + ".")
    if failed_checks:
        lines.append("Evidence against reliability:")
        for item in failed_checks[:3]:
            observed = item.get("observed_value")
            observed_text = f", observed {observed}" if observed is not None else ""
            lines.append(f"- {item.get('factor', 'factor')}: {item.get('description', 'check failed')}{observed_text}.")
    return "\n".join(lines)


def _compact_bayesian_assessment(assessment):
    if not isinstance(assessment, dict):
        return {}
    return {
        "status": assessment.get("status"),
        "probability": assessment.get("probability"),
        "confidence": assessment.get("confidence"),
        "credible_interval": assessment.get("credible_interval"),
        "warnings": assessment.get("warnings") or [],
    }


def _recent_security_artifacts(artifact_dir, limit):
    if not artifact_dir:
        return None
    directory = Path(artifact_dir) / "security_suite"
    if not directory.exists():
        return []
    artifacts = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return artifacts[:limit]


def _assessment_from_artifact(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    assessment = dict(metrics)
    for key in (
        "timestamp",
        "target_path",
        "risk_score",
        "risk_level",
        "verdict",
        "findings",
        "detectors",
        "file_hashes",
        "recommended_actions",
        "skipped_checks",
        "summary",
    ):
        if key in data:
            assessment[key] = data[key]
    return assessment


def _compact_assessment_json(assessment):
    compact = {
        "target_path": assessment.get("target_path"),
        "risk_score": assessment.get("risk_score"),
        "risk_level": assessment.get("risk_level"),
        "verdict": assessment.get("verdict"),
        "finding_count": len(assessment.get("findings") or []),
        "top_findings": (assessment.get("findings") or [])[:8],
        "recommended_actions": assessment.get("recommended_actions") or [],
        "skipped_checks": assessment.get("skipped_checks") or [],
        "summary": assessment.get("summary") or {},
        "local_only": True,
    }
    return json.dumps(compact, indent=2)


def _format_assessment_response(assessment, artifact_path):
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    findings = assessment.get("findings") or []
    lines = [
        "security_suite local risk assessment complete.",
        f"Target: {assessment['target_path']}",
        f"Verdict: {assessment['risk_level']} risk, score {assessment['risk_score']}/100.",
        assessment["verdict"],
    ]
    if findings:
        lines.append("Evidence:")
        for finding in findings[:6]:
            lines.append(f"- {finding['severity']}: {finding['message']} ({finding['path']})")
    else:
        lines.append("Evidence: no suspicious local indicators were found in the scanned files.")
    skipped = assessment.get("skipped_checks") or []
    if skipped:
        lines.append("Skipped checks: " + "; ".join(f"{item['name']} ({item['reason']})" for item in skipped[:4]))
    lines.append("Note: this is a local indicator review, not a guarantee.")
    lines.append(artifact_line.strip())
    return "\n".join(line for line in lines if line)


def _collect_files(target_path, max_files):
    if target_path.is_file():
        return [target_path], {"max_files_reached": False}
    files = []
    max_files_reached = False
    for root, dirs, names in os.walk(target_path):
        dirs[:] = [name for name in sorted(dirs) if not name.startswith(".") and name not in {"__pycache__"}]
        for name in sorted(names):
            path = Path(root) / name
            try:
                if path.is_symlink() or not path.is_file():
                    continue
            except OSError:
                continue
            if len(files) >= max_files:
                max_files_reached = True
                break
            files.append(path)
        if max_files_reached:
            break
    return files, {"max_files_reached": max_files_reached}


def _resolve_target_path(raw_path):
    value = str(raw_path or "").strip().strip('"').strip("'")
    if not value:
        raise ValueError("target_path is required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise ValueError(f"target_path does not exist: {path}")
    if not path.is_file() and not path.is_dir():
        raise ValueError(f"target_path must be a file or folder: {path}")
    return path


def _safe_stat(path):
    try:
        stat = Path(path).stat()
    except OSError:
        return {"size_bytes": 0, "modified_at": None, "modified_ts": datetime.now(timezone.utc).timestamp()}
    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    return {"size_bytes": stat.st_size, "modified_at": modified.isoformat(), "modified_ts": stat.st_mtime}


def _sha256(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()


def _read_binary_sample(path, max_bytes):
    try:
        with Path(path).open("rb") as handle:
            return handle.read(max_bytes)
    except OSError:
        return b""


def _read_text_prefix(path, max_bytes):
    sample = _read_binary_sample(path, max_bytes)
    return sample.decode("utf-8", errors="replace")


def _looks_text(sample):
    if not sample:
        return False
    if b"\x00" in sample[:4096]:
        return False
    textish = sum(1 for byte in sample[:4096] if byte in b"\r\n\t" or 32 <= byte <= 126)
    return textish / max(1, min(len(sample), 4096)) > 0.82


def _entropy(data):
    if not data:
        return 0.0
    counts = {}
    for byte in data:
        counts[byte] = counts.get(byte, 0) + 1
    total = len(data)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _relative_label(path, target_path):
    path = Path(path)
    target_path = Path(target_path)
    base = target_path if target_path.is_dir() else target_path.parent
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _finding(severity, category, path, message, evidence):
    return {
        "severity": severity,
        "category": category,
        "path": str(path),
        "message": message,
        "evidence": str(evidence),
        "score": SEVERITY_SCORES.get(severity, 0),
    }


def _count_findings(findings, categories):
    return len([finding for finding in findings if finding.get("category") in categories])


def _risk_score(findings):
    if not findings:
        return 0
    category_caps = {
        "script_pattern": 70,
        "double_extension": 70,
        "extension_mismatch": 60,
        "entropy": 35,
        "archive_contents": 70,
        "download_zone": 30,
        "binary_indicator": 55,
        "network_indicator": 20,
        "macro_document": 35,
        "shortcut_file": 35,
        "rule_match": 75,
        "folderguardian_log": 35,
        "defender_status": 35,
        "partial_scan": 8,
        "executable_inventory": 8,
        "script_inventory": 8,
    }
    by_category = {}
    for finding in findings:
        category = finding.get("category", "misc")
        by_category[category] = by_category.get(category, 0) + int(finding.get("score") or 0)
    total = 0
    for category, score in by_category.items():
        total += min(score, category_caps.get(category, 30))
    return min(100, total)


def _risk_level(score):
    if score >= 55:
        return "critical"
    if score >= 40:
        return "high"
    if score >= 15:
        return "elevated"
    return "low"


def _verdict(risk_level):
    if risk_level == "critical":
        return "Critical local indicators were found. Keep this target isolated and do not run files from it until reviewed."
    if risk_level == "high":
        return "High local risk indicators were found. Do not execute these files until you review the evidence."
    if risk_level == "elevated":
        return "Some suspicious local indicators were found. Review before running or trusting this target."
    return "No strong local malicious indicators were found in the scanned files."


def _recommendations(risk_level, findings):
    if risk_level in {"critical", "high"}:
        return [
            "Do not run files from this target yet.",
            "Run a full Microsoft Defender scan manually if the target came from the internet.",
            "Keep the folder isolated; use FolderGuardian monitoring if you need to watch changes.",
            "Review the listed scripts/executables and verify the source.",
        ]
    if risk_level == "elevated":
        return [
            "Review the listed findings before opening or executing files.",
            "Verify the source and hashes for downloaded content.",
            "Run a full antivirus scan if the target is untrusted.",
        ]
    if findings:
        return ["Review the low-severity findings and verify the source if this came from outside your machine."]
    return ["No immediate action suggested; keep normal backups and verify unknown downloads before running them."]


def _bounded_int(value, default, minimum, maximum):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _bool_param(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
