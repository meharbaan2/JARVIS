import csv
import io
import json
import re
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

from bayesian_adapter import BayesianAdapter
from runtime_artifacts import update_tool_run_artifact, write_tool_run_artifact
from security_adapter import run_security_suite_capability


CHANNELS = {
    "awgn": 1,
    "rayleigh": 2,
    "fading": 2,
    "ideal": 3,
}

MODULATIONS = {
    "qpsk": 1,
    "16-qam": 2,
    "16qam": 2,
    "qam16": 2,
    "64-qam": 3,
    "64qam": 3,
    "qam64": 3,
}

EQUALIZERS = {
    "mmse": 1,
    "zf": 2,
    "zero-forcing": 2,
}

CHANNEL_LABELS = {
    1: "AWGN",
    2: "Rayleigh",
    3: "Ideal",
}

MODULATION_LABELS = {
    1: "QPSK",
    2: "16-QAM",
    3: "64-QAM",
}

EQUALIZER_LABELS = {
    1: "MMSE",
    2: "ZF",
}

FOLDERGUARDIAN_SUSPICIOUS_EXTENSIONS = {".exe", ".ps1", ".bat", ".cmd", ".vbs", ".js", ".scr"}
_FOLDERGUARDIAN_WATCHERS = {}
_FOLDERGUARDIAN_WATCHERS_LOCK = threading.Lock()


@dataclass
class ToolCapability:
    name: str
    description: str
    input_schema: dict
    safety_level: str = "normal"
    supports_progress: bool = False
    supports_cancel: bool = False
    examples: List[str] = field(default_factory=list)


@dataclass
class ExternalTool:
    name: str
    description: str
    root: Path
    runtime: str
    capabilities: List[ToolCapability]

    def health(self):
        return {
            "name": self.name,
            "available": self.root.exists(),
            "root": str(self.root),
            "runtime": self.runtime,
            "capabilities": [cap.name for cap in self.capabilities],
        }


class ToolRegistry:
    def __init__(
        self,
        artifact_dir=None,
        bayesian_adapter=None,
        cadconverter_python=None,
        cadconverter_output_dir=None,
        folderguardian_bridge_exe=None,
        folderguardian_live_actions=True,
    ):
        self._tools = {}
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.bayesian_adapter = bayesian_adapter
        self.cadconverter_python = str(cadconverter_python or "python")
        self.cadconverter_output_dir = Path(cadconverter_output_dir) if cadconverter_output_dir else None
        self.folderguardian_bridge_exe = Path(folderguardian_bridge_exe) if folderguardian_bridge_exe else None
        self.folderguardian_live_actions = bool(folderguardian_live_actions)

    def register(self, tool):
        self._tools[tool.name] = tool

    def list_tools(self):
        return list(self._tools.values())

    def health(self):
        return [tool.health() for tool in self.list_tools()]

    def describe(self):
        lines = ["Available engineering capabilities:"]
        for tool in self.list_tools():
            state = "available" if tool.root.exists() else "missing"
            lines.append(f"- {tool.name} ({state}, {tool.runtime}): {tool.description}")
            for cap in tool.capabilities:
                lines.append(f"  - {cap.name}: {cap.description}")
        return "\n".join(lines)

    def run_safe_probe(self, tool_name, capability_name, memory=None):
        return self.run_capability(tool_name, capability_name, {}, memory=memory)

    def run_capability(self, tool_name, capability_name, params=None, memory=None):
        params = params or {}
        tool = self._tools.get(tool_name)
        if not tool:
            return f"Tool '{tool_name}' is not registered."
        if not tool.root.exists():
            return f"Tool '{tool_name}' is registered but its root path is missing: {tool.root}"

        if tool_name in {"ofdmsim", "ofdmsim_emi"} and capability_name == "selftest":
            return _run_ofdm_selftest(tool, params, memory, self.artifact_dir)

        if tool_name == "ofdmsim" and capability_name == "run_simulation":
            return _run_ofdm_simulation(tool, params, memory, self.artifact_dir, self.bayesian_adapter)

        if tool_name == "ofdmsim_emi" and capability_name == "compare_emi":
            return _run_ofdm_emi_compare(tool, params, memory, self.artifact_dir)

        if tool_name == "cadconverter":
            return _run_cadconverter_capability(
                tool,
                capability_name,
                params,
                memory=memory,
                artifact_dir=self.artifact_dir,
                python_executable=self.cadconverter_python,
                output_dir=self.cadconverter_output_dir,
                bayesian_adapter=self.bayesian_adapter,
            )

        if tool_name == "folderguardian":
            return _run_folderguardian_capability(
                tool,
                capability_name,
                params,
                memory,
                self.artifact_dir,
                bridge_executable=self.folderguardian_bridge_exe,
                live_actions_enabled=self.folderguardian_live_actions,
                bayesian_adapter=self.bayesian_adapter,
            )

        if tool_name in {"cpp_workspace", "python_workspace", "csharp_workspace"}:
            return _run_workspace_capability(tool, capability_name, params, memory, self.artifact_dir)

        if tool_name == "security_suite":
            return run_security_suite_capability(
                capability_name,
                params,
                memory=memory,
                artifact_dir=self.artifact_dir,
                bayesian_adapter=self.bayesian_adapter,
            )

        if tool_name == "bayesian_engine":
            if not self.bayesian_adapter:
                return "Bayesian adapter is not configured."
            return _run_bayesian_capability(self.bayesian_adapter, capability_name, params)

        return f"Capability '{capability_name}' is registered for '{tool_name}', but only safe probes are executable from the current adapter."


def _find_ofdm_exe(root):
    candidates = [
        root / "x64" / "Release" / "OFDMSim.exe",
        root / "x64" / "Debug" / "OFDMSim.exe",
        root / "build" / "Release" / "OFDMSim.exe",
        root / "build" / "Debug" / "OFDMSim.exe",
    ]
    existing = [candidate for candidate in candidates if candidate.exists()]
    if not existing:
        return None
    return max(existing, key=lambda candidate: candidate.stat().st_mtime)


def _record_and_format(tool_name, capability_name, params, result, memory, label):
    status = "success" if result.returncode == 0 else "failed"
    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()
    if memory:
        memory.record_tool_run(tool_name, capability_name, json.dumps(params, sort_keys=True), status, output, error)
    details = output or error or "No output captured."
    return f"{tool_name} {label} {status}.\n{details}"


def _run_ofdm_selftest(tool, params, memory=None, artifact_dir=None):
    exe = _find_ofdm_exe(tool.root)
    if not exe:
        return f"{tool.name} is present, but no built OFDMSim.exe was found under {tool.root}."
    command = [str(exe), "test"]
    result = subprocess.run(command, cwd=str(tool.root), capture_output=True, text=True, timeout=60)
    status = "success" if result.returncode == 0 else "failed"
    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()
    metrics = _parse_selftest_metrics(output, error)
    artifact_path = _write_tool_artifact(
        artifact_dir,
        tool.name,
        "selftest",
        params,
        status,
        output,
        error,
        metrics,
        command,
    )
    _record_tool_memory(memory, tool.name, "selftest", params, status, output, error, artifact_path, metrics, "not_requested")
    details = output or error or "No output captured."
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    return f"{tool.name} self-test {status}.\n{details}{artifact_line}"


def _coerce_choice(value, choices, default):
    if value is None:
        return default
    if isinstance(value, int):
        return value
    value_text = str(value).strip().lower()
    if value_text.isdigit():
        return int(value_text)
    return choices.get(value_text, default)


def _normalize_ofdm_params(params):
    channel = _coerce_choice(params.get("channel"), CHANNELS, 1)
    modulation = _coerce_choice(params.get("modulation"), MODULATIONS, 1)
    equalizer = _coerce_choice(params.get("equalizer"), EQUALIZERS, 1)
    snr_db = float(params.get("snr_db", 20))
    frames = int(params.get("frames", 100))
    seed = int(params.get("seed", 42))

    if channel not in {1, 2, 3}:
        raise ValueError("channel must be AWGN, Rayleigh, Ideal, or 1-3")
    if modulation not in {1, 2, 3}:
        raise ValueError("modulation must be QPSK, 16-QAM, 64-QAM, or 1-3")
    if equalizer not in {1, 2}:
        raise ValueError("equalizer must be MMSE, ZF, 1, or 2")
    if frames < 1 or frames > 100000:
        raise ValueError("frames must be between 1 and 100000")
    if snr_db < -50 or snr_db > 100:
        raise ValueError("snr_db must be between -50 and 100")

    return {
        "channel": channel,
        "modulation": modulation,
        "equalizer": equalizer,
        "snr_db": snr_db,
        "frames": frames,
        "seed": seed,
    }


def _run_ofdm_simulation(tool, params, memory=None, artifact_dir=None, bayesian_adapter=None):
    exe = _find_ofdm_exe(tool.root)
    if not exe:
        return f"{tool.name} is present, but no built OFDMSim.exe was found under {tool.root}."
    try:
        normalized = _normalize_ofdm_params(params)
    except ValueError as exc:
        return f"Invalid OFDMSim parameters: {exc}"

    command = [
        str(exe),
        "run",
        "--channel",
        str(normalized["channel"]),
        "--modulation",
        str(normalized["modulation"]),
        "--eq",
        str(normalized["equalizer"]),
        "--snr-db",
        str(normalized["snr_db"]),
        "--frames",
        str(normalized["frames"]),
        "--seed",
        str(normalized["seed"]),
        "--quiet",
    ]
    result = subprocess.run(command, cwd=str(tool.root), capture_output=True, text=True, timeout=120)
    status = "success" if result.returncode == 0 else "failed"
    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()
    metrics = _parse_ofdm_quiet_metrics(output) if result.returncode == 0 else {}
    artifact_path = None
    evidence_status = "not_requested"
    bayesian_summary = ""
    artifact_evidence_status = "pending" if params.get("update_bayesian") else evidence_status
    if artifact_dir:
        artifact_path = write_tool_run_artifact(
            artifact_dir,
            tool.name,
            "run_simulation",
            normalized,
            status,
            output_text=output,
            error_text=error,
            metrics=metrics,
            command=command,
            evidence_status=artifact_evidence_status,
        )

    if params.get("update_bayesian"):
        if bayesian_adapter and artifact_path:
            try:
                evidence = bayesian_adapter.add_tool_run_evidence(
                    tool.name,
                    artifact_path,
                    ber_threshold=float(params.get("ber_threshold", 0.01)),
                )
                evidence_status = "updated"
                bayesian_summary = _format_bayesian_update(evidence)
                update_tool_run_artifact(
                    artifact_path,
                    evidence_status=evidence_status,
                    bayesian_assessment=_compact_bayesian_assessment(evidence.get("assessment", {})),
                    bayesian_evidence=evidence.get("evidence", []),
                    bayesian_config=evidence.get("config", {}),
                )
            except Exception as exc:
                evidence_status = f"failed: {exc}"
                bayesian_summary = f"\nBayesian evidence update failed: {exc}"
                update_tool_run_artifact(artifact_path, evidence_status=evidence_status)
        else:
            evidence_status = "skipped: missing adapter or artifact"
            bayesian_summary = "\nBayesian evidence update skipped because the adapter or artifact path is missing."
            update_tool_run_artifact(artifact_path, evidence_status=evidence_status)

    if memory:
        memory.record_tool_run(
            tool.name,
            "run_simulation",
            json.dumps(normalized, sort_keys=True),
            status,
            output,
            error,
            artifact_path=artifact_path,
            metrics_json=json.dumps(metrics, sort_keys=True),
            evidence_status=evidence_status,
        )
    if result.returncode != 0:
        artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
        return f"{tool.name} simulation failed.\n{error or output or 'No output captured.'}{artifact_line}{bayesian_summary}"
    summary = _format_ofdm_quiet_output(output)
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    return f"{tool.name} simulation success.\nParameters: {_format_ofdm_params(normalized)}\n{summary}{artifact_line}{bayesian_summary}"


def _normalize_emi_compare_params(params):
    normalized = _normalize_ofdm_params(params)
    emi_probability = float(params.get("emi_probability", params.get("emi_impulse_probability", 0.003)))
    emi_amplitude = float(params.get("emi_amplitude", params.get("emi_impulse_amplitude_rms", 15.0)))
    emi_width = int(params.get("emi_width", params.get("emi_impulse_width", 1)))
    clip_threshold = float(params.get("clip_threshold", params.get("clip_threshold_rms", 6.0)))

    if emi_probability < 0 or emi_probability > 1:
        raise ValueError("emi_probability must be between 0 and 1")
    if emi_amplitude < 0 or emi_amplitude > 1000:
        raise ValueError("emi_amplitude must be between 0 and 1000")
    if emi_width < 1 or emi_width > 10000:
        raise ValueError("emi_width must be between 1 and 10000")
    if clip_threshold <= 0 or clip_threshold > 1000:
        raise ValueError("clip_threshold must be greater than 0 and at most 1000")

    normalized.update(
        {
            "emi_probability": emi_probability,
            "emi_amplitude": emi_amplitude,
            "emi_width": emi_width,
            "clip_threshold": clip_threshold,
        }
    )
    return normalized


def _run_ofdm_emi_compare(tool, params, memory=None, artifact_dir=None):
    exe = _find_ofdm_exe(tool.root)
    if not exe:
        return f"{tool.name} is present, but no built OFDMSim.exe was found under {tool.root}."
    try:
        normalized = _normalize_emi_compare_params(params)
    except ValueError as exc:
        return f"Invalid OFDMSim EMI parameters: {exc}"

    command = [str(exe), "compare"]
    config = Path(tool.root) / "examples" / "sim_default.ini"
    if config.exists():
        command.extend(["--config", str(config)])
    command.extend(
        [
            "--channel",
            str(normalized["channel"]),
            "--modulation",
            str(normalized["modulation"]),
            "--equalizer",
            str(normalized["equalizer"]),
            "--snr-db",
            str(normalized["snr_db"]),
            "--frames",
            str(normalized["frames"]),
            "--seed",
            str(normalized["seed"]),
            "--emi-probability",
            str(normalized["emi_probability"]),
            "--emi-amplitude",
            str(normalized["emi_amplitude"]),
            "--emi-width",
            str(normalized["emi_width"]),
            "--clip-threshold",
            str(normalized["clip_threshold"]),
        ]
    )
    result = subprocess.run(command, cwd=str(tool.root), capture_output=True, text=True, timeout=180)
    status = "success" if result.returncode == 0 else "failed"
    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()
    metrics = _parse_emi_compare_metrics(output) if result.returncode == 0 else {}
    artifact_path = _write_tool_artifact(artifact_dir, tool.name, "compare_emi", normalized, status, output, error, metrics, command)
    _record_tool_memory(memory, tool.name, "compare_emi", normalized, status, output, error, artifact_path, metrics, "not_requested")
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    if result.returncode != 0:
        return f"{tool.name} EMI comparison failed.\n{error or output or 'No output captured.'}{artifact_line}"
    return f"{tool.name} EMI comparison success.\n{_format_emi_compare_metrics(metrics)}{artifact_line}"


def _parse_emi_compare_metrics(output):
    csv_lines = [
        line
        for line in (output or "").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not csv_lines:
        return {}
    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
    cases = []
    for row in reader:
        if not row.get("case"):
            continue
        cases.append(
            {
                "case": row["case"],
                "ber": _float_or_none(row.get("ber")),
                "errors": _int_or_none(row.get("errors")),
                "bits": _int_or_none(row.get("bits")),
                "emi_impulses": _int_or_none(row.get("emi_impulses")),
                "emi_samples_hit": _int_or_none(row.get("emi_samples_hit")),
                "clipped_samples": _int_or_none(row.get("clipped_samples")),
            }
        )
    metrics = {"cases": cases}
    by_case = {case["case"]: case for case in cases}
    for name, prefix in (("baseline", "baseline"), ("emi", "emi"), ("emi_clipping", "emi_clipping")):
        if name in by_case:
            metrics[f"{prefix}_ber"] = by_case[name]["ber"]
            metrics[f"{prefix}_errors"] = by_case[name]["errors"]
            metrics[f"{prefix}_bits"] = by_case[name]["bits"]
    baseline_ber = metrics.get("baseline_ber")
    emi_ber = metrics.get("emi_ber")
    clipped_ber = metrics.get("emi_clipping_ber")
    if baseline_ber is not None and emi_ber is not None:
        metrics["emi_ber_delta"] = emi_ber - baseline_ber
    if emi_ber is not None and clipped_ber is not None:
        metrics["clipping_ber_improvement"] = emi_ber - clipped_ber
    return metrics


def _format_emi_compare_metrics(metrics):
    if not metrics.get("cases"):
        return "No comparison metrics parsed."
    baseline = metrics.get("baseline_ber")
    emi = metrics.get("emi_ber")
    clipped = metrics.get("emi_clipping_ber")
    delta = metrics.get("emi_ber_delta")
    improvement = metrics.get("clipping_ber_improvement")
    parts = [
        f"baseline BER {baseline}" if baseline is not None else "baseline BER unknown",
        f"EMI BER {emi}" if emi is not None else "EMI BER unknown",
        f"clipped BER {clipped}" if clipped is not None else "clipped BER unknown",
    ]
    if delta is not None:
        parts.append(f"EMI delta {delta}")
    if improvement is not None:
        parts.append(f"clipping improvement {improvement}")
    return "; ".join(parts) + "."


def _parse_ofdm_quiet_metrics(output):
    parts = output.split()
    if len(parts) < 5:
        return {}
    try:
        return {
            "ber": float(parts[0]),
            "errors": int(parts[1]),
            "bits": int(parts[2]),
            "es_n0_db": float(parts[3]),
            "eb_n0_db": float(parts[4]),
        }
    except ValueError:
        return {}


def _parse_selftest_metrics(output, error=""):
    combined = "\n".join(part for part in [output, error] if part)
    ok_count = len(re.findall(r"^\[ OK \]", combined, flags=re.MULTILINE))
    failed_count = len(re.findall(r"^\[FAIL\]|^\[ FAILED \]", combined, flags=re.MULTILINE))
    return {
        "ok_count": ok_count,
        "failed_count": failed_count,
        "all_passed": "All self-tests passed." in combined,
    }


def _format_ofdm_quiet_output(output):
    parts = output.split()
    if len(parts) >= 5:
        ber, errors, bits, es_n0, eb_n0 = parts[:5]
        return f"BER {ber}, errors {errors}/{bits}, Es/N0 {es_n0} dB, Eb/N0 {eb_n0} dB."
    return output or "No output captured."


def _format_ofdm_params(params):
    channel = CHANNEL_LABELS.get(params.get("channel"), params.get("channel"))
    modulation = MODULATION_LABELS.get(params.get("modulation"), params.get("modulation"))
    equalizer = EQUALIZER_LABELS.get(params.get("equalizer"), params.get("equalizer"))
    return (
        f"channel {channel}, modulation {modulation}, equalizer {equalizer}, "
        f"SNR {params.get('snr_db')} dB, frames {params.get('frames')}, seed {params.get('seed')}."
    )


def _float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_bayesian_update(evidence):
    assessment = evidence.get("assessment", {})
    probability = assessment.get("probability")
    confidence = assessment.get("confidence")
    if probability is None:
        return "\nBayesian evidence updated, but no probability was returned."
    confidence_text = f"{confidence * 100:.1f}%" if confidence is not None else "unknown"
    interval = assessment.get("credible_interval") or []
    interval_text = ""
    if len(interval) == 2:
        interval_text = f", interval {interval[0] * 100:.1f}-{interval[1] * 100:.1f}%"
    lines = [
        (
            "\nBayesian evidence updated. "
            f"Tool confidence probability: {probability * 100:.1f}% "
            f"(confidence {confidence_text}{interval_text})."
        )
    ]
    breakdown = _format_bayesian_evidence_breakdown(evidence.get("evidence") or [])
    if breakdown:
        lines.append(breakdown)
    return "\n".join(lines)


def _compact_bayesian_assessment(assessment):
    return {
        "status": assessment.get("status"),
        "probability": assessment.get("probability"),
        "confidence": assessment.get("confidence"),
        "credible_interval": assessment.get("credible_interval"),
        "warnings": assessment.get("warnings") or [],
    }


def _format_bayesian_evidence_breakdown(items):
    if not items:
        return ""
    labels = {
        "Simulation executed successfully": "execution",
        "Output is machine-readable": "machine-readable output",
        "Simulation result meets expected quality": "quality threshold",
        "CAD operation executed successfully": "CAD execution",
        "CAD output is machine-readable": "CAD metrics",
        "CAD result has usable geometry or output artifact": "CAD quality",
        "FolderGuardian operation executed successfully": "FolderGuardian execution",
        "FolderGuardian bridge output is machine-readable": "FolderGuardian metrics",
        "FolderGuardian operation completed without item errors": "FolderGuardian item errors",
    }
    parts = []
    skipped = 0
    for item in items:
        label = labels.get(item.get("factor"), item.get("factor", "evidence"))
        if item.get("status") == "skipped":
            skipped += 1
            parts.append(f"{label} skipped ({item.get('reason', 'policy')})")
            continue
        state = "passed" if item.get("passed") else "failed"
        parts.append(f"{label} {state}")
    if skipped == len(items):
        prefix = "Evidence checks: no new evidence added; "
    else:
        prefix = "Evidence checks: "
    return prefix + "; ".join(parts) + "."


def _maybe_update_bayesian_evidence(tool_name, artifact_path, params, bayesian_adapter):
    if not params.get("update_bayesian"):
        return "not_requested", ""
    if not bayesian_adapter or not artifact_path:
        evidence_status = "skipped: missing adapter or artifact"
        if artifact_path:
            update_tool_run_artifact(artifact_path, evidence_status=evidence_status)
        return evidence_status, "\nBayesian evidence update skipped because the adapter or artifact path is missing."
    try:
        evidence = bayesian_adapter.add_tool_run_evidence(
            tool_name,
            artifact_path,
            ber_threshold=float(params.get("ber_threshold", 0.01)),
        )
        evidence_status = "updated"
        update_tool_run_artifact(
            artifact_path,
            evidence_status=evidence_status,
            bayesian_assessment=_compact_bayesian_assessment(evidence.get("assessment", {})),
            bayesian_evidence=evidence.get("evidence", []),
            bayesian_config=evidence.get("config", {}),
        )
        return evidence_status, _format_bayesian_update(evidence)
    except Exception as exc:
        evidence_status = f"failed: {exc}"
        update_tool_run_artifact(artifact_path, evidence_status=evidence_status)
        return evidence_status, f"\nBayesian evidence update failed: {exc}"


def _run_cadconverter_capability(
    tool,
    capability_name,
    params,
    memory=None,
    artifact_dir=None,
    python_executable="python",
    output_dir=None,
    bayesian_adapter=None,
):
    if capability_name == "inspect":
        return _inspect_cadconverter(tool, params, memory, artifact_dir)
    if capability_name == "recommend_conversion":
        return _recommend_cadconverter_workflow(tool, params, memory, artifact_dir)
    if capability_name == "batch_validate_glb":
        return _batch_validate_cadconverter_glb(tool, params, memory, artifact_dir, python_executable, bayesian_adapter)
    if capability_name == "validate_glb":
        return _run_cadconverter_validate(tool, params, memory, artifact_dir, python_executable, bayesian_adapter)
    if capability_name == "convert_step_to_glb":
        return _run_cadconverter_step_to_glb(tool, params, memory, artifact_dir, python_executable, output_dir, bayesian_adapter)
    if capability_name == "convert_glb_to_step":
        return _run_cadconverter_glb_to_step(tool, params, memory, artifact_dir, python_executable, output_dir, bayesian_adapter)
    return f"Unknown CADConverter capability: {capability_name}"


def _inspect_cadconverter(tool, params, memory=None, artifact_dir=None):
    root = Path(tool.root)
    step_files = _cad_files(root, {".stp", ".step"})
    glb_files = _cad_files(root, {".glb", ".gltf"})
    metrics = {
        "step_count": len(step_files),
        "glb_count": len(glb_files),
        "step_files": [str(path.relative_to(root)) for path in step_files[:20]],
        "glb_files": [str(path.relative_to(root)) for path in glb_files[:20]],
    }
    output = json.dumps(metrics, indent=2)
    artifact_path = _write_tool_artifact(
        artifact_dir,
        tool.name,
        "inspect",
        params,
        "success",
        output,
        "",
        metrics,
        command=None,
    )
    _record_tool_memory(memory, tool.name, "inspect", params, "success", output, "", artifact_path, metrics, "not_applicable")
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    return (
        "cadconverter inspect success.\n"
        f"STEP/STP files: {metrics['step_count']}. GLB/GLTF files: {metrics['glb_count']}.\n"
        f"Sample STEP files: {', '.join(metrics['step_files'][:5]) or 'none'}.\n"
        f"Sample GLB files: {', '.join(metrics['glb_files'][:5]) or 'none'}."
        f"{artifact_line}"
    )


def _recommend_cadconverter_workflow(tool, params, memory=None, artifact_dir=None):
    root = Path(tool.root)
    raw_input = params.get("input_path") or params.get("file") or params.get("path")
    goal = str(params.get("goal") or params.get("intent") or "").strip()
    candidates = []
    target_path = None
    if raw_input:
        try:
            suffixes = {".stp", ".step", ".glb", ".gltf"}
            target_path = _resolve_cad_input_path(root, raw_input, suffixes)
        except ValueError:
            target_path = None
    if not target_path:
        candidates = [str(path.relative_to(root)) for path in (_cad_files(root, {".stp", ".step"}) + _cad_files(root, {".glb", ".gltf"}))[:12]]

    recommendations = []
    if target_path and target_path.suffix.lower() in {".stp", ".step"}:
        recommendations.extend(
            [
                "Use convert_step_to_glb with validation enabled for AR/VR viewing.",
                "Start with default deflection settings; tighten linear_deflection only if curved faces look rough.",
                "Keep output in Hivemind runtime state so generated assets do not clutter the CADConverter repo.",
            ]
        )
        next_action = "convert_step_to_glb"
    elif target_path and target_path.suffix.lower() in {".glb", ".gltf"}:
        recommendations.extend(
            [
                "Validate the GLB first to check mesh count, materials, colors, normals, and named nodes.",
                "Use reconstructed mode for cleaner STEP recovery; use faceted mode if reconstruction fails.",
                "Use unit_scale only when the source model clearly has a scale mismatch.",
            ]
        )
        next_action = "validate_glb"
    else:
        recommendations.extend(
            [
                "Inspect available STEP/STP and GLB/GLTF files, then choose a specific input file.",
                "For AR/VR, convert STEP/STP to GLB and validate immediately after conversion.",
                "For reverse engineering, validate GLB first, then reconstruct to STEP with a conservative mode.",
            ]
        )
        next_action = "inspect"

    metrics = {
        "goal": goal,
        "input_path": str(target_path) if target_path else None,
        "candidate_files": candidates,
        "next_action": next_action,
        "recommendations": recommendations,
    }
    output = json.dumps(metrics, indent=2)
    artifact_path = _write_tool_artifact(artifact_dir, tool.name, "recommend_conversion", params, "success", output, "", metrics, None)
    _record_tool_memory(memory, tool.name, "recommend_conversion", params, "success", output, "", artifact_path, metrics, "not_applicable")
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    target_text = f" for {target_path.name}" if target_path else ""
    return (
        f"cadconverter recommendation ready{target_text}.\n"
        f"Next action: {next_action}.\n"
        + "\n".join(f"- {item}" for item in recommendations)
        + (f"\nCandidate files: {', '.join(candidates[:6])}." if candidates else "")
        + artifact_line
    )


def _batch_validate_cadconverter_glb(tool, params, memory=None, artifact_dir=None, python_executable="python", bayesian_adapter=None):
    root = Path(tool.root)
    max_files = int(params.get("max_files", 5) or 5)
    max_files = max(1, min(max_files, 25))
    explicit = params.get("input_paths") or params.get("files")
    if isinstance(explicit, str):
        explicit = [item.strip() for item in explicit.split(",") if item.strip()]
    if explicit:
        candidates = []
        for item in explicit:
            try:
                candidates.append(_resolve_cad_input_path(root, item, {".glb", ".gltf"}))
            except ValueError:
                continue
    else:
        candidates = _cad_files(root, {".glb", ".gltf"})
    candidates = candidates[:max_files]
    if not candidates:
        return "cadconverter batch validation found no GLB/GLTF files to validate."

    results = []
    for path in candidates:
        command = [str(python_executable), "-m", "cadconverter.cli", "validate", str(path)]
        completed = subprocess.run(command, cwd=str(tool.root), capture_output=True, text=True, timeout=180)
        output = (completed.stdout or "").strip()
        error = (completed.stderr or "").strip()
        results.append(
            {
                "file": str(path.relative_to(root)),
                "status": "success" if completed.returncode == 0 else "failed",
                "metrics": _parse_cadconverter_metrics(output),
                "stdout": output,
                "stderr": error,
            }
        )
    status = "success" if all(item["status"] == "success" for item in results) else "failed"
    metrics = {
        "validated_count": len(results),
        "success_count": sum(1 for item in results if item["status"] == "success"),
        "failed_count": sum(1 for item in results if item["status"] != "success"),
        "results": results,
    }
    output = json.dumps(metrics, indent=2)
    artifact_path = _write_tool_artifact(artifact_dir, tool.name, "batch_validate_glb", params, status, output, "", metrics, None)
    evidence_status, bayesian_summary = _maybe_update_bayesian_evidence(tool.name, artifact_path, params, bayesian_adapter)
    _record_tool_memory(memory, tool.name, "batch_validate_glb", params, status, output, "", artifact_path, metrics, evidence_status)
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    lines = [
        f"cadconverter batch_validate_glb {status}.",
        f"Validated {metrics['validated_count']} files: {metrics['success_count']} success, {metrics['failed_count']} failed.",
    ]
    for item in results[:5]:
        lines.append(f"- {item['file']}: {item['status']}")
    if bayesian_summary:
        lines.append(bayesian_summary.strip())
    lines.append(artifact_line.strip())
    return "\n".join(line for line in lines if line)


def _run_cadconverter_validate(tool, params, memory=None, artifact_dir=None, python_executable="python", bayesian_adapter=None):
    try:
        input_path = _resolve_cad_input_path(tool.root, params.get("input_path") or params.get("glb_path"), {".glb", ".gltf"})
    except ValueError as exc:
        return f"Invalid CADConverter validate request: {exc}"
    command = [str(python_executable), "-m", "cadconverter.cli", "validate", str(input_path)]
    return _run_cadconverter_command(tool, "validate_glb", params, command, memory, artifact_dir, timeout=180, bayesian_adapter=bayesian_adapter)


def _run_cadconverter_step_to_glb(tool, params, memory=None, artifact_dir=None, python_executable="python", output_dir=None, bayesian_adapter=None):
    try:
        input_path = _resolve_cad_input_path(tool.root, params.get("input_path") or params.get("step_path"), {".stp", ".step"})
        output_path = _resolve_cad_output_path(
            tool.root,
            output_dir,
            params.get("output_path"),
            {".glb"},
            default_name=f"{input_path.stem}.glb",
        )
    except ValueError as exc:
        return f"Invalid CADConverter STEP to GLB request: {exc}"

    command = [
        str(python_executable),
        "-m",
        "cadconverter.cli",
        "convert",
        str(input_path),
        "-o",
        str(output_path),
        "--overwrite",
    ]
    if params.get("validate", True):
        command.append("--validate")
    if params.get("single_sided"):
        command.append("--single-sided")
    for key, flag in (("linear_deflection", "--linear-deflection"), ("angular_deflection", "--angular-deflection")):
        if params.get(key) is not None:
            command.extend([flag, str(params[key])])

    return _run_cadconverter_command(
        tool,
        "convert_step_to_glb",
        {**params, "input_path": str(input_path), "output_path": str(output_path)},
        command,
        memory,
        artifact_dir,
        timeout=900,
        output_path=output_path,
        bayesian_adapter=bayesian_adapter,
    )


def _run_cadconverter_glb_to_step(tool, params, memory=None, artifact_dir=None, python_executable="python", output_dir=None, bayesian_adapter=None):
    mode = str(params.get("mode", "reconstructed")).lower().strip()
    if mode not in {"advanced", "reconstructed", "faceted"}:
        return "Invalid CADConverter GLB to STEP request: mode must be advanced, reconstructed, or faceted."
    try:
        input_path = _resolve_cad_input_path(tool.root, params.get("input_path") or params.get("glb_path"), {".glb", ".gltf"})
        output_path = _resolve_cad_output_path(
            tool.root,
            output_dir,
            params.get("output_path"),
            {".stp", ".step"},
            default_name=f"{input_path.stem}_{mode}.step",
        )
    except ValueError as exc:
        return f"Invalid CADConverter GLB to STEP request: {exc}"

    command = [
        str(python_executable),
        "-m",
        "cadconverter.cli",
        "glb-to-step",
        str(input_path),
        "-o",
        str(output_path),
        "--mode",
        mode,
        "--overwrite",
    ]
    for key, flag in (
        ("unit_scale", "--unit-scale"),
        ("angle_tolerance", "--angle-tolerance"),
        ("plane_tolerance", "--plane-tolerance"),
    ):
        if params.get(key) is not None:
            command.extend([flag, str(params[key])])
    for key, flag in (
        ("reconstruct_holes", "--reconstruct-holes"),
        ("reconstruct_cylinders", "--reconstruct-cylinders"),
        ("reconstruct_spheres", "--reconstruct-spheres"),
        ("reconstruct_cones", "--reconstruct-cones"),
        ("make_solids", "--make-solids"),
        ("no_sew", "--no-sew"),
    ):
        if params.get(key):
            command.append(flag)

    return _run_cadconverter_command(
        tool,
        "convert_glb_to_step",
        {**params, "input_path": str(input_path), "output_path": str(output_path), "mode": mode},
        command,
        memory,
        artifact_dir,
        timeout=900,
        output_path=output_path,
        bayesian_adapter=bayesian_adapter,
    )


def _run_cadconverter_command(tool, capability_name, params, command, memory, artifact_dir, timeout, output_path=None, bayesian_adapter=None):
    result = subprocess.run(command, cwd=str(tool.root), capture_output=True, text=True, timeout=timeout)
    status = "success" if result.returncode == 0 else "failed"
    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()
    metrics = _parse_cadconverter_metrics(output)
    if output_path:
        metrics["output_path"] = str(output_path)
        if Path(output_path).exists():
            metrics["output_size_bytes"] = Path(output_path).stat().st_size
    artifact_path = _write_tool_artifact(artifact_dir, tool.name, capability_name, params, status, output, error, metrics, command)
    evidence_status, bayesian_summary = _maybe_update_bayesian_evidence(tool.name, artifact_path, params, bayesian_adapter)
    _record_tool_memory(memory, tool.name, capability_name, params, status, output, error, artifact_path, metrics, evidence_status)
    details = output or error or "No output captured."
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    output_line = f"\nOutput: {output_path}" if output_path else ""
    return f"cadconverter {capability_name} {status}.\n{details}{output_line}{artifact_line}{bayesian_summary}"


def _write_tool_artifact(artifact_dir, tool_name, capability_name, params, status, output, error, metrics, command):
    if not artifact_dir:
        return None
    return write_tool_run_artifact(
        artifact_dir,
        tool_name,
        capability_name,
        params,
        status,
        output_text=output,
        error_text=error,
        metrics=metrics,
        command=command,
        evidence_status=None,
    )


def _record_tool_memory(memory, tool_name, capability_name, params, status, output, error, artifact_path, metrics, evidence_status):
    if not memory:
        return None
    return memory.record_tool_run(
        tool_name,
        capability_name,
        json.dumps(params, sort_keys=True),
        status,
        output,
        error,
        artifact_path=artifact_path,
        metrics_json=json.dumps(metrics, sort_keys=True),
        evidence_status=evidence_status,
    )


def _cad_files(root, suffixes):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _is_within(path, parent):
    path = Path(path).resolve()
    parent = Path(parent).resolve()
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_cad_input_path(root, raw_path, allowed_suffixes):
    if not raw_path:
        raise ValueError("input_path is required")
    root = Path(root).resolve()
    path = Path(str(raw_path).strip())
    candidate = path if path.is_absolute() else root / path
    candidate = candidate.resolve()
    if not _is_within(candidate, root):
        raise ValueError(f"input_path must be inside the CADConverter root: {root}")
    if candidate.suffix.lower() not in allowed_suffixes:
        raise ValueError(f"input_path must use one of: {', '.join(sorted(allowed_suffixes))}")
    if not candidate.exists():
        raise ValueError(f"input_path does not exist: {candidate}")
    return candidate


def _resolve_cad_output_path(root, output_dir, raw_path, allowed_suffixes, default_name):
    root = Path(root).resolve()
    output_base = Path(output_dir).resolve() if output_dir else root
    output_base.mkdir(parents=True, exist_ok=True)
    if raw_path:
        path = Path(str(raw_path).strip())
        candidate = path if path.is_absolute() else output_base / path
    else:
        candidate = output_base / default_name
    candidate = candidate.resolve()
    if not (_is_within(candidate, output_base) or _is_within(candidate, root)):
        raise ValueError(f"output_path must stay inside {output_base} or {root}")
    if candidate.suffix.lower() not in allowed_suffixes:
        raise ValueError(f"output_path must use one of: {', '.join(sorted(allowed_suffixes))}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _parse_cadconverter_metrics(output):
    metrics = {}
    patterns = {
        "nodes": r"nodes=(\d+)",
        "meshes": r"meshes=(\d+)",
        "materials": r"materials=(\d+)",
        "primitives": r"primitives=(\d+)",
        "triangles": r"triangles=(\d+)",
        "faces": r"faces=(\d+)",
        "skipped": r"skipped=(\d+)",
        "planar_regions": r"planar_regions=(\d+)",
        "free_edges": r"free_edges=(\d+)",
        "shells": r"shells=(\d+)",
        "solids": r"solids=(\d+)",
        "unit_scale": r"unit_scale=([0-9.eE+-]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output or "")
        if not match:
            continue
        value = match.group(1)
        metrics[key] = float(value) if key == "unit_scale" else int(value)
    color_match = re.search(r"colors=(\d+)/(\d+)", output or "")
    if color_match:
        metrics["colored_primitives"] = int(color_match.group(1))
        metrics["color_primitive_total"] = int(color_match.group(2))
    normal_match = re.search(r"normals=(\d+)/(\d+)", output or "")
    if normal_match:
        metrics["normal_primitives"] = int(normal_match.group(1))
        metrics["normal_primitive_total"] = int(normal_match.group(2))
    named_match = re.search(r"named_nodes=(\d+)/(\d+)", output or "")
    if named_match:
        metrics["named_nodes"] = int(named_match.group(1))
        metrics["named_node_total"] = int(named_match.group(2))
    return metrics


def _run_folderguardian_capability(
    tool,
    capability_name,
    params,
    memory=None,
    artifact_dir=None,
    bridge_executable=None,
    live_actions_enabled=True,
    bayesian_adapter=None,
):
    if capability_name == "inspect":
        return _inspect_folderguardian(tool, params, memory, artifact_dir)
    if capability_name == "security_review":
        return _folderguardian_security_review(tool, params, memory, artifact_dir)
    if capability_name in {"encrypt_folder", "decrypt_folder"}:
        return _folderguardian_live_crypto_action(
            tool,
            capability_name,
            params,
            memory,
            artifact_dir,
            bridge_executable,
            live_actions_enabled,
            bayesian_adapter,
        )
    if capability_name == "check_logs":
        return _folderguardian_check_logs(tool, params, memory, artifact_dir)
    if capability_name == "start_watch":
        return _folderguardian_start_watch(tool, params, memory, artifact_dir)
    if capability_name == "stop_watch":
        return _folderguardian_stop_watch(tool, params, memory, artifact_dir)
    if capability_name == "watch_status":
        return _folderguardian_watch_status(tool, params, memory, artifact_dir)
    if capability_name == "dry_run_protection_plan":
        return _folderguardian_dry_run_plan(tool, params, memory, artifact_dir)
    return (
        f"Capability '{capability_name}' is registered for FolderGuardian, but this adapter is "
        "not implemented by the current bridge adapter."
    )


def _inspect_folderguardian(tool, params, memory=None, artifact_dir=None):
    root = Path(tool.root)
    metrics = _folderguardian_project_metrics(root)
    output = json.dumps(metrics, indent=2)
    artifact_path = _write_tool_artifact(
        artifact_dir,
        tool.name,
        "inspect",
        params,
        "success",
        output,
        "",
        metrics,
        command=None,
    )
    _record_tool_memory(memory, tool.name, "inspect", params, "success", output, "", artifact_path, metrics, "not_applicable")
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    frameworks = ", ".join(metrics["target_frameworks"]) or "unknown"
    feature_text = "; ".join(metrics["detected_features"]) or "none detected"
    return (
        "folderguardian inspect success.\n"
        f"Files: {metrics['solution_count']} solution, {metrics['project_count']} project, "
        f"{metrics['csharp_file_count']} C# files, {metrics['xaml_file_count']} XAML files.\n"
        f"Frameworks: {frameworks}. WPF: {_yes_no(metrics['uses_wpf'])}. "
        f"Windows Forms: {_yes_no(metrics['uses_windows_forms'])}.\n"
        f"Detected capabilities: {feature_text}.\n"
        "Adapter safety: live encrypt/decrypt uses the FolderGuardianBridge sidecar; dry-run remains available for planning."
        f"{artifact_line}"
    )


def _folderguardian_resolve_user_folder(raw_path, create=False):
    value = str(raw_path or "").strip().strip('"').strip("'")
    if not value:
        raise ValueError("folder_path is required")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("folder_path must resolve to an absolute path")
    path = path.resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        raise ValueError(f"folder_path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"folder_path is not a folder: {path}")
    return path


def _folderguardian_live_crypto_action(
    tool,
    capability_name,
    params,
    memory=None,
    artifact_dir=None,
    bridge_executable=None,
    live_actions_enabled=True,
    bayesian_adapter=None,
):
    action = "encrypt" if capability_name == "encrypt_folder" else "decrypt"
    if not live_actions_enabled:
        plan_params = dict(params or {})
        plan_params["action"] = action
        return _folderguardian_dry_run_plan(tool, plan_params, memory, artifact_dir)

    try:
        folder_path = _folderguardian_resolve_user_folder(params.get("folder_path") or params.get("path"), create=False)
    except ValueError as exc:
        return f"Invalid FolderGuardian {action} request: {exc}"

    try:
        command = _folderguardian_bridge_command(bridge_executable)
    except ValueError as exc:
        metrics = {
            "folder_path": str(folder_path),
            "action": action,
            "bridge_executable": str(bridge_executable) if bridge_executable else None,
            "error_message": str(exc),
        }
        output = json.dumps(metrics, indent=2)
        artifact_path = _write_tool_artifact(
            artifact_dir,
            tool.name,
            capability_name,
            params,
            "failed",
            output,
            str(exc),
            metrics,
            None,
        )
        _record_tool_memory(memory, tool.name, capability_name, params, "failed", output, str(exc), artifact_path, metrics, "not_applicable")
        artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
        return f"FolderGuardian bridge executable is missing or unavailable: {exc}{artifact_line}"

    command = command + [action, "--folder", str(folder_path), "--json"]
    timeout_seconds = _int_or_none(params.get("timeout_seconds")) or 3600
    timeout_seconds = max(30, min(timeout_seconds, 24 * 60 * 60))
    try:
        result = subprocess.run(
            command,
            cwd=str(tool.root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
        error = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else f"Timed out after {timeout_seconds} seconds."
        metrics = {
            "status": "failed",
            "action": action,
            "folder_path": str(folder_path),
            "timeout_seconds": timeout_seconds,
            "error_message": error,
        }
        artifact_path = _write_tool_artifact(artifact_dir, tool.name, capability_name, params, "failed", output, error, metrics, command)
        _record_tool_memory(memory, tool.name, capability_name, params, "failed", output, error, artifact_path, metrics, "not_applicable")
        artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
        return f"folderguardian {action} live execution failed: timed out after {timeout_seconds} seconds.{artifact_line}"

    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()
    metrics = _parse_folderguardian_bridge_json(output)
    if metrics is None:
        metrics = {
            "status": "failed" if result.returncode else "unknown",
            "action": action,
            "folder_path": str(folder_path),
            "raw_stdout": output,
        }
    bridge_status = str(metrics.get("status") or "").lower()
    status = "success" if result.returncode == 0 and bridge_status in {"success", "partial"} else "failed"
    if bridge_status == "partial":
        status = "partial"
    artifact_path = _write_tool_artifact(
        artifact_dir,
        tool.name,
        capability_name,
        params,
        status,
        output,
        error,
        metrics,
        command,
    )
    evidence_status, bayesian_summary = _maybe_update_bayesian_evidence(tool.name, artifact_path, params, bayesian_adapter)
    _record_tool_memory(memory, tool.name, capability_name, params, status, output, error, artifact_path, metrics, evidence_status)
    return _format_folderguardian_crypto_result(action, folder_path, result.returncode, metrics, error, artifact_path) + bayesian_summary


def _folderguardian_bridge_command(bridge_executable):
    candidates = []
    if bridge_executable:
        configured = Path(bridge_executable)
        if not configured.is_absolute():
            configured = Path(__file__).resolve().parents[2] / configured
        candidates.append(configured)
        if configured.suffix.lower() == ".exe":
            candidates.append(configured.with_suffix(".dll"))

    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            repo_root / "FolderGuardianBridge" / "bin" / "Debug" / "net8.0-windows" / "FolderGuardianBridge.exe",
            repo_root / "FolderGuardianBridge" / "bin" / "Release" / "net8.0-windows" / "FolderGuardianBridge.exe",
            repo_root / "FolderGuardianBridge" / "bin" / "Debug" / "net8.0-windows" / "FolderGuardianBridge.dll",
            repo_root / "FolderGuardianBridge" / "bin" / "Release" / "net8.0-windows" / "FolderGuardianBridge.dll",
        ]
    )
    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists():
            continue
        if candidate.suffix.lower() == ".dll":
            return ["dotnet", str(candidate)]
        return [str(candidate)]
    wanted = str(candidates[0]) if candidates else "FolderGuardianBridge.exe"
    raise ValueError(f"{wanted}. Build it with dotnet build FolderGuardianBridge\\FolderGuardianBridge.csproj.")


def _parse_folderguardian_bridge_json(output):
    text = (output or "").strip().lstrip("\ufeff")
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _format_folderguardian_crypto_result(action, folder_path, returncode, metrics, error, artifact_path):
    bridge_status = str(metrics.get("status") or ("success" if returncode == 0 else "failed")).lower()
    summary = metrics.get("summary") if isinstance(metrics.get("summary"), dict) else {}
    processed = _int_or_none(summary.get("processed_count")) or 0
    skipped = _int_or_none(summary.get("skipped_count")) or 0
    errors = _int_or_none(summary.get("failed_count")) or 0
    total = _int_or_none(summary.get("total_count")) or processed + skipped + errors
    duration_ms = _int_or_none(summary.get("duration_ms"))
    duration_text = f" in {duration_ms / 1000:.2f}s" if duration_ms is not None else ""
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    key_location = metrics.get("key_location")
    key_line = f"\nKey location: {key_location}" if key_location else ""
    if bridge_status in {"success", "partial"} and returncode == 0:
        label = "success" if bridge_status == "success" else "partial"
        return (
            f"folderguardian {action} live execution {label}.\n"
            f"Folder: {folder_path}\n"
            f"Processed {processed}/{total}{duration_text}; skipped {skipped}; errors {errors}."
            f"{key_line}{artifact_line}"
        )

    message = metrics.get("error_message") or error or "No bridge error message was returned."
    return (
        f"folderguardian {action} live execution failed.\n"
        f"Folder: {folder_path}\n"
        f"Error: {message}{artifact_line}"
    )


def _folderguardian_check_logs(tool, params, memory=None, artifact_dir=None):
    max_lines = int(params.get("max_lines", 40) or 40)
    max_lines = max(1, min(max_lines, 500))
    try:
        folder_path = _folderguardian_resolve_user_folder(params.get("folder_path") or params.get("path"), create=False)
    except ValueError as exc:
        return f"Invalid FolderGuardian log request: {exc}"
    log_path = folder_path / "SecurityLog.txt"
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = lines[-max_lines:]
        status = "success"
        message = "\n".join(tail) if tail else "SecurityLog.txt is empty."
    else:
        tail = []
        status = "success"
        message = "No SecurityLog.txt exists for that folder yet."
    metrics = {
        "folder_path": str(folder_path),
        "log_path": str(log_path),
        "line_count": len(tail),
        "max_lines": max_lines,
        "log_exists": log_path.exists(),
        "tail": tail,
    }
    output = json.dumps(metrics, indent=2)
    artifact_path = _write_tool_artifact(artifact_dir, tool.name, "check_logs", params, status, output, "", metrics, None)
    _record_tool_memory(memory, tool.name, "check_logs", params, status, output, "", artifact_path, metrics, "not_applicable")
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    return f"folderguardian log check complete for {folder_path}.\n{message}{artifact_line}"


def _folderguardian_start_watch(tool, params, memory=None, artifact_dir=None):
    try:
        folder_path = _folderguardian_resolve_user_folder(params.get("folder_path") or params.get("path"), create=True)
    except ValueError as exc:
        return f"Invalid FolderGuardian watch request: {exc}"
    interval_seconds = float(params.get("interval_seconds", 1.0) or 1.0)
    interval_seconds = max(0.25, min(interval_seconds, 10.0))
    key = str(folder_path).lower()
    with _FOLDERGUARDIAN_WATCHERS_LOCK:
        existing = _FOLDERGUARDIAN_WATCHERS.get(key)
        if existing and existing["thread"].is_alive():
            return f"folderguardian watch is already active for {folder_path}."
        stop_event = threading.Event()
        thread = threading.Thread(
            target=_folderguardian_watch_loop,
            args=(folder_path, stop_event, interval_seconds),
            daemon=True,
        )
        _FOLDERGUARDIAN_WATCHERS[key] = {
            "folder_path": str(folder_path),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "stop_event": stop_event,
            "thread": thread,
            "interval_seconds": interval_seconds,
        }
        thread.start()
    metrics = {"folder_path": str(folder_path), "status": "active", "interval_seconds": interval_seconds}
    output = json.dumps(metrics, indent=2)
    artifact_path = _write_tool_artifact(artifact_dir, tool.name, "start_watch", params, "success", output, "", metrics, None)
    _record_tool_memory(memory, tool.name, "start_watch", params, "success", output, "", artifact_path, metrics, "not_applicable")
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    return f"folderguardian watch started for {folder_path}. Security events will be written to SecurityLog.txt.{artifact_line}"


def _folderguardian_stop_watch(tool, params, memory=None, artifact_dir=None):
    try:
        folder_path = _folderguardian_resolve_user_folder(params.get("folder_path") or params.get("path"), create=False)
    except ValueError as exc:
        return f"Invalid FolderGuardian stop-watch request: {exc}"
    key = str(folder_path).lower()
    stopped = False
    with _FOLDERGUARDIAN_WATCHERS_LOCK:
        existing = _FOLDERGUARDIAN_WATCHERS.get(key)
        if existing:
            existing["stop_event"].set()
            stopped = True
            _FOLDERGUARDIAN_WATCHERS.pop(key, None)
    if stopped:
        _folderguardian_write_log(folder_path, "Monitoring paused.")
    metrics = {"folder_path": str(folder_path), "status": "stopped" if stopped else "not_active"}
    output = json.dumps(metrics, indent=2)
    artifact_path = _write_tool_artifact(artifact_dir, tool.name, "stop_watch", params, "success", output, "", metrics, None)
    _record_tool_memory(memory, tool.name, "stop_watch", params, "success", output, "", artifact_path, metrics, "not_applicable")
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    if stopped:
        return f"folderguardian watch stopped for {folder_path}.{artifact_line}"
    return f"No active folderguardian watch was found for {folder_path}.{artifact_line}"


def _folderguardian_watch_status(tool, params, memory=None, artifact_dir=None):
    with _FOLDERGUARDIAN_WATCHERS_LOCK:
        sessions = [
            {
                "folder_path": value["folder_path"],
                "started_at": value["started_at"],
                "interval_seconds": value["interval_seconds"],
                "active": value["thread"].is_alive(),
            }
            for value in _FOLDERGUARDIAN_WATCHERS.values()
        ]
    metrics = {"active_watch_count": len([item for item in sessions if item["active"]]), "sessions": sessions}
    output = json.dumps(metrics, indent=2)
    artifact_path = _write_tool_artifact(artifact_dir, tool.name, "watch_status", params, "success", output, "", metrics, None)
    _record_tool_memory(memory, tool.name, "watch_status", params, "success", output, "", artifact_path, metrics, "not_applicable")
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    if not sessions:
        return f"No FolderGuardian watches are active.{artifact_line}"
    lines = ["Active FolderGuardian watches:"]
    for session in sessions:
        state = "active" if session["active"] else "stopped"
        lines.append(f"- {session['folder_path']}: {state}, started {session['started_at']}")
    lines.append(artifact_line.strip())
    return "\n".join(line for line in lines if line)


def _folderguardian_watch_loop(folder_path, stop_event, interval_seconds):
    previous = _folderguardian_snapshot(folder_path)
    _folderguardian_write_log(folder_path, "Monitoring enabled.")
    while not stop_event.wait(interval_seconds):
        current = _folderguardian_snapshot(folder_path)
        previous_paths = set(previous)
        current_paths = set(current)
        for path in sorted(current_paths - previous_paths):
            action = "New executable or script created" if Path(path).suffix.lower() in FOLDERGUARDIAN_SUSPICIOUS_EXTENSIONS else "Created"
            _folderguardian_write_log(folder_path, f"{action}: {path}")
        for path in sorted(previous_paths - current_paths):
            _folderguardian_write_log(folder_path, f"Deleted: {path}")
        for path in sorted(current_paths & previous_paths):
            if current[path] != previous[path]:
                _folderguardian_write_log(folder_path, f"Modified: {path}")
        previous = current


def _folderguardian_snapshot(folder_path):
    snapshot = {}
    for path in Path(folder_path).rglob("*"):
        if not path.is_file() or _folderguardian_ignore_log_event(path):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        snapshot[str(path)] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _folderguardian_ignore_log_event(path):
    name = Path(path).name
    return (
        name.lower() == "securitylog.txt"
        or name.lower().endswith(".enc")
        or name.lower().endswith(".tmp")
        or name.startswith("~$")
    )


def _folderguardian_write_log(folder_path, message):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}"
    try:
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        with (Path(folder_path) / "SecurityLog.txt").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def _folderguardian_security_review(tool, params, memory=None, artifact_dir=None):
    root = Path(tool.root)
    metrics = _folderguardian_project_metrics(root)
    findings = []
    if metrics["uses_wpf"]:
        findings.append("Desktop UI is WPF-based; runtime automation should go through a dedicated service or CLI, not UI clicking.")
    if "AES encryption" in metrics["detected_features"]:
        findings.append("Encryption capability is present; Hivemind routes live actions through FolderGuardianBridge and records artifacts.")
    if "DPAPI key protection" in metrics["detected_features"]:
        findings.append("DPAPI key protection appears present; machine/user scope needs to be reported before any restore/decrypt workflow.")
    if "FileSystemWatcher monitoring" in metrics["detected_features"]:
        findings.append("Monitoring appears present; a future sidecar should stream watch events instead of polling logs.")
    if not metrics["readme_present"]:
        findings.append("README is missing; add a tool contract document before enabling live actions.")
    recommendations = [
        "Keep the external FolderGuardian app independent; use FolderGuardianBridge for headless runtime calls.",
        "Add a manifest that declares allowed operations, required confirmations, and rollback expectations.",
        "Store every live action as a runtime artifact with target path, mode, timestamp, and result.",
    ]
    report = {
        "metrics": metrics,
        "findings": findings,
        "recommendations": recommendations,
        "live_actions_enabled": True,
    }
    output = json.dumps(report, indent=2)
    artifact_path = _write_tool_artifact(artifact_dir, tool.name, "security_review", params, "success", output, "", report, None)
    _record_tool_memory(memory, tool.name, "security_review", params, "success", output, "", artifact_path, report, "not_applicable")
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    return (
        "folderguardian security review complete.\n"
        + "\n".join(f"- {item}" for item in findings[:6])
        + "\nRecommendations:\n"
        + "\n".join(f"- {item}" for item in recommendations)
        + artifact_line
    )


def _folderguardian_dry_run_plan(tool, params, memory=None, artifact_dir=None):
    action = str(params.get("action") or "protect").strip().lower()
    folder_path = str(params.get("folder_path") or params.get("path") or "").strip()
    allowed_actions = {"protect", "encrypt", "decrypt", "watch", "monitor", "inspect"}
    if action not in allowed_actions:
        action = "protect"
    plan = {
        "requested_action": action,
        "folder_path": folder_path or None,
        "will_execute_live_action": False,
        "required_before_live_enablement": [
            "Live execution is available through FolderGuardianBridge.",
            "Use this dry-run plan when you want to inspect the target before mutating files.",
            "Every live action should keep artifact logging enabled.",
        ],
        "preflight_checks": [
            "Resolve target path and verify it exists.",
            "Reject system folders and paths outside the user-approved target.",
            "Check available disk space for backup/artifact output.",
            "Record FolderGuardian version and selected mode.",
        ],
    }
    output = json.dumps(plan, indent=2)
    artifact_path = _write_tool_artifact(artifact_dir, tool.name, "dry_run_protection_plan", params, "success", output, "", plan, None)
    _record_tool_memory(memory, tool.name, "dry_run_protection_plan", params, "success", output, "", artifact_path, plan, "not_applicable")
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    target = folder_path or "the selected folder"
    return (
        f"folderguardian action request prepared for {action} on {target}.\n"
        "No live encryption or decryption was started because this was a dry-run/plan request.\n"
        "Live execution note:\n"
        + "\n".join(f"- {item}" for item in plan["required_before_live_enablement"])
        + artifact_line
    )


def _folderguardian_project_metrics(root):
    root = Path(root)
    project_files = _project_files(root, {".csproj"})
    solution_files = _project_files(root, {".sln"})
    csharp_files = _project_files(root, {".cs"})
    xaml_files = _project_files(root, {".xaml"})
    csproj_metadata = [_parse_csproj_metadata(path) for path in project_files]
    target_frameworks = sorted(
        {
            framework
            for metadata in csproj_metadata
            for framework in metadata.get("target_frameworks", [])
            if framework
        }
    )
    package_references = sorted(
        {
            package
            for metadata in csproj_metadata
            for package in metadata.get("package_references", [])
            if package
        }
    )
    source_text = _folderguardian_source_excerpt(root, csharp_files)
    readme_text = _read_text(root / "README.md", max_chars=20000)
    detected_features = _detect_folderguardian_features(source_text, readme_text)
    return {
        "root": str(root),
        "solution_count": len(solution_files),
        "project_count": len(project_files),
        "csharp_file_count": len(csharp_files),
        "xaml_file_count": len(xaml_files),
        "core_file_count": len([path for path in csharp_files if "Core" in path.relative_to(root).parts]),
        "readme_present": (root / "README.md").exists(),
        "target_frameworks": target_frameworks,
        "uses_wpf": any(metadata.get("uses_wpf") for metadata in csproj_metadata),
        "uses_windows_forms": any(metadata.get("uses_windows_forms") for metadata in csproj_metadata),
        "package_references": package_references,
        "detected_features": detected_features,
        "blocked_actions": ["encrypt", "decrypt", "start_watchtower", "stop_watchtower"],
        "sample_files": [str(path.relative_to(root)) for path in (solution_files + project_files + csharp_files + xaml_files)[:20]],
    }


def _project_files(root, suffixes):
    root = Path(root)
    if not root.exists():
        return []
    ignored = {".git", ".vs", "bin", "obj"}
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        relative_parts = set(path.relative_to(root).parts[:-1])
        if relative_parts & ignored:
            continue
        files.append(path)
    return sorted(files)


def _parse_csproj_metadata(path):
    metadata = {
        "path": str(path),
        "target_frameworks": [],
        "uses_wpf": False,
        "uses_windows_forms": False,
        "package_references": [],
    }
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError):
        metadata["parse_error"] = True
        return metadata
    root = tree.getroot()
    for element in root.iter():
        tag = _xml_tag_name(element.tag)
        text = (element.text or "").strip()
        if tag == "TargetFramework" and text:
            metadata["target_frameworks"].append(text)
        elif tag == "TargetFrameworks" and text:
            metadata["target_frameworks"].extend(item.strip() for item in text.split(";") if item.strip())
        elif tag == "UseWPF":
            metadata["uses_wpf"] = text.lower() == "true"
        elif tag == "UseWindowsForms":
            metadata["uses_windows_forms"] = text.lower() == "true"
        elif tag == "PackageReference":
            package = element.attrib.get("Include") or element.attrib.get("Update")
            if package:
                metadata["package_references"].append(package)
    metadata["target_frameworks"] = sorted(set(metadata["target_frameworks"]))
    metadata["package_references"] = sorted(set(metadata["package_references"]))
    return metadata


def _xml_tag_name(tag):
    return tag.rsplit("}", 1)[-1]


def _folderguardian_source_excerpt(root, csharp_files):
    snippets = []
    interesting_names = {
        "EncryptionHelper.cs",
        "FolderEncryptor.cs",
        "FolderMonitor.cs",
        "MainWindow.xaml.cs",
    }
    for path in csharp_files:
        if path.name not in interesting_names:
            continue
        snippets.append(_read_text(path, max_chars=50000))
    return "\n".join(snippets)


def _read_text(path, max_chars=20000):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[:max_chars]
    except OSError:
        return ""


def _detect_folderguardian_features(source_text, readme_text):
    combined = f"{source_text}\n{readme_text}".lower()
    checks = [
        ("AES encryption", ["aes", "aes-256"]),
        ("DPAPI key protection", ["protecteddata", "dpapi"]),
        ("recursive folder encrypt/decrypt", ["encryptfolderasync", "decryptfolderasync", "recursive"]),
        ("filename/folder obfuscation", ["obfuscat"]),
        ("integrity validation", ["integrity", "hashingwritestream"]),
        ("FileSystemWatcher monitoring", ["filesystemwatcher"]),
        ("SecurityLog.txt logging", ["securitylog.txt"]),
        ("suspicious executable/script detection", [".ps1", ".bat", ".cmd", ".vbs", ".scr"]),
    ]
    return [label for label, terms in checks if any(term in combined for term in terms)]


def _run_workspace_capability(tool, capability_name, params, memory=None, artifact_dir=None):
    if capability_name in {"inspect", "discover_projects"}:
        return _inspect_engineering_workspace(tool, params, memory, artifact_dir)
    return f"Unknown workspace capability: {capability_name}"


def _inspect_engineering_workspace(tool, params, memory=None, artifact_dir=None):
    root = Path(tool.root)
    max_projects = int(params.get("max_projects", 20) or 20)
    max_projects = max(1, min(max_projects, 100))
    projects = _workspace_project_summaries(root, max_projects=max_projects)
    metrics = {
        "root": str(root),
        "project_count": len(projects),
        "projects": projects,
    }
    output = json.dumps(metrics, indent=2)
    artifact_path = _write_tool_artifact(artifact_dir, tool.name, "inspect", params, "success", output, "", metrics, None)
    _record_tool_memory(memory, tool.name, "inspect", params, "success", output, "", artifact_path, metrics, "not_applicable")
    artifact_line = f"\nArtifact: {artifact_path}" if artifact_path else ""
    lines = [f"{tool.name} inspect success. Found {len(projects)} candidate projects under {root}."]
    for project in projects[:10]:
        manifest_text = ", ".join(project["manifests"]) or "no manifest"
        lines.append(f"- {project['name']}: {manifest_text}")
    if len(projects) > 10:
        lines.append(f"- ...and {len(projects) - 10} more.")
    lines.append(artifact_line.strip())
    return "\n".join(line for line in lines if line)


def _workspace_project_summaries(root, max_projects=20):
    root = Path(root)
    if not root.exists():
        return []
    ignored = {".git", ".vs", ".venv", "venv", "__pycache__", "bin", "obj", "node_modules", "runtime_state"}
    manifest_names = {
        "README.md",
        "pyproject.toml",
        "package.json",
        "CMakeLists.txt",
    }
    manifest_suffixes = {".sln", ".csproj", ".vcxproj", ".fsproj", ".vbproj", ".proto"}
    candidates = []
    for child in sorted(root.iterdir()):
        if len(candidates) >= max_projects:
            break
        if not child.is_dir() or child.name in ignored:
            continue
        manifests = []
        try:
            for path in child.iterdir():
                if not path.is_file():
                    continue
                if path.name in manifest_names or path.suffix.lower() in manifest_suffixes:
                    manifests.append(path.name)
        except OSError:
            continue
        source_counts = {}
        for suffix in (".py", ".cs", ".cpp", ".c", ".h", ".hpp"):
            try:
                source_counts[suffix] = sum(1 for path in child.rglob(f"*{suffix}") if path.is_file())
            except OSError:
                source_counts[suffix] = 0
        if manifests or any(source_counts.values()):
            candidates.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "manifests": sorted(manifests),
                    "source_counts": {key: value for key, value in source_counts.items() if value},
                }
            )
    return candidates


def _yes_no(value):
    return "yes" if value else "no"


def _run_bayesian_capability(adapter, capability_name, params):
    tool_name = params.get("tool_name", "ofdmsim")
    if capability_name == "health":
        return adapter.health_text()
    if capability_name == "init_tool_confidence_model":
        paths = adapter.init_tool_confidence_model(tool_name)
        return f"Initialized Bayesian tool-confidence mission for {tool_name}.\nModel: {paths['model_path']}\nMission: {paths['mission_path']}"
    if capability_name == "assess_tool_confidence":
        assessment = adapter.assess_tool_confidence(tool_name)
        if assessment.probability is None:
            return assessment.report_text
        confidence_text = f"{assessment.confidence * 100:.1f}%" if assessment.confidence is not None else "unknown"
        interval_text = ""
        credible_interval = getattr(assessment, "credible_interval", None)
        if credible_interval and len(credible_interval) == 2:
            interval_text = (
                f", interval {credible_interval[0] * 100:.1f}-"
                f"{credible_interval[1] * 100:.1f}%"
            )
        return (
            f"Bayesian confidence for {tool_name}: {assessment.probability * 100:.1f}% "
            f"(confidence {confidence_text}{interval_text}).\n"
            "Interpretation: this is a tool-output reliability estimate based on stored evidence, not a guarantee that the engineering result is physically correct."
        )
    if capability_name == "tool_confidence_breakdown":
        breakdown = adapter.tool_confidence_breakdown(tool_name)
        if not breakdown:
            return f"No Bayesian factor breakdown is available for {tool_name} yet."
        lines = [f"Bayesian factor breakdown for {tool_name}:"]
        for item in breakdown:
            lines.append(
                f"- {item['factor']}: evidence {item['evidence_count']} "
                f"(supports {item['supports']}, against {item['against']}), "
                f"prior {item.get('prior_probability', 'unknown')}, confidence {item.get('confidence', 'unknown')}"
            )
        return "\n".join(lines)
    if capability_name == "add_tool_run_evidence":
        artifact_path = params.get("artifact_path")
        if not artifact_path:
            return "artifact_path is required for add_tool_run_evidence."
        result = adapter.add_tool_run_evidence(tool_name, artifact_path)
        return f"Added Bayesian evidence for {tool_name} from {artifact_path}.\n{result['assessment']}"
    if capability_name == "predict_scenario":
        model_path = params.get("model_path")
        if not model_path:
            return "model_path is required for predict_scenario."
        result = adapter.predict_scenario(model_path, params.get("scenario"), params.get("assignments", []))
        return json.dumps(result, indent=2)
    return f"Unknown Bayesian capability: {capability_name}"


def create_default_registry(settings):
    bayesian_adapter = None
    if hasattr(settings, "bayesian_engine_root") and hasattr(settings, "bayesian_python") and hasattr(settings, "bayesian_state_dir"):
        bayesian_adapter = BayesianAdapter(settings)
    registry = ToolRegistry(
        artifact_dir=getattr(settings, "tool_runs_dir", None),
        bayesian_adapter=bayesian_adapter,
        cadconverter_python=getattr(settings, "cadconverter_python", "python"),
        cadconverter_output_dir=getattr(settings, "cadconverter_output_dir", None),
        folderguardian_bridge_exe=getattr(settings, "folderguardian_bridge_exe", None),
        folderguardian_live_actions=getattr(settings, "folderguardian_live_actions", True),
    )
    registry.register(
        ExternalTool(
            name="ofdmsim",
            description="C++ OFDM baseband simulator for BER, channel, modulation, and equalizer experiments.",
            root=Path(settings.ofdm_root),
            runtime="C++ CLI",
            capabilities=[
                ToolCapability(
                    name="selftest",
                    description="Run built-in simulator self-tests.",
                    input_schema={},
                    safety_level="safe",
                    examples=["Run OFDMSim self-test"],
                ),
                ToolCapability(
                    name="run_simulation",
                    description="Run one parameterized OFDM simulation and capture BER output.",
                    input_schema={
                        "channel": "AWGN|Rayleigh|Ideal",
                        "modulation": "QPSK|16-QAM|64-QAM",
                        "equalizer": "MMSE|ZF",
                        "snr_db": "number",
                        "frames": "integer",
                        "seed": "integer",
                        "update_bayesian": "boolean",
                        "ber_threshold": "number",
                    },
                    examples=["Run QPSK AWGN MMSE at 18 dB for 100 frames"],
                ),
            ],
        )
    )
    registry.register(
        ExternalTool(
            name="ofdmsim_emi",
            description="OFDM simulator variant for EMI/EMC transient disturbance experiments.",
            root=Path(settings.ofdm_emi_root),
            runtime="C++ CLI",
            capabilities=[
                ToolCapability(name="selftest", description="Run built-in EMI simulator self-tests.", input_schema={}, safety_level="safe"),
                ToolCapability(
                    name="compare_emi",
                    description="Compare baseline, EMI, and clipping cases.",
                    input_schema={
                        "channel": "AWGN|Rayleigh|Ideal",
                        "modulation": "QPSK|16-QAM|64-QAM",
                        "equalizer": "MMSE|ZF",
                        "snr_db": "number",
                        "frames": "integer",
                        "seed": "integer",
                        "emi_probability": "number",
                        "emi_amplitude": "number",
                        "emi_width": "integer",
                        "clip_threshold": "number",
                    },
                    safety_level="safe",
                    examples=["Compare OFDMSim EMI AWGN QPSK at 18 dB for 100 frames EMI probability 0.003 amplitude 15"],
                ),
            ],
        )
    )
    registry.register(
        ExternalTool(
            name="cadconverter",
            description="Python CAD workbench for STEP/STP to GLB and GLB to STEP conversion.",
            root=Path(settings.cadconverter_root),
            runtime="Python CLI/workbench",
            capabilities=[
                ToolCapability(
                    name="inspect",
                    description="List available STEP/STP and GLB/GLTF files in the CADConverter workbench root.",
                    input_schema={},
                    safety_level="safe",
                    examples=["Inspect CADConverter files"],
                ),
                ToolCapability(
                    name="recommend_conversion",
                    description="Recommend a CAD conversion or validation workflow without executing it.",
                    input_schema={"input_path": "path", "goal": "string"},
                    safety_level="safe",
                    examples=['Recommend CADConverter workflow for "python half.stp" for AR viewing'],
                ),
                ToolCapability(
                    name="batch_validate_glb",
                    description="Validate multiple GLB/GLTF files from the CADConverter workbench.",
                    input_schema={"input_paths": "list", "max_files": "integer", "update_bayesian": "boolean"},
                    safety_level="safe",
                    examples=["Validate all CADConverter GLB files"],
                ),
                ToolCapability(
                    name="validate_glb",
                    description="Validate a GLB/GLTF file for meshes, materials, colors, normals, and named nodes.",
                    input_schema={"input_path": "path", "validate": "boolean", "update_bayesian": "boolean"},
                    safety_level="safe",
                    examples=['Validate CADConverter GLB "python half.glb"'],
                ),
                ToolCapability(
                    name="convert_step_to_glb",
                    description="Convert STEP/STP CAD files to hierarchy-preserving GLB.",
                    input_schema={
                        "input_path": "path",
                        "output_path": "path",
                        "validate": "boolean",
                        "single_sided": "boolean",
                        "linear_deflection": "number",
                        "angular_deflection": "number",
                        "update_bayesian": "boolean",
                    },
                    examples=['Convert STEP to GLB "python half.stp"'],
                ),
                ToolCapability(
                    name="convert_glb_to_step",
                    description="Convert GLB/GLTF meshes back to STEP using selected reconstruction mode.",
                    input_schema={
                        "input_path": "path",
                        "output_path": "path",
                        "mode": "faceted|reconstructed|advanced",
                        "unit_scale": "number",
                        "angle_tolerance": "number",
                        "plane_tolerance": "number",
                        "reconstruct_holes": "boolean",
                        "reconstruct_cylinders": "boolean",
                        "reconstruct_spheres": "boolean",
                        "reconstruct_cones": "boolean",
                        "make_solids": "boolean",
                        "no_sew": "boolean",
                        "update_bayesian": "boolean",
                    },
                    examples=['Convert GLB to STEP "python half.glb" mode reconstructed'],
                ),
            ],
        )
    )
    registry.register(
        ExternalTool(
            name="folderguardian",
            description="C# WPF folder encryption and monitoring tool.",
            root=Path(settings.folderguardian_root),
            runtime="C# WPF",
            capabilities=[
                ToolCapability(
                    name="inspect",
                    description="Inspect project metadata and security-oriented capabilities without running encryption, decryption, or monitoring.",
                    input_schema={},
                    safety_level="safe",
                    examples=["Inspect FolderGuardian project"],
                ),
                ToolCapability(
                    name="security_review",
                    description="Review FolderGuardian safety boundaries, detected security features, and integration risks without executing live actions.",
                    input_schema={},
                    safety_level="safe",
                    examples=["Review FolderGuardian security integration"],
                ),
                ToolCapability(
                    name="encrypt_folder",
                    description="Encrypt a selected folder through the FolderGuardianBridge headless sidecar and record a runtime artifact.",
                    input_schema={"folder_path": "path", "timeout_seconds": "integer", "update_bayesian": "boolean"},
                    safety_level="mutating",
                    supports_progress=False,
                    supports_cancel=False,
                    examples=['Encrypt FolderGuardian folder "D:\\Temp\\Demo"'],
                ),
                ToolCapability(
                    name="decrypt_folder",
                    description="Decrypt a selected folder through the FolderGuardianBridge headless sidecar and record a runtime artifact.",
                    input_schema={"folder_path": "path", "timeout_seconds": "integer", "update_bayesian": "boolean"},
                    safety_level="mutating",
                    supports_progress=False,
                    supports_cancel=False,
                    examples=['Decrypt FolderGuardian folder "D:\\Temp\\Demo"'],
                ),
                ToolCapability(
                    name="check_logs",
                    description="Read the tail of SecurityLog.txt for a selected protected folder.",
                    input_schema={"folder_path": "path", "max_lines": "integer"},
                    safety_level="safe",
                    examples=['Check FolderGuardian logs for "D:\\Temp\\Demo"'],
                ),
                ToolCapability(
                    name="start_watch",
                    description="Start a Hivemind-managed FolderGuardian-style watch that records events to SecurityLog.txt.",
                    input_schema={"folder_path": "path", "interval_seconds": "number"},
                    safety_level="safe",
                    examples=['Start FolderGuardian watch for "D:\\Temp\\Demo"'],
                ),
                ToolCapability(
                    name="stop_watch",
                    description="Stop a Hivemind-managed FolderGuardian watch for a selected folder.",
                    input_schema={"folder_path": "path"},
                    safety_level="safe",
                    examples=['Stop FolderGuardian watch for "D:\\Temp\\Demo"'],
                ),
                ToolCapability(
                    name="watch_status",
                    description="List active Hivemind-managed FolderGuardian watches.",
                    input_schema={},
                    safety_level="safe",
                    examples=["Show FolderGuardian watch status"],
                ),
                ToolCapability(
                    name="dry_run_protection_plan",
                    description="Prepare an encrypt/decrypt/protect action request without executing live cryptographic changes.",
                    input_schema={"folder_path": "path", "action": "protect|encrypt|decrypt|watch|monitor|inspect"},
                    safety_level="safe",
                    examples=['Prepare FolderGuardian encryption for "D:\\Temp\\Demo"'],
                ),
            ],
        )
    )
    registry.register(
        ExternalTool(
            name="security_suite",
            description="Local-only defensive file and folder risk assessment suite.",
            root=Path(__file__).resolve().parents[2],
            runtime="Python local scanner",
            capabilities=[
                ToolCapability(
                    name="assess_path",
                    description="Run a local defensive maliciousness/risk assessment for one file or folder and record evidence.",
                    input_schema={
                        "target_path": "path",
                        "max_files": "integer",
                        "include_defender_status": "boolean",
                        "update_bayesian": "boolean",
                    },
                    safety_level="safe",
                    examples=[
                        'Check if "D:\\Temp\\Demo" is malicious',
                        "Scan Downloads for suspicious files",
                    ],
                ),
                ToolCapability(
                    name="audit_project",
                    description="Run a local static project security audit for secrets, sensitive config files, and dependency manifests.",
                    input_schema={"target_path": "path", "max_files": "integer"},
                    safety_level="safe",
                    examples=['Audit project security for "D:\\Coding\\Hivemind"'],
                ),
                ToolCapability(
                    name="defender_scan",
                    description="Prepare or run a Microsoft Defender custom scan for a target path; execution requires confirm_scan=true.",
                    input_schema={"target_path": "path", "confirm_scan": "boolean", "timeout_seconds": "integer"},
                    safety_level="guarded",
                    examples=['Prepare Defender scan for "D:\\Temp\\Demo"'],
                ),
                ToolCapability(
                    name="quarantine_plan",
                    description="Prepare a defensive quarantine plan without moving, deleting, or modifying files.",
                    input_schema={"target_path": "path"},
                    safety_level="safe",
                    examples=['Prepare quarantine plan for "D:\\Temp\\invoice.pdf.exe"'],
                ),
                ToolCapability(
                    name="quarantine_path",
                    description="Move a target into local Hivemind quarantine storage; requires confirm_quarantine=true.",
                    input_schema={"target_path": "path", "confirm_quarantine": "boolean", "reason": "string"},
                    safety_level="guarded",
                    examples=['Quarantine "D:\\Temp\\invoice.pdf.exe" with confirm_quarantine true'],
                ),
                ToolCapability(
                    name="restore_quarantine",
                    description="Restore a quarantined target using its metadata; requires confirm_restore=true.",
                    input_schema={
                        "quarantine_id": "string",
                        "metadata_path": "path",
                        "artifact_path": "path",
                        "restore_path": "path",
                        "confirm_restore": "boolean",
                        "allow_overwrite": "boolean",
                    },
                    safety_level="guarded",
                    examples=["Restore quarantine by quarantine_id with confirm_restore true"],
                ),
                ToolCapability(
                    name="defender_history",
                    description="Read Microsoft Defender threat history locally when available.",
                    input_schema={"limit": "integer"},
                    safety_level="safe",
                    examples=["Show Defender threat history"],
                ),
                ToolCapability(
                    name="rule_scan",
                    description="Run built-in or JSON rule-pack text patterns over a file/folder without uploading data.",
                    input_schema={"target_path": "path", "rules_path": "path", "max_files": "integer"},
                    safety_level="safe",
                    examples=['Run local security rules on "D:\\Downloads"'],
                ),
                ToolCapability(
                    name="security_history",
                    description="Show recent local security assessments recorded by Jarvis.",
                    input_schema={"limit": "integer"},
                    safety_level="safe",
                    examples=["Show recent security scans"],
                ),
                ToolCapability(
                    name="explain_assessment",
                    description="Explain a recorded security assessment artifact in plain English.",
                    input_schema={"artifact_path": "path"},
                    safety_level="safe",
                    examples=["Explain the last security assessment"],
                ),
            ],
        )
    )
    registry.register(
        ExternalTool(
            name="cpp_workspace",
            description="Safe inspector for C++ engineering projects under the configured C++ workspace root.",
            root=Path(getattr(settings, "cpp_workspace_root", "external/cpp")),
            runtime="Workspace inspector",
            capabilities=[
                ToolCapability(
                    name="inspect",
                    description="Discover likely C++ projects, manifests, and source counts.",
                    input_schema={"max_projects": "integer"},
                    safety_level="safe",
                    examples=["Inspect C++ workspace projects"],
                )
            ],
        )
    )
    registry.register(
        ExternalTool(
            name="python_workspace",
            description="Safe inspector for Python engineering projects under the configured Python workspace root.",
            root=Path(getattr(settings, "python_workspace_root", "external/python")),
            runtime="Workspace inspector",
            capabilities=[
                ToolCapability(
                    name="inspect",
                    description="Discover likely Python projects, manifests, and source counts.",
                    input_schema={"max_projects": "integer"},
                    safety_level="safe",
                    examples=["Inspect Python workspace projects"],
                )
            ],
        )
    )
    registry.register(
        ExternalTool(
            name="csharp_workspace",
            description="Safe inspector for C# engineering projects under the configured C# workspace root.",
            root=Path(getattr(settings, "csharp_workspace_root", "external/csharp")),
            runtime="Workspace inspector",
            capabilities=[
                ToolCapability(
                    name="inspect",
                    description="Discover likely C# projects, manifests, and source counts.",
                    input_schema={"max_projects": "integer"},
                    safety_level="safe",
                    examples=["Inspect C# workspace projects"],
                )
            ],
        )
    )
    registry.register(
        ExternalTool(
            name="bayesian_engine",
            description="Bayesian evidence broker for tool confidence, scenario prediction, and mission assessment.",
            root=Path(getattr(settings, "bayesian_engine_root", "missing-bayesian-engine")),
            runtime="Python CLI evidence broker",
            capabilities=[
                ToolCapability(name="health", description="Check Bayesian Engine root, Python executable, and Hivemind state paths.", input_schema={}, safety_level="safe"),
                ToolCapability(name="init_tool_confidence_model", description="Create Hivemind-owned Bayesian model and mission files for a tool.", input_schema={"tool_name": "string"}, safety_level="safe"),
                ToolCapability(name="add_tool_run_evidence", description="Attach a tool-run artifact as Bayesian evidence.", input_schema={"tool_name": "string", "artifact_path": "path"}, safety_level="safe"),
                ToolCapability(name="assess_tool_confidence", description="Assess the current probability that a tool is producing reliable outputs.", input_schema={"tool_name": "string"}, safety_level="safe"),
                ToolCapability(name="tool_confidence_breakdown", description="Show Bayesian factor/evidence breakdown for a tool confidence mission.", input_schema={"tool_name": "string"}, safety_level="safe"),
                ToolCapability(name="predict_scenario", description="Run a Bayesian scenario prediction through an existing model.", input_schema={"model_path": "path", "scenario": "path|json", "assignments": "list"}, safety_level="safe"),
            ],
        )
    )
    return registry
