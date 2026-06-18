import json
import os
import subprocess
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional


DEFAULT_TOOL_CONFIDENCE_FACTORS = (
    ("Simulation executed successfully", 60),
    ("Output is machine-readable", 60),
    ("Simulation result meets expected quality", 50),
)

CAD_TOOL_CONFIDENCE_FACTORS = (
    ("CAD operation executed successfully", 60),
    ("CAD output is machine-readable", 60),
    ("CAD result has usable geometry or output artifact", 50),
)

FOLDERGUARDIAN_TOOL_CONFIDENCE_FACTORS = (
    ("FolderGuardian operation executed successfully", 60),
    ("FolderGuardian bridge output is machine-readable", 60),
    ("FolderGuardian operation completed without item errors", 50),
)

SECURITY_SUITE_CONFIDENCE_FACTORS = (
    ("Security assessment completed successfully", 60),
    ("Security assessment output is machine-readable", 60),
    ("Security assessment found no high-risk indicators", 50),
)

TOOL_CONFIDENCE_FACTORS = DEFAULT_TOOL_CONFIDENCE_FACTORS
VALID_EVIDENCE_STRENGTHS = {"weak", "medium", "strong"}


@dataclass
class BayesianAssessment:
    status: str
    probability: Optional[float]
    confidence: Optional[float]
    credible_interval: Optional[list]
    warnings: List[str]
    report_text: str


class BayesianAdapter:
    def __init__(self, settings, runner=None):
        self.root = Path(settings.bayesian_engine_root)
        self.python = Path(settings.bayesian_python)
        self.state_dir = Path(settings.bayesian_state_dir)
        self.ber_threshold = float(getattr(settings, "bayesian_ber_threshold", 0.01))
        self.dedupe_evidence = bool(getattr(settings, "bayesian_evidence_dedupe", True))
        self.max_evidence_per_factor = int(getattr(settings, "bayesian_max_evidence_per_factor", 12))
        self.success_strength = _valid_strength(getattr(settings, "bayesian_success_strength", "medium"))
        self.machine_readable_strength = _valid_strength(
            getattr(settings, "bayesian_machine_readable_strength", "medium")
        )
        self.quality_strength = _valid_strength(getattr(settings, "bayesian_quality_strength", "weak"))
        self.uses_custom_tool_confidence_factors = bool(
            getattr(settings, "bayesian_tool_confidence_factors_json", "")
        )
        self.tool_confidence_factors = _parse_tool_confidence_factors(
            getattr(settings, "bayesian_tool_confidence_factors_json", "")
        )
        self.runner = runner

    def health(self):
        fallback_python = Path(sys.executable)
        return {
            "available": self.root.exists() and (self.python.exists() or fallback_python.exists()),
            "root": str(self.root),
            "python": str(self.python),
            "fallback_python": str(fallback_python),
            "state_dir": str(self.state_dir),
            "ber_threshold": self.ber_threshold,
            "evidence_dedupe": self.dedupe_evidence,
            "max_evidence_per_factor": self.max_evidence_per_factor,
        }

    def health_text(self):
        health = self.health()
        state = "available" if health["available"] else "missing"
        return (
            f"bayesian_engine is {state}.\n"
            f"Root: {health['root']}\n"
            f"Python: {health['python']}\n"
            f"Fallback Python: {health['fallback_python']}\n"
            f"Hivemind state: {health['state_dir']}"
        )

    def init_tool_confidence_model(self, tool_name="ofdmsim"):
        self._ensure_available()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.tool_confidence_model_path()
        mission_path = self.tool_confidence_mission_path(tool_name)

        if not model_path.exists():
            self._run_cli(["--model", str(model_path), "init-graph-model", "--name", "hivemind_tool_confidence"])

        if not mission_path.exists():
            self._run_cli(
                [
                    "--model",
                    str(model_path),
                    "create-graph-mission",
                    str(mission_path),
                    "--title",
                    f"{tool_name} confidence",
                    "--success-criteria",
                    f"{tool_name} produces reliable, machine-readable outputs for Hivemind.",
                ]
            )
            for factor_name, probability in self._tool_confidence_factors_for(tool_name):
                self._run_cli(
                    [
                        "--model",
                        str(model_path),
                        "add-factor",
                        str(mission_path),
                        "--name",
                        factor_name,
                        "--probability",
                        str(probability),
                        "--confidence",
                        "low",
                    ]
                )
        else:
            self._ensure_missing_tool_confidence_factors(model_path, mission_path, tool_name)

        return {"model_path": str(model_path), "mission_path": str(mission_path)}

    def add_tool_run_evidence(self, tool_name, artifact_path, ber_threshold=None):
        if ber_threshold is None:
            ber_threshold = self.ber_threshold
        paths = self.init_tool_confidence_model(tool_name)
        mission_path = paths["mission_path"]
        evidence_dir = self.state_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        ledger = self._load_evidence_ledger()
        existing_counts = _mission_evidence_counts(mission_path)
        artifact_data = _read_json_file(artifact_path)
        results = []
        specs = evidence_specs_for_tool_run(
            tool_name,
            artifact_path,
            ber_threshold,
            success_strength=self.success_strength,
            machine_readable_strength=self.machine_readable_strength,
            quality_strength=self.quality_strength,
        )
        for spec in specs:
            evaluation = _evaluate_artifact_spec(artifact_data, spec)
            fingerprint = _evidence_fingerprint(tool_name, spec, artifact_data, evaluation)
            result = {
                "factor": spec["factor"],
                "check": spec["check"],
                "json_path": spec.get("json_path"),
                "strength": spec["strength"],
                "description": spec["description"],
                "passed": evaluation["passed"],
                "direction": "supports_occurrence" if evaluation["passed"] else "against_occurrence",
                "observed_value": evaluation.get("observed_value"),
                "error": evaluation.get("error"),
                "fingerprint": fingerprint,
            }
            if self.max_evidence_per_factor >= 0 and existing_counts.get(spec["factor"], 0) >= self.max_evidence_per_factor:
                result.update({"status": "skipped", "reason": "factor evidence cap reached"})
                results.append(result)
                continue
            if self.dedupe_evidence and fingerprint in ledger.get("fingerprints", []):
                result.update({"status": "skipped", "reason": "duplicate evidence"})
                results.append(result)
                continue
            args = [
                "--model",
                paths["model_path"],
                "add-local-artifact-evidence",
                mission_path,
                "--name",
                spec["factor"],
                "--file",
                str(artifact_path),
                "--check",
                spec["check"],
                "--description",
                spec["description"],
                "--strength",
                spec["strength"],
                "--snapshot-dir",
                str(evidence_dir),
            ]
            if spec.get("json_path"):
                args.extend(["--json-path", spec["json_path"]])
            if spec.get("equals") is not None:
                args.extend(["--equals", str(spec["equals"])])
            if spec.get("min_value") is not None:
                args.extend(["--min-value", str(spec["min_value"])])
            if spec.get("max_value") is not None:
                args.extend(["--max-value", str(spec["max_value"])])
            completed = self._run_cli(args)
            existing_counts[spec["factor"]] = existing_counts.get(spec["factor"], 0) + 1
            result.update({"status": "added", "returncode": completed.returncode})
            results.append(result)
            self._record_evidence_fingerprint(ledger, fingerprint, tool_name, spec, evaluation)
        assessment = self.assess_tool_confidence(tool_name)
        self._save_evidence_ledger(ledger)
        return {
            "evidence": results,
            "assessment": assessment.__dict__,
            "config": {
                "ber_threshold": ber_threshold,
                "dedupe": self.dedupe_evidence,
                "max_evidence_per_factor": self.max_evidence_per_factor,
            },
        }

    def assess_tool_confidence(self, tool_name="ofdmsim", no_record=True):
        paths = self.init_tool_confidence_model(tool_name)
        args = ["--model", paths["model_path"], "assess-mission", paths["mission_path"], "--format", "json"]
        if no_record:
            args.append("--no-record")
        completed = self._run_cli(args)
        return parse_assessment(completed.stdout)

    def tool_confidence_breakdown(self, tool_name="ofdmsim"):
        paths = self.init_tool_confidence_model(tool_name)
        return _mission_factor_breakdown(paths["mission_path"])

    def predict_scenario(self, model_path, scenario=None, assignments=None, output_format="json"):
        self._ensure_available()
        args = ["--model", str(model_path), "predict"]
        if scenario:
            args.append(str(scenario))
        for assignment in assignments or []:
            args.extend(["--set", assignment])
        args.extend(["--format", output_format])
        completed = self._run_cli(args)
        if output_format == "json":
            return json.loads(completed.stdout)
        return {"report_text": completed.stdout}

    def tool_confidence_model_path(self):
        return self.state_dir / "tool_confidence_model.json"

    def tool_confidence_mission_path(self, tool_name):
        return self.state_dir / f"{_slug(tool_name)}_confidence_mission.json"

    def evidence_ledger_path(self):
        return self.state_dir / "tool_confidence_evidence_ledger.json"

    def _tool_confidence_factors_for(self, tool_name):
        if self.uses_custom_tool_confidence_factors:
            return self.tool_confidence_factors
        normalized = str(tool_name or "").strip().lower()
        if normalized == "cadconverter":
            return CAD_TOOL_CONFIDENCE_FACTORS
        if normalized == "folderguardian":
            return FOLDERGUARDIAN_TOOL_CONFIDENCE_FACTORS
        if normalized == "security_suite":
            return SECURITY_SUITE_CONFIDENCE_FACTORS
        return DEFAULT_TOOL_CONFIDENCE_FACTORS

    def _ensure_missing_tool_confidence_factors(self, model_path, mission_path, tool_name):
        mission = _read_json_file(mission_path)
        existing_names = {
            factor.get("name")
            for factor in (mission.get("factors") or {}).values()
            if isinstance(factor, dict)
        }
        for factor_name, probability in self._tool_confidence_factors_for(tool_name):
            if factor_name in existing_names:
                continue
            self._run_cli(
                [
                    "--model",
                    str(model_path),
                    "add-factor",
                    str(mission_path),
                    "--name",
                    factor_name,
                    "--probability",
                    str(probability),
                    "--confidence",
                    "low",
                ]
            )

    def _load_evidence_ledger(self):
        path = self.evidence_ledger_path()
        if not path.exists():
            return {"fingerprints": [], "records": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"fingerprints": [], "records": []}
        data.setdefault("fingerprints", [])
        data.setdefault("records", [])
        return data

    def _save_evidence_ledger(self, ledger):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.evidence_ledger_path()
        path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    def _record_evidence_fingerprint(self, ledger, fingerprint, tool_name, spec, evaluation):
        if fingerprint not in ledger["fingerprints"]:
            ledger["fingerprints"].append(fingerprint)
        ledger["records"].append(
            {
                "fingerprint": fingerprint,
                "tool_name": tool_name,
                "factor": spec["factor"],
                "check": spec["check"],
                "json_path": spec.get("json_path"),
                "strength": spec["strength"],
                "passed": evaluation["passed"],
                "observed_value": evaluation.get("observed_value"),
            }
        )

    def _ensure_available(self):
        health = self.health()
        if not health["available"]:
            raise RuntimeError(self.health_text())

    def _run_cli(self, args, timeout=120):
        last_completed = None
        for python in self._python_candidates():
            command = [str(python), "-m", "engine.cli", *[str(arg) for arg in args]]
            if self.runner:
                completed = self.runner(command, cwd=str(self.root), timeout=timeout)
            else:
                env = os.environ.copy()
                env["PYTHONUTF8"] = "1"
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                completed = subprocess.run(
                    command,
                    cwd=str(self.root),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    shell=False,
                    env=env,
                )
            if completed.returncode == 0:
                return completed
            last_completed = completed
            if not _looks_like_launcher_failure(completed):
                break
        raise RuntimeError(
            f"Bayesian Engine command failed with code {last_completed.returncode}: "
            f"{last_completed.stderr or last_completed.stdout}"
        )

    def _python_candidates(self):
        candidates = []
        for value in (self.python, Path(sys.executable)):
            if value and value.exists() and value not in candidates:
                candidates.append(value)
        return candidates


def evidence_specs_for_tool_run(
    tool_name,
    artifact_path,
    ber_threshold=0.01,
    success_strength="medium",
    machine_readable_strength="medium",
    quality_strength="weak",
):
    artifact_data = _read_json_file(artifact_path)
    normalized_tool = str(tool_name or artifact_data.get("source_tool") or "").strip().lower()
    if normalized_tool == "cadconverter":
        return _cadconverter_evidence_specs(
            artifact_data,
            success_strength,
            machine_readable_strength,
            quality_strength,
        )
    if normalized_tool == "folderguardian":
        return _folderguardian_evidence_specs(
            artifact_data,
            success_strength,
            machine_readable_strength,
            quality_strength,
        )
    if normalized_tool == "security_suite":
        return _security_suite_evidence_specs(
            artifact_data,
            success_strength,
            machine_readable_strength,
            quality_strength,
        )
    if normalized_tool == "ofdmsim_emi":
        return _ofdmsim_emi_evidence_specs(
            artifact_data,
            ber_threshold,
            success_strength,
            machine_readable_strength,
            quality_strength,
        )
    return [
        {
            "factor": "Simulation executed successfully",
            "check": "json_equals",
            "json_path": "status",
            "equals": "success",
            "strength": _valid_strength(success_strength),
            "description": f"{tool_name} run completed successfully.",
        },
        {
            "factor": "Output is machine-readable",
            "check": "json_number",
            "json_path": "metrics.ber",
            "min_value": 0,
            "max_value": 1,
            "strength": _valid_strength(machine_readable_strength),
            "description": f"{tool_name} output contained parseable BER metrics.",
        },
        {
            "factor": "Simulation result meets expected quality",
            "check": "json_number",
            "json_path": "metrics.ber",
            "max_value": ber_threshold,
            "strength": _valid_strength(quality_strength),
            "description": f"{tool_name} BER was at or below {ber_threshold}.",
        },
    ]


def _cadconverter_evidence_specs(artifact_data, success_strength, machine_readable_strength, quality_strength):
    capability = str(artifact_data.get("capability") or "")
    machine_path = "metrics.meshes"
    quality_path = "metrics.primitives"
    quality_min = 1
    quality_description = "CAD validation reported usable mesh primitives."
    if capability == "batch_validate_glb":
        machine_path = "metrics.validated_count"
        quality_path = "metrics.failed_count"
        quality_min = None
        quality_description = "Batch validation completed without failed files."
    elif capability in {"convert_step_to_glb", "convert_glb_to_step"}:
        machine_path = "metrics.output_size_bytes"
        quality_path = "metrics.output_size_bytes"
        quality_description = "CAD conversion produced a non-empty output artifact."
    elif capability == "inspect":
        machine_path = "metrics.step_count"
        quality_path = "metrics.glb_count"
        quality_min = 0
        quality_description = "CADConverter inspection returned file inventory counts."

    quality_spec = {
        "factor": "CAD result has usable geometry or output artifact",
        "check": "json_number",
        "json_path": quality_path,
        "strength": _valid_strength(quality_strength),
        "description": quality_description,
    }
    if capability == "batch_validate_glb":
        quality_spec["max_value"] = 0
    else:
        quality_spec["min_value"] = quality_min if quality_min is not None else 1

    return [
        {
            "factor": "CAD operation executed successfully",
            "check": "json_equals",
            "json_path": "status",
            "equals": "success",
            "strength": _valid_strength(success_strength),
            "description": "CADConverter command completed successfully.",
        },
        {
            "factor": "CAD output is machine-readable",
            "check": "json_number",
            "json_path": machine_path,
            "min_value": 0,
            "strength": _valid_strength(machine_readable_strength),
            "description": f"CADConverter artifact contained parseable numeric metrics at {machine_path}.",
        },
        quality_spec,
    ]


def _folderguardian_evidence_specs(artifact_data, success_strength, machine_readable_strength, quality_strength):
    return [
        {
            "factor": "FolderGuardian operation executed successfully",
            "check": "json_equals",
            "json_path": "status",
            "equals": "success",
            "strength": _valid_strength(success_strength),
            "description": "FolderGuardian live bridge command completed successfully.",
        },
        {
            "factor": "FolderGuardian bridge output is machine-readable",
            "check": "json_number",
            "json_path": "metrics.summary.total_count",
            "min_value": 0,
            "strength": _valid_strength(machine_readable_strength),
            "description": "FolderGuardian artifact contained parseable summary counts.",
        },
        {
            "factor": "FolderGuardian operation completed without item errors",
            "check": "json_number",
            "json_path": "metrics.summary.failed_count",
            "max_value": 0,
            "strength": _valid_strength(quality_strength),
            "description": "FolderGuardian reported zero item-level errors.",
        },
    ]


def _security_suite_evidence_specs(artifact_data, success_strength, machine_readable_strength, quality_strength):
    capability = str(artifact_data.get("capability") or "")
    risk_path = "metrics.risk_score"
    scanned_path = "metrics.summary.scanned_file_count"
    if capability == "audit_project":
        scanned_path = "metrics.scanned_text_file_count"
    return [
        {
            "factor": "Security assessment completed successfully",
            "check": "json_equals",
            "json_path": "status",
            "equals": "success",
            "strength": _valid_strength(success_strength),
            "description": "Security Suite assessment completed successfully.",
        },
        {
            "factor": "Security assessment output is machine-readable",
            "check": "json_number",
            "json_path": scanned_path,
            "min_value": 0,
            "strength": _valid_strength(machine_readable_strength),
            "description": f"Security Suite artifact contained parseable scan coverage at {scanned_path}.",
        },
        {
            "factor": "Security assessment found no high-risk indicators",
            "check": "json_number",
            "json_path": risk_path,
            "max_value": 39,
            "strength": _valid_strength(quality_strength),
            "description": "Security Suite local risk score stayed below the high-risk threshold.",
        },
    ]


def _ofdmsim_emi_evidence_specs(artifact_data, ber_threshold, success_strength, machine_readable_strength, quality_strength):
    return [
        {
            "factor": "Simulation executed successfully",
            "check": "json_equals",
            "json_path": "status",
            "equals": "success",
            "strength": _valid_strength(success_strength),
            "description": "OFDMSim EMI comparison completed successfully.",
        },
        {
            "factor": "Output is machine-readable",
            "check": "json_number",
            "json_path": "metrics.baseline_ber",
            "min_value": 0,
            "max_value": 1,
            "strength": _valid_strength(machine_readable_strength),
            "description": "OFDMSim EMI output contained parseable baseline BER.",
        },
        {
            "factor": "Simulation result meets expected quality",
            "check": "json_number",
            "json_path": "metrics.baseline_ber",
            "max_value": ber_threshold,
            "strength": _valid_strength(quality_strength),
            "description": f"OFDMSim EMI baseline BER was at or below {ber_threshold}.",
        },
    ]


def parse_assessment(stdout):
    data = json.loads(stdout)
    prediction = data.get("assessment", {}).get("prediction", {})
    return BayesianAssessment(
        status="success",
        probability=prediction.get("probability"),
        confidence=prediction.get("confidence"),
        credible_interval=prediction.get("credible_interval"),
        warnings=prediction.get("warnings") or [],
        report_text=stdout,
    )


def _slug(value):
    return "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_") or "tool"


def _valid_strength(value):
    text = str(value or "weak").strip().lower()
    return text if text in VALID_EVIDENCE_STRENGTHS else "weak"


def _parse_tool_confidence_factors(value):
    if not value:
        return DEFAULT_TOOL_CONFIDENCE_FACTORS
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return DEFAULT_TOOL_CONFIDENCE_FACTORS
    factors = []
    if isinstance(raw, dict):
        raw = raw.get("factors", [])
    if not isinstance(raw, list):
        return DEFAULT_TOOL_CONFIDENCE_FACTORS
    for item in raw:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            probability = item.get("probability")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            name = str(item[0]).strip()
            probability = item[1]
        else:
            continue
        if not name:
            continue
        try:
            probability = float(probability)
        except (TypeError, ValueError):
            continue
        if 0 <= probability <= 1:
            probability *= 100
        if 0 <= probability <= 100:
            factors.append((name, probability))
    return tuple(factors) or DEFAULT_TOOL_CONFIDENCE_FACTORS


def _read_json_file(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _mission_evidence_counts(mission_path):
    mission = _read_json_file(mission_path)
    counts = {}
    for factor in (mission.get("factors") or {}).values():
        if not isinstance(factor, dict):
            continue
        name = factor.get("name")
        evidence_count = sum(
            1
            for item in factor.get("evidence", [])
            if isinstance(item, dict) and item.get("type") == "bayesian_evidence"
        )
        if name:
            counts[name] = evidence_count
    return counts


def _mission_factor_breakdown(mission_path):
    mission = _read_json_file(mission_path)
    breakdown = []
    for factor in (mission.get("factors") or {}).values():
        if not isinstance(factor, dict):
            continue
        evidence_items = [
            item
            for item in factor.get("evidence", [])
            if isinstance(item, dict) and item.get("type") == "bayesian_evidence"
        ]
        supports = 0
        against = 0
        for item in evidence_items:
            direction = str(item.get("direction") or "").lower()
            if "against" in direction:
                against += 1
            elif "support" in direction:
                supports += 1
        breakdown.append(
            {
                "factor": factor.get("name", "factor"),
                "prior_probability": factor.get("probability"),
                "confidence": factor.get("confidence"),
                "evidence_count": len(evidence_items),
                "supports": supports,
                "against": against,
            }
        )
    return breakdown


def _evaluate_artifact_spec(artifact_data, spec):
    try:
        observed = _json_lookup(artifact_data, spec.get("json_path"))
        if spec["check"] == "json_equals":
            passed = str(observed) == str(spec.get("equals"))
        elif spec["check"] == "json_number":
            observed = float(observed)
            passed = True
            if spec.get("min_value") is not None:
                passed = passed and observed >= float(spec["min_value"])
            if spec.get("max_value") is not None:
                passed = passed and observed <= float(spec["max_value"])
        else:
            passed = False
        return {"passed": bool(passed), "observed_value": observed, "error": None}
    except Exception as exc:
        return {"passed": False, "observed_value": None, "error": str(exc)}


def _json_lookup(data, path):
    value = data
    for part in str(path or "").split("."):
        if not part:
            continue
        if isinstance(value, dict):
            value = value[part]
        elif isinstance(value, list):
            value = value[int(part)]
        else:
            raise ValueError(f"Cannot descend into non-container value at '{part}'")
    return value


def _evidence_fingerprint(tool_name, spec, artifact_data, evaluation):
    payload = {
        "tool_name": tool_name,
        "capability": artifact_data.get("capability"),
        "params": artifact_data.get("params") or {},
        "factor": spec["factor"],
        "check": spec["check"],
        "json_path": spec.get("json_path"),
        "equals": spec.get("equals"),
        "min_value": spec.get("min_value"),
        "max_value": spec.get("max_value"),
        "strength": spec.get("strength"),
        "passed": evaluation.get("passed"),
        "observed_value": _stable_observed_value(evaluation.get("observed_value")),
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _stable_observed_value(value):
    if isinstance(value, float):
        return round(value, 12)
    return value


def _looks_like_launcher_failure(completed):
    text = f"{completed.stderr or ''}\n{completed.stdout or ''}".lower()
    return any(
        marker in text
        for marker in (
            "access is denied",
            "unable to create process",
            "the system cannot find the file",
            "no such file or directory",
        )
    )
