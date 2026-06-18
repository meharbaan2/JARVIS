import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PYTHON_SERVER = Path(__file__).resolve().parents[1] / "AICoreClient" / "PythonServer"
sys.path.insert(0, str(PYTHON_SERVER))

from bayesian_adapter import BayesianAdapter, evidence_specs_for_tool_run, parse_assessment, _mission_factor_breakdown
from runtime_artifacts import write_tool_run_artifact


class BayesianAdapterTests(unittest.TestCase):
    def _settings(self, tmp, root_exists=True, python_exists=True):
        root = Path(tmp) / "Bayesian Engine"
        python = root / ".venv" / "Scripts" / "python.exe"
        if root_exists:
            root.mkdir(parents=True)
        if python_exists:
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("", encoding="utf-8")
        return SimpleNamespace(
            bayesian_engine_root=root,
            bayesian_python=python,
            bayesian_state_dir=Path(tmp) / "hivemind_state" / "bayesian",
            bayesian_ber_threshold=0.01,
            bayesian_evidence_dedupe=True,
            bayesian_max_evidence_per_factor=12,
            bayesian_success_strength="medium",
            bayesian_machine_readable_strength="medium",
            bayesian_quality_strength="weak",
            bayesian_tool_confidence_factors_json="",
        )

    def test_health_reports_available_and_missing_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            available = BayesianAdapter(self._settings(tmp)).health()
            missing = BayesianAdapter(self._settings(Path(tmp) / "missing", root_exists=False, python_exists=False)).health()

        self.assertTrue(available["available"])
        self.assertFalse(missing["available"])

    def test_tool_run_artifact_writes_machine_readable_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_tool_run_artifact(
                Path(tmp),
                "ofdmsim",
                "run_simulation",
                {"snr_db": 18},
                "success",
                output_text="0.00000000 0 1120 18.00000000 14.98970004",
                metrics={"ber": 0.0, "errors": 0, "bits": 1120},
            )
            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["source_tool"], "ofdmsim")
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["metrics"]["ber"], 0.0)

    def test_evidence_specs_map_ofdm_artifacts_to_factors(self):
        specs = evidence_specs_for_tool_run("ofdmsim", "artifact.json", ber_threshold=0.02)

        self.assertEqual(specs[0]["factor"], "Simulation executed successfully")
        self.assertEqual(specs[0]["json_path"], "status")
        self.assertEqual(specs[1]["json_path"], "metrics.ber")
        self.assertEqual(specs[2]["max_value"], 0.02)

    def test_evidence_specs_map_cadconverter_artifacts_to_cad_factors(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "cad.json"
            artifact.write_text(
                json.dumps(
                    {
                        "source_tool": "cadconverter",
                        "capability": "validate_glb",
                        "status": "success",
                        "metrics": {"meshes": 2, "primitives": 4},
                    }
                ),
                encoding="utf-8",
            )

            specs = evidence_specs_for_tool_run("cadconverter", artifact)

        self.assertEqual(specs[0]["factor"], "CAD operation executed successfully")
        self.assertEqual(specs[1]["json_path"], "metrics.meshes")
        self.assertEqual(specs[2]["json_path"], "metrics.primitives")

    def test_evidence_specs_map_folderguardian_artifacts_to_bridge_factors(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "folderguardian.json"
            artifact.write_text(
                json.dumps(
                    {
                        "source_tool": "folderguardian",
                        "capability": "encrypt_folder",
                        "status": "success",
                        "metrics": {"summary": {"total_count": 1, "failed_count": 0}},
                    }
                ),
                encoding="utf-8",
            )

            specs = evidence_specs_for_tool_run("folderguardian", artifact)

        self.assertEqual(specs[0]["factor"], "FolderGuardian operation executed successfully")
        self.assertEqual(specs[1]["json_path"], "metrics.summary.total_count")
        self.assertEqual(specs[2]["json_path"], "metrics.summary.failed_count")

    def test_evidence_specs_map_security_suite_artifacts_to_risk_factors(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "security.json"
            artifact.write_text(
                json.dumps(
                    {
                        "source_tool": "security_suite",
                        "capability": "assess_path",
                        "status": "success",
                        "metrics": {"risk_score": 12, "summary": {"scanned_file_count": 3}},
                    }
                ),
                encoding="utf-8",
            )

            specs = evidence_specs_for_tool_run("security_suite", artifact)

        self.assertEqual(specs[0]["factor"], "Security assessment completed successfully")
        self.assertEqual(specs[1]["json_path"], "metrics.summary.scanned_file_count")
        self.assertEqual(specs[2]["json_path"], "metrics.risk_score")
        self.assertEqual(specs[2]["max_value"], 39)

    def test_parse_assessment_extracts_probability_fields(self):
        assessment = parse_assessment(
            json.dumps(
                {
                    "assessment": {
                        "prediction": {
                            "probability": 0.72,
                            "confidence": 0.44,
                            "credible_interval": [0.3, 0.9],
                            "warnings": ["rough estimate"],
                        }
                    }
                }
            )
        )

        self.assertEqual(assessment.probability, 0.72)
        self.assertEqual(assessment.confidence, 0.44)
        self.assertEqual(assessment.credible_interval, [0.3, 0.9])
        self.assertEqual(assessment.warnings, ["rough estimate"])

    def test_mission_factor_breakdown_counts_supporting_and_against_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            mission = Path(tmp) / "mission.json"
            mission.write_text(
                json.dumps(
                    {
                        "factors": {
                            "factor-1": {
                                "name": "Security assessment found no high-risk indicators",
                                "probability": 50,
                                "confidence": "low",
                                "evidence": [
                                    {"type": "bayesian_evidence", "direction": "supports_occurrence"},
                                    {"type": "bayesian_evidence", "direction": "against_occurrence"},
                                    {"type": "note", "direction": "supports_occurrence"},
                                ],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            breakdown = _mission_factor_breakdown(mission)

        self.assertEqual(breakdown[0]["evidence_count"], 2)
        self.assertEqual(breakdown[0]["supports"], 1)
        self.assertEqual(breakdown[0]["against"], 1)

    def test_add_tool_run_evidence_builds_bayesian_cli_commands(self):
        commands = []

        def runner(command, cwd, timeout):
            commands.append(command)
            if "assess-mission" in command:
                stdout = json.dumps({"assessment": {"prediction": {"probability": 0.8, "confidence": 0.5}}})
            else:
                stdout = "{}"
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(tmp)
            artifact = Path(tmp) / "artifact.json"
            artifact.write_text('{"status":"success","metrics":{"ber":0.0}}', encoding="utf-8")
            adapter = BayesianAdapter(settings, runner=runner)

            result = adapter.add_tool_run_evidence("ofdmsim", artifact)

        flattened = [" ".join(command) for command in commands]
        self.assertTrue(any("init-graph-model" in command for command in flattened))
        self.assertTrue(any("create-graph-mission" in command for command in flattened))
        self.assertEqual(sum("add-local-artifact-evidence" in command for command in flattened), 3)
        self.assertEqual(result["assessment"]["probability"], 0.8)
        self.assertEqual(result["evidence"][0]["status"], "added")
        self.assertTrue(result["evidence"][0]["passed"])

    def test_duplicate_tool_run_evidence_is_skipped(self):
        commands = []

        def runner(command, cwd, timeout):
            commands.append(command)
            if "assess-mission" in command:
                stdout = json.dumps({"assessment": {"prediction": {"probability": 0.8, "confidence": 0.5}}})
            else:
                stdout = "{}"
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(tmp)
            artifact = Path(tmp) / "artifact.json"
            artifact.write_text(
                json.dumps(
                    {
                        "capability": "run_simulation",
                        "params": {"snr_db": 20, "frames": 100},
                        "status": "success",
                        "metrics": {"ber": 0.0},
                    }
                ),
                encoding="utf-8",
            )
            adapter = BayesianAdapter(settings, runner=runner)

            first = adapter.add_tool_run_evidence("ofdmsim", artifact)
            second = adapter.add_tool_run_evidence("ofdmsim", artifact)

        flattened = [" ".join(command) for command in commands]
        self.assertEqual(sum("add-local-artifact-evidence" in command for command in flattened), 3)
        self.assertEqual([item["status"] for item in first["evidence"]], ["added", "added", "added"])
        self.assertEqual([item["status"] for item in second["evidence"]], ["skipped", "skipped", "skipped"])
        self.assertIn("duplicate evidence", second["evidence"][0]["reason"])

    def test_custom_bayesian_config_is_used_for_evidence_specs(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(tmp)
            settings.bayesian_ber_threshold = 0.02
            settings.bayesian_success_strength = "strong"
            settings.bayesian_machine_readable_strength = "medium"
            settings.bayesian_quality_strength = "strong"
            adapter = BayesianAdapter(settings, runner=lambda command, cwd, timeout: subprocess.CompletedProcess(command, 0, stdout="{}", stderr=""))

        specs = evidence_specs_for_tool_run(
            "ofdmsim",
            "artifact.json",
            adapter.ber_threshold,
            success_strength=adapter.success_strength,
            machine_readable_strength=adapter.machine_readable_strength,
            quality_strength=adapter.quality_strength,
        )

        self.assertEqual(specs[0]["strength"], "strong")
        self.assertEqual(specs[2]["strength"], "strong")
        self.assertEqual(specs[2]["max_value"], 0.02)

    def test_cli_falls_back_to_current_python_on_launcher_failure(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            if len(calls) == 1:
                return subprocess.CompletedProcess(command, 101, stdout="", stderr="Access is denied.")
            return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(tmp)
            adapter = BayesianAdapter(settings)
            with patch("bayesian_adapter.subprocess.run", side_effect=fake_run):
                completed = adapter._run_cli(["summary"])

        self.assertEqual(completed.returncode, 0)
        self.assertEqual(calls[0][0][0], str(settings.bayesian_python))
        self.assertEqual(calls[1][0][0], sys.executable)
        self.assertEqual(calls[0][1]["env"]["PYTHONDONTWRITEBYTECODE"], "1")


if __name__ == "__main__":
    unittest.main()
