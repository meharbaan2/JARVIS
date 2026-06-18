import sys
import tempfile
import unittest
import sqlite3
import json
import zipfile
from pathlib import Path
from unittest.mock import patch
import subprocess
from types import SimpleNamespace


PYTHON_SERVER = Path(__file__).resolve().parents[1] / "AICoreClient" / "PythonServer"
sys.path.insert(0, str(PYTHON_SERVER))

from memory_service import MemoryService
from orchestrator import RuntimeOrchestrator
from runtime_service import RuntimeServiceFacade
from model_runtime_service import RuntimeModelServiceFacade
from embedding_service import HashEmbeddingProvider, create_embedding_provider
from model_registry import create_default_model_registry
from intent_router import IntentRouter
from math_service import MathService
from punjabi_service import PunjabiConversationService, contains_gurmukhi
from tool_registry import (
    _format_ofdm_quiet_output,
    _format_ofdm_params,
    _normalize_ofdm_params,
    _parse_cadconverter_metrics,
    _parse_emi_compare_metrics,
    _parse_ofdm_quiet_metrics,
    create_default_registry,
)
from voice_utils import normalize_for_tts, split_into_speech_chunks

try:
    import runtime_contract_pb2
    from runtime_grpc_services import RuntimeGrpcService, RuntimeModelGrpcService, VoiceGrpcService
except ModuleNotFoundError:
    runtime_contract_pb2 = None
    RuntimeGrpcService = None
    RuntimeModelGrpcService = None
    VoiceGrpcService = None
from speech_style import (
    casual_jarvis_reply,
    clean_model_disclaimers,
    dynamic_speech_params,
    format_jarvis_response,
    prepare_spoken_response_text,
    process_english_speech,
)


class RuntimeFoundationTests(unittest.TestCase):
    def test_tts_chunking_keeps_decimals_together(self):
        chunks = split_into_speech_chunks("Temperature is 28.4°C. Wind is 1.2 m/s.")

        self.assertEqual(chunks[0], "Temperature is 28 point 4 degrees Celsius.")
        self.assertEqual(chunks[1], "Wind is 1 point 2 meters per second.")

    def test_tts_shortens_machine_precision_decimals(self):
        spoken = normalize_for_tts("BER 0.00000000, Es/N0 20.00000000 dB, Eb/N0 16.98970004 dB, confidence 94.4%.")

        self.assertIn("BER 0,", spoken)
        self.assertIn("Es/N0 20 dB", spoken)
        self.assertIn("SNR 20 dB", normalize_for_tts("SNR 20.0 dB"))
        self.assertIn("Eb/N0 16 point 99 dB", spoken)
        self.assertIn("94 point 4 percent", spoken)

    def test_jarvis_style_avoids_mechanical_prefixes_and_sir(self):
        response = format_jarvis_response("According to my analysis, the OFDM run succeeded, sir", "ToolRegistry")

        self.assertNotIn("According to my analysis", response)
        self.assertNotRegex(response.lower(), r"\bsir\b")
        self.assertTrue(response.endswith("."))

    def test_jarvis_style_does_not_prefix_casual_model_replies(self):
        response = format_jarvis_response("Hey, I'm online and ready.", "Phi-3")

        self.assertEqual(response, "Hey, I'm online and ready.")
        self.assertNotIn("important bit", response.lower())

    def test_casual_jarvis_reply_handles_whats_up_without_phi(self):
        response = casual_jarvis_reply("whats up")

        self.assertIsNotNone(response)
        self.assertNotRegex(response.lower(), r"\btext-based\b|\bdigital assistant\b")

    def test_model_disclaimer_cleanup_removes_text_only_claims(self):
        response = clean_model_disclaimers(
            "I'm just a digital assistant, but if you need help, feel free to ask. How can I assist you today?"
        )

        self.assertNotRegex(response.lower(), r"\bdigital assistant\b|\btext-based\b")
        self.assertIn("What do you need?", response)

    def test_jarvis_style_preserves_runtime_names(self):
        response = format_jarvis_response("ofdmsim simulation success. ber 0.", "ToolRegistry")

        self.assertIn("OFDMSim", response)
        self.assertIn("BER", response)

    def test_jarvis_style_does_not_treat_zero_errors_as_failure(self):
        response = format_jarvis_response(
            "OFDMSim simulation success. BER 0.00000000, errors 0/11200.",
            "ToolRegistry",
        )

        self.assertNotIn("snag", response.lower())
        self.assertNotIn("constraint", response.lower())

    def test_spoken_response_omits_artifacts_paths_and_links(self):
        spoken = prepare_spoken_response_text(
            "Done. See https://example.com/report. Artifact: D:\\Coding\\Hivemind\\AICoreClient\\runtime_state\\tool_runs\\run.json Bayesian evidence updated."
        )

        self.assertNotIn("https://example.com", spoken)
        self.assertNotIn("D:\\Coding", spoken)
        self.assertNotIn("Artifact:", spoken)
        self.assertIn("left the link in the text", spoken)
        self.assertIn("Bayesian evidence updated", spoken)

    def test_english_speech_style_cleans_contractions_and_name(self):
        response = process_english_speech("Yes sir, im checking J.A.R.V.I.S now")

        self.assertNotRegex(response.lower(), r"\bsir\b")
        self.assertIn("I'm checking Jarvis now", response)
        self.assertNotIn("100000 .", process_english_speech("frames must be between 1 and 100000."))

    def test_dynamic_speech_params_shift_for_alerts(self):
        normal = dynamic_speech_params("The simulation completed successfully.")
        alert = dynamic_speech_params("Warning: the run failed.")

        self.assertLess(alert["rate"], normal["rate"])
        self.assertGreaterEqual(alert["volume"], normal["volume"])

    def test_punjabi_local_greeting_is_fast_and_gurmukhi(self):
        response = PunjabiConversationService().handle("ਕੀ ਹਾਲ ਹੈ ਠੀਕ ਹੋ")

        self.assertIn("ਮੈਂ ਠੀਕ ਹਾਂ", response)
        self.assertTrue(contains_gurmukhi(response))
        self.assertNotIn("Reply in Punjabi", response)

    def test_memory_indexes_and_retrieves_project_docs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                memory.index_document("ofdmsim", "README.md", "OFDM simulator supports QPSK BER sweeps.")

                hits = memory.search("QPSK BER")

        self.assertEqual(hits[0]["project"], "ofdmsim")
        self.assertIn("QPSK", hits[0]["snippet"])

    def test_memory_chunks_documents_for_retrieval(self):
        content = "\n\n".join(
            [
                "Forward conversion exports STEP files into GLB assets for AR viewing.",
                "Reverse conversion can use faceted fallback when reconstruction quality is risky.",
                "Workbench diagnostics show mesh counts, material counts, and named nodes.",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                memory.index_document("cadconverter", "README.md", content)

                hits = memory.search("faceted reconstruction quality", limit=1)
                stats = memory.stats()

        self.assertEqual(hits[0]["project"], "cadconverter")
        self.assertIn("faceted fallback", hits[0]["snippet"])
        self.assertGreaterEqual(stats["project_document_chunk_count"], 1)
        self.assertEqual(stats["embedding_dimensions"], 96)

    def test_memory_search_can_filter_tool_run_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                memory.index_document("ofdmsim", "README.md", "OFDM simulator documentation.")
                memory.record_tool_run(
                    "ofdmsim",
                    "run_simulation",
                    '{"snr_db": 20}',
                    "success",
                    output_text="BER 0.00000000 with probability confidence evidence.",
                    metrics_json='{"ber": 0.0}',
                )

                hits = memory.search("BER probability evidence", source_kind="tool_run")

        self.assertTrue(hits)
        self.assertTrue(all(hit["source_kind"] == "tool_run" for hit in hits))
        self.assertEqual(hits[0]["project"], "tool:ofdmsim")

    def test_memory_search_diversifies_repeated_chunks_from_same_source(self):
        repeated = "\n\n".join(
            [
                "radio reliability adapter evidence alpha",
                "radio reliability adapter evidence beta",
                "radio reliability adapter evidence gamma",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                memory.index_document("runtime", "big.md", repeated)
                memory.index_document("runtime", "small.md", "radio reliability adapter evidence delta")

                hits = memory.search("radio reliability adapter evidence", limit=2, max_per_source=1)

        self.assertEqual(len(hits), 2)
        self.assertEqual(len({hit["source_path"] for hit in hits}), 2)

    def test_project_document_indexer_includes_manifests_and_docs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FolderGuardian"
            (root / "docs").mkdir(parents=True)
            (root / "README.md").write_text("FolderGuardian protects folders.", encoding="utf-8")
            (root / "FolderGuardian.csproj").write_text(
                "<Project><PropertyGroup><TargetFramework>net8.0-windows</TargetFramework><UseWPF>true</UseWPF></PropertyGroup></Project>",
                encoding="utf-8",
            )
            (root / "docs" / "security.md").write_text("SecurityLog.txt records watchtower events.", encoding="utf-8")

            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                count = memory.index_project_documents("folderguardian", root)
                hits = memory.search("UseWPF net8.0-windows", project="folderguardian")

        self.assertEqual(count, 3)
        self.assertIn("TargetFramework", hits[0]["snippet"])

    def test_project_source_summary_indexes_symbols_without_full_source_ingest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FolderGuardian"
            (root / "Core").mkdir(parents=True)
            (root / "Core" / "FolderMonitor.cs").write_text(
                """
                namespace FolderGuardian.Core;
                internal sealed class FolderMonitor
                {
                    public void StartWatchtower() {}
                    private void AppendSecurityLog() {}
                }
                """,
                encoding="utf-8",
            )

            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                count = memory.index_project_source_summary("folderguardian", root)
                hits = memory.search("FolderMonitor StartWatchtower", project="folderguardian")

        self.assertEqual(count, 1)
        self.assertIn("__source_summary__.md", hits[0]["source_path"])
        self.assertIn("FolderMonitor", hits[0]["snippet"])

    def test_memory_stores_embedding_provider_metadata(self):
        class TinyProvider:
            name = "tiny-test"
            dimensions = 3

            def embed(self, text):
                return [1.0, 0.0, 0.0]

            def health(self):
                return {"name": self.name, "available": True, "dimensions": self.dimensions}

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.sqlite3"
            with MemoryService(db_path, embedding_provider=TinyProvider()) as memory:
                memory.index_document("project", "doc.md", "A tiny embedding test document.")
                stats = memory.stats()

            conn = sqlite3.connect(str(db_path))
            row = conn.execute(
                """
                SELECT embedding_provider, embedding_dimensions, content_hash
                FROM project_document_chunks
                LIMIT 1
                """
            ).fetchone()
            conn.close()

        self.assertEqual(stats["embedding_provider"], "tiny-test")
        self.assertEqual(stats["embedding_dimensions"], 3)
        self.assertEqual(row[0], "tiny-test")
        self.assertEqual(row[1], 3)
        self.assertTrue(row[2])

    def test_embedding_provider_config_defaults_to_hash(self):
        settings = SimpleNamespace(
            embedding_provider="transformers",
            embedding_model="",
            embedding_hash_dimensions=32,
        )
        provider = create_embedding_provider(settings)

        self.assertIsInstance(provider, HashEmbeddingProvider)
        self.assertEqual(provider.dimensions, 32)

    def test_tool_run_outputs_become_retrievable_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                memory.record_tool_run(
                    "ofdmsim",
                    "run_simulation",
                    '{"snr_db": 18}',
                    "success",
                    output_text="BER 0.00000000, errors 0/1120, Es/N0 18 dB.",
                    metrics_json='{"ber": 0.0, "errors": 0, "bits": 1120}',
                )

                hits = memory.search("BER errors Es/N0", limit=1)

        self.assertEqual(hits[0]["project"], "tool:ofdmsim")
        self.assertIn("BER", hits[0]["snippet"])

    def test_memory_stats_include_recent_tool_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                memory.record_tool_run(
                    "cadconverter",
                    "inspect",
                    "{}",
                    "success",
                    metrics_json='{"files": 2}',
                )

                stats = memory.stats()
                runs = memory.recent_tool_runs()

        self.assertEqual(stats["tool_run_count"], 1)
        self.assertEqual(runs[0]["tool_name"], "cadconverter")
        self.assertEqual(runs[0]["metrics_json"], '{"files": 2}')

    def test_ofdm_param_normalization(self):
        params = _normalize_ofdm_params(
            {
                "channel": "awgn",
                "modulation": "16-qam",
                "equalizer": "zf",
                "snr_db": "18.5",
                "frames": "250",
                "seed": "123",
            }
        )

        self.assertEqual(params["channel"], 1)
        self.assertEqual(params["modulation"], 2)
        self.assertEqual(params["equalizer"], 2)
        self.assertEqual(params["snr_db"], 18.5)
        self.assertEqual(params["frames"], 250)
        self.assertEqual(params["seed"], 123)

    def test_ofdm_quiet_output_format(self):
        summary = _format_ofdm_quiet_output("0.00000000 0 22400 18.00000000 11.97940009")

        self.assertIn("BER 0.00000000", summary)
        self.assertIn("errors 0/22400", summary)

    def test_ofdm_parameter_format_uses_names_not_numeric_codes(self):
        summary = _format_ofdm_params(
            {
                "channel": 1,
                "modulation": 1,
                "equalizer": 1,
                "snr_db": 20.0,
                "frames": 100,
                "seed": 42,
            }
        )

        self.assertIn("channel AWGN", summary)
        self.assertIn("modulation QPSK", summary)
        self.assertIn("equalizer MMSE", summary)
        self.assertNotIn('"channel": 1', summary)

    def test_ofdm_quiet_output_metrics_parse(self):
        metrics = _parse_ofdm_quiet_metrics("0.00000000 0 22400 18.00000000 11.97940009")

        self.assertEqual(metrics["ber"], 0.0)
        self.assertEqual(metrics["errors"], 0)
        self.assertEqual(metrics["bits"], 22400)

    def test_ofdmsim_selftest_records_artifact_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "OFDMSim"
            exe_dir = root / "x64" / "Release"
            exe_dir.mkdir(parents=True)
            exe = exe_dir / "OFDMSim.exe"
            exe.write_text("placeholder", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=root,
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            completed = subprocess.CompletedProcess(
                [str(exe), "test"],
                0,
                stdout="[ OK ] fft_roundtrip\n[ OK ] mod_demod QPSK\n\nAll self-tests passed.\n",
                stderr="",
            )

            with patch("tool_registry.subprocess.run", return_value=completed) as run:
                with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                    response = registry.run_capability("ofdmsim", "selftest", {}, memory=memory)
                    runs = memory.recent_tool_runs()
                    artifact = json.loads(Path(runs[0]["artifact_path"]).read_text(encoding="utf-8"))

        self.assertEqual(run.call_args.args[0], [str(exe), "test"])
        self.assertIn("ofdmsim self-test success", response)
        self.assertIn("Artifact:", response)
        self.assertEqual(runs[0]["tool_name"], "ofdmsim")
        metrics = json.loads(runs[0]["metrics_json"])
        self.assertEqual(metrics["ok_count"], 2)
        self.assertTrue(metrics["all_passed"])
        self.assertEqual(artifact["capability"], "selftest")
        self.assertEqual(artifact["metrics"]["ok_count"], 2)

    def test_ofdmsim_bayesian_update_rewrites_artifact_status(self):
        class FakeBayesianAdapter:
            def add_tool_run_evidence(self, tool_name, artifact_path, ber_threshold=0.01):
                return {
                    "assessment": {
                        "status": "success",
                        "probability": 0.75,
                        "confidence": 0.5,
                        "credible_interval": [0.4, 0.9],
                        "warnings": [],
                        "report_text": "large report omitted",
                    }
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "OFDMSim"
            exe_dir = root / "x64" / "Release"
            exe_dir.mkdir(parents=True)
            exe = exe_dir / "OFDMSim.exe"
            exe.write_text("placeholder", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=root,
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            registry.bayesian_adapter = FakeBayesianAdapter()
            completed = subprocess.CompletedProcess(
                [str(exe), "run"],
                0,
                stdout="0.00000000 0 1120 18.00000000 14.98970004",
                stderr="",
            )

            with patch("tool_registry.subprocess.run", return_value=completed):
                with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                    response = registry.run_capability(
                        "ofdmsim",
                        "run_simulation",
                        {"snr_db": 18, "frames": 10, "update_bayesian": True},
                        memory=memory,
                    )
                    runs = memory.recent_tool_runs()
                    artifact = json.loads(Path(runs[0]["artifact_path"]).read_text(encoding="utf-8"))

        self.assertIn("Bayesian evidence updated", response)
        self.assertEqual(runs[0]["evidence_status"], "updated")
        self.assertEqual(artifact["evidence_status"], "updated")
        self.assertEqual(artifact["bayesian_assessment"]["probability"], 0.75)
        self.assertNotIn("report_text", artifact["bayesian_assessment"])

    def test_orchestrator_routes_basic_ofdm_simulation_test_to_run_with_probability(self):
        class FakeBayesianAdapter:
            def add_tool_run_evidence(self, tool_name, artifact_path, ber_threshold=0.01):
                return {
                    "assessment": {
                        "status": "success",
                        "probability": 0.82,
                        "confidence": 0.61,
                        "credible_interval": [0.5, 0.93],
                        "warnings": [],
                    }
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "OFDMSim"
            exe_dir = root / "x64" / "Release"
            exe_dir.mkdir(parents=True)
            exe = exe_dir / "OFDMSim.exe"
            exe.write_text("placeholder", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=root,
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            registry.bayesian_adapter = FakeBayesianAdapter()
            completed = subprocess.CompletedProcess(
                [str(exe), "run"],
                0,
                stdout="0.00000000 0 11200 20.00000000 16.98970004",
                stderr="",
            )

            with patch("tool_registry.subprocess.run", return_value=completed) as run:
                with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                    orchestrator = RuntimeOrchestrator(memory, registry)
                    response, source = orchestrator.try_handle(
                        "can you run a basic ofdm simulation test in general conditions and give me values and probability"
                    )
                    runs = memory.recent_tool_runs()

        command = run.call_args.args[0]
        self.assertEqual(source, "ToolRegistry")
        self.assertEqual(command[1], "run")
        self.assertIn("--quiet", command)
        self.assertEqual(command[command.index("--channel") + 1], "1")
        self.assertEqual(command[command.index("--modulation") + 1], "1")
        self.assertEqual(command[command.index("--eq") + 1], "1")
        self.assertEqual(command[command.index("--snr-db") + 1], "20.0")
        self.assertIn("BER 0.00000000", response)
        self.assertIn("Bayesian evidence updated", response)
        self.assertIn("82.0%", response)
        self.assertEqual(runs[0]["capability"], "run_simulation")
        self.assertEqual(runs[0]["evidence_status"], "updated")

    def test_orchestrator_parses_ofdm_request(self):
        settings = SimpleNamespace(
            ofdm_root=Path("missing-ofdm"),
            ofdm_emi_root=Path("missing-emi"),
            cadconverter_root=Path("missing-cad"),
            folderguardian_root=Path("missing-folder"),
        )
        registry = create_default_registry(settings)
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                params = orchestrator._parse_ofdm_request(
                    "run ofdmsim awgn 16-qam zf at 18 db for 250 frames seed 123"
                )

        self.assertEqual(params["channel"], "awgn")
        self.assertEqual(params["modulation"], "16-qam")
        self.assertEqual(params["equalizer"], "zf")
        self.assertEqual(params["snr_db"], 18)
        self.assertEqual(params["frames"], 250)
        self.assertEqual(params["seed"], 123)

    def test_orchestrator_runtime_snapshot_is_tool_agnostic(self):
        settings = SimpleNamespace(
            ofdm_root=Path("missing-ofdm"),
            ofdm_emi_root=Path("missing-emi"),
            cadconverter_root=Path("missing-cad"),
            folderguardian_root=Path("missing-folder"),
        )
        registry = create_default_registry(settings)
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                snapshot = orchestrator.snapshot()

        tool_names = {tool["name"] for tool in snapshot["tools"]}
        self.assertIn("ofdmsim", tool_names)
        self.assertIn("cadconverter", tool_names)
        self.assertIn("folderguardian", tool_names)
        self.assertIn("memory", snapshot)

    def test_runtime_service_facade_lists_capabilities_and_schemas(self):
        settings = SimpleNamespace(
            ofdm_root=Path("missing-ofdm"),
            ofdm_emi_root=Path("missing-emi"),
            cadconverter_root=Path("missing-cad"),
            folderguardian_root=Path("missing-folder"),
        )
        registry = create_default_registry(settings)
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                service = RuntimeServiceFacade(memory, registry)
                capabilities = service.list_capabilities()
                schema = service.get_capability_schema("cadconverter", "convert_step_to_glb")
                snapshot = service.runtime_snapshot()

        names = {(item["tool_name"], item["capability_name"]) for item in capabilities}
        self.assertIn(("cadconverter", "convert_step_to_glb"), names)
        self.assertEqual(schema["input_schema"]["input_path"], "path")
        self.assertIn("tools", snapshot)

    def test_runtime_service_facade_indexes_selected_project_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FolderGuardian"
            root.mkdir()
            (root / "README.md").write_text("FolderGuardian service facade memory.", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=root,
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                service = RuntimeServiceFacade(memory, registry)
                result = service.index_project_memory(["folderguardian"])
                hits = service.search_memory("service facade", project="folderguardian")

        self.assertEqual(result["total_indexed_document_count"], 2)
        self.assertEqual(result["results"][0]["project"], "folderguardian")
        self.assertEqual(result["results"][0]["indexed_overview_count"], 1)
        self.assertTrue(hits)

    def test_runtime_service_facade_indexes_source_summary_with_project_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FolderGuardian"
            (root / "Core").mkdir(parents=True)
            (root / "README.md").write_text("FolderGuardian service facade memory.", encoding="utf-8")
            (root / "Core" / "FolderEncryptor.cs").write_text(
                "internal static class FolderEncryptor { public static void EncryptFolderAsync() {} }",
                encoding="utf-8",
            )
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=root,
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                service = RuntimeServiceFacade(memory, registry)
                result = service.index_project_memory(["folderguardian"])
                hits = service.search_memory("FolderEncryptor EncryptFolderAsync", project="folderguardian")

        self.assertEqual(result["total_indexed_document_count"], 3)
        self.assertEqual(result["results"][0]["indexed_regular_document_count"], 1)
        self.assertEqual(result["results"][0]["indexed_source_summary_count"], 1)
        self.assertEqual(result["results"][0]["indexed_overview_count"], 1)
        self.assertTrue(hits)

    def test_runtime_service_facade_runs_capability_with_structured_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FolderGuardian"
            root.mkdir()
            (root / "FolderGuardian.csproj").write_text(
                "<Project><PropertyGroup><TargetFramework>net8.0-windows</TargetFramework></PropertyGroup></Project>",
                encoding="utf-8",
            )
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=root,
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                service = RuntimeServiceFacade(memory, registry)
                result = service.run_capability("folderguardian", "inspect")
                runs = service.recent_tool_runs()

        self.assertEqual(result["status"], "completed")
        self.assertIn("folderguardian inspect success", result["output_text"])
        self.assertEqual(runs[0]["tool_name"], "folderguardian")

    def test_runtime_service_facade_rejects_missing_required_params(self):
        settings = SimpleNamespace(
            ofdm_root=Path("missing-ofdm"),
            ofdm_emi_root=Path("missing-emi"),
            cadconverter_root=Path("missing-cad"),
            folderguardian_root=Path("missing-folder"),
        )
        registry = create_default_registry(settings)
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                service = RuntimeServiceFacade(memory, registry)
                result = service.run_capability("cadconverter", "validate_glb", {})
                runs = service.recent_tool_runs()

        self.assertEqual(result["status"], "failed")
        self.assertIn("input_path is required", result["error_message"])
        self.assertEqual(runs, [])

    def test_runtime_service_facade_rejects_unknown_and_wrong_type_params(self):
        settings = SimpleNamespace(
            ofdm_root=Path("missing-ofdm"),
            ofdm_emi_root=Path("missing-emi"),
            cadconverter_root=Path("missing-cad"),
            folderguardian_root=Path("missing-folder"),
        )
        registry = create_default_registry(settings)
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                service = RuntimeServiceFacade(memory, registry)
                result = service.run_capability(
                    "ofdmsim",
                    "run_simulation",
                    {"frames": "ten", "surprise": True},
                )

        self.assertEqual(result["status"], "failed")
        self.assertIn("frames must be an integer", result["error_message"])
        self.assertIn("surprise is not a supported parameter", result["error_message"])

    @unittest.skipIf(runtime_contract_pb2 is None, "runtime gRPC bindings require grpc/protobuf")
    def test_runtime_grpc_service_lists_and_runs_capabilities(self):
        class FakeRuntimeService:
            def list_capabilities(self):
                return [
                    {
                        "tool_name": "folderguardian",
                        "capability_name": "inspect",
                        "description": "Inspect project metadata.",
                        "input_schema": {},
                        "safety_level": "safe",
                        "supports_progress": False,
                        "supports_cancel": False,
                    }
                ]

            def run_capability(self, tool_name, capability_name, params):
                return {
                    "status": "completed",
                    "output_json": json.dumps({"text": f"{tool_name}.{capability_name} ok", "params": params}),
                    "error_message": "",
                }

        service = RuntimeGrpcService(FakeRuntimeService())
        listed = service.ListCapabilities(runtime_contract_pb2.ListCapabilitiesRequest(), None)
        response = service.RunCapability(
            runtime_contract_pb2.RunCapabilityRequest(
                capability=runtime_contract_pb2.CapabilityRef(
                    tool_name="folderguardian",
                    capability_name="inspect",
                ),
                input_json="{}",
            ),
            None,
        )

        self.assertEqual(listed.capabilities[0].tool_name, "folderguardian")
        self.assertEqual(response.status, "completed")
        self.assertIn("folderguardian.inspect ok", response.output_json)

    @unittest.skipIf(runtime_contract_pb2 is None or VoiceGrpcService is None, "runtime voice gRPC bindings require grpc/protobuf")
    def test_voice_grpc_stream_collects_chunks_and_returns_final_event(self):
        class FakeAIService:
            def __init__(self):
                self.request = None

            def RecognizeSpeech(self, request, context):
                self.request = request
                return SimpleNamespace(response_text="hello there", ai_source="EnglishSpeechRecognition")

        fake = FakeAIService()
        service = VoiceGrpcService(fake)
        events = list(
            service.StreamRecognizeSpeech(
                iter(
                    [
                        runtime_contract_pb2.AudioChunk(
                            audio_data=b"abc",
                            language_code="en-US",
                            sample_rate=16000,
                        ),
                        runtime_contract_pb2.AudioChunk(
                            audio_data=b"def",
                            language_code="en-US",
                            sample_rate=16000,
                            end_of_utterance=True,
                        ),
                    ]
                ),
                None,
            )
        )

        self.assertEqual(events[0].type, "partial")
        self.assertIn("audio_started", [event.text for event in events])
        self.assertIn("recognizing", [event.text for event in events])
        self.assertEqual(events[-1].type, "final")
        self.assertEqual(events[-1].text, "hello there")
        self.assertEqual(bytes(fake.request.audio_data), b"abcdef")

    def test_orchestrator_returns_capability_schema(self):
        settings = SimpleNamespace(
            ofdm_root=Path("missing-ofdm"),
            ofdm_emi_root=Path("missing-emi"),
            cadconverter_root=Path("missing-cad"),
            folderguardian_root=Path("missing-folder"),
        )
        registry = create_default_registry(settings)
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                response, source = orchestrator.try_handle("schema for cadconverter.convert_step_to_glb")

        payload = json.loads(response)
        self.assertEqual(source, "RuntimeOrchestrator")
        self.assertEqual(payload["tool_name"], "cadconverter")
        self.assertEqual(payload["capability_name"], "convert_step_to_glb")

    def test_orchestrator_runs_generic_capability_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FolderGuardian"
            root.mkdir()
            (root / "FolderGuardian.csproj").write_text(
                "<Project><PropertyGroup><TargetFramework>net8.0-windows</TargetFramework></PropertyGroup></Project>",
                encoding="utf-8",
            )
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=root,
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                response, source = orchestrator.try_handle("run capability folderguardian.inspect {}")
                runs = memory.recent_tool_runs()

        self.assertEqual(source, "ToolRegistry")
        self.assertIn("folderguardian inspect success", response)
        self.assertEqual(runs[0]["tool_name"], "folderguardian")

    def test_orchestrator_generic_capability_command_rejects_bad_json(self):
        settings = SimpleNamespace(
            ofdm_root=Path("missing-ofdm"),
            ofdm_emi_root=Path("missing-emi"),
            cadconverter_root=Path("missing-cad"),
            folderguardian_root=Path("missing-folder"),
        )
        registry = create_default_registry(settings)
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                response, source = orchestrator.try_handle("run capability folderguardian.inspect {bad")

        self.assertEqual(source, "RuntimeOrchestrator")
        self.assertIn("Capability parameters must be a JSON object", response)

    def test_model_registry_routes_query_types(self):
        settings = SimpleNamespace(
            mistral_api_key="",
            mistral_model="mistral-large-latest",
            wizardmath_model="WizardMath",
            phi3_model="Phi-3",
            embedding_provider="hash",
            embedding_model="",
        )
        registry = create_default_model_registry(settings, HashEmbeddingProvider())

        self.assertEqual(registry.route_query("calculate 2+2"), "math_service")
        self.assertEqual(registry.route_query("what is OFDM"), "phi3")
        self.assertEqual(registry.route_query("run OFDMSim at 15 dB and update Bayesian confidence"), "runtime_orchestrator")
        self.assertEqual(registry.route_query("what BER do we get for OFDM at 18 dB"), "runtime_orchestrator")
        self.assertEqual(registry.route_query("convert the STEP file to GLB with the CAD converter"), "runtime_orchestrator")
        self.assertEqual(registry.route_query("write a detailed architecture note"), "mistral")
        decision = registry.route_decision("solve x^2 - 4 = 0")
        self.assertEqual(decision.intent, "math")
        self.assertEqual(decision.target, "math_service")

    def test_runtime_model_service_lists_and_controls_phi(self):
        settings = SimpleNamespace(
            mistral_api_key="",
            mistral_model="mistral-large-latest",
            wizardmath_model="WizardMath",
            phi3_model="Phi-3",
            embedding_provider="hash",
            embedding_model="",
        )
        registry = create_default_model_registry(settings, HashEmbeddingProvider())
        service = RuntimeModelServiceFacade(
            registry,
            load_handlers={"phi3": lambda: (registry.set_availability("phi3", True, "loaded") or "loaded")},
            unload_handlers={"phi3": lambda: (registry.set_availability("phi3", False, "unloaded") or "unloaded")},
            probe_handlers={"phi3": lambda prompt: f"probe response for {prompt[:12]}"},
        )

        listed = service.list_models()
        loaded = service.load_model("phi")
        probe = service.probe_model("phi3", [{"category": "routing", "prompt": "route this", "expected_hint": "probe"}])
        unloaded = service.unload_model("phi-3")
        unsupported = service.load_model("wizardmath")

        self.assertGreaterEqual(listed["model_count"], 4)
        self.assertTrue(loaded["success"])
        self.assertEqual(loaded["model"]["status"], "loaded")
        self.assertTrue(probe["success"])
        self.assertEqual(probe["results"][0]["category"], "routing")
        self.assertTrue(unloaded["success"])
        self.assertEqual(unloaded["model"]["status"], "unloaded")
        self.assertFalse(unsupported["success"])
        self.assertEqual(unsupported["status"], "unsupported")

    @unittest.skipIf(runtime_contract_pb2 is None, "runtime gRPC bindings require grpc/protobuf")
    def test_runtime_model_grpc_service_lists_and_loads_models(self):
        settings = SimpleNamespace(
            mistral_api_key="",
            mistral_model="mistral-large-latest",
            wizardmath_model="WizardMath",
            phi3_model="Phi-3",
            embedding_provider="hash",
            embedding_model="",
        )
        registry = create_default_model_registry(settings, HashEmbeddingProvider())
        service = RuntimeModelGrpcService(
            RuntimeModelServiceFacade(
                registry,
                load_handlers={"phi3": lambda: (registry.set_availability("phi3", True, "loaded") or "loaded")},
                probe_handlers={"phi3": lambda prompt: "tool"},
            )
        )

        listed = service.ListModels(runtime_contract_pb2.ListModelsRequest(), None)
        loaded = service.LoadModel(runtime_contract_pb2.ModelRef(model_name="phi3"), None)
        probe = service.ProbeModel(
            runtime_contract_pb2.ModelProbeRequest(
                model_name="phi3",
                prompts=[
                    runtime_contract_pb2.ModelProbePrompt(
                        category="routing",
                        prompt="run OFDMSim",
                        expected_hint="tool",
                    )
                ],
            ),
            None,
        )

        self.assertGreaterEqual(listed.model_count, 4)
        self.assertEqual(listed.models[0].name, "runtime_orchestrator")
        self.assertTrue(loaded.success)
        self.assertEqual(loaded.model.name, "phi3")
        self.assertEqual(loaded.model.status, "loaded")
        self.assertTrue(probe.success)
        self.assertEqual(probe.results[0].response_text, "tool")

    def test_intent_router_routes_math_before_phi(self):
        decision = IntentRouter().route("what is the derivative of x^3?")
        self.assertEqual(decision.intent, "math")
        self.assertEqual(decision.target, "math_service")

    def test_intent_router_routes_tool_before_math_like_numbers(self):
        decision = IntentRouter().route("run OFDMSim at 15 dB and update Bayesian confidence")
        self.assertEqual(decision.intent, "tool")
        self.assertEqual(decision.target, "runtime_orchestrator")

    def test_math_service_safe_arithmetic_without_sympy(self):
        service = MathService()
        service.sympy = None

        self.assertEqual(service.handle("calculate 2*(3+4)"), "Result: 14")

    def test_math_service_symbolic_when_sympy_available(self):
        service = MathService()
        if not service.sympy:
            self.skipTest("sympy is not installed in this Python environment")

        self.assertIn("Derivative with respect to x: 3*x**2", service.handle("derivative of x^3"))

    def test_orchestrator_snapshot_includes_model_registry_when_configured(self):
        settings = SimpleNamespace(
            ofdm_root=Path("missing-ofdm"),
            ofdm_emi_root=Path("missing-emi"),
            cadconverter_root=Path("missing-cad"),
            folderguardian_root=Path("missing-folder"),
            mistral_api_key="",
            mistral_model="mistral-large-latest",
            wizardmath_model="WizardMath",
            phi3_model="Phi-3",
            embedding_provider="hash",
            embedding_model="",
        )
        tool_registry = create_default_registry(settings)
        model_registry = create_default_model_registry(settings, HashEmbeddingProvider())
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, tool_registry, model_registry)
                snapshot = orchestrator.snapshot()

        self.assertIn("models", snapshot)
        self.assertGreaterEqual(snapshot["models"]["model_count"], 4)

    def test_orchestrator_can_return_structured_plan_without_executing(self):
        settings = SimpleNamespace(
            ofdm_root=Path("missing-ofdm"),
            ofdm_emi_root=Path("missing-emi"),
            cadconverter_root=Path("missing-cad"),
            folderguardian_root=Path("missing-folder"),
        )
        registry = create_default_registry(settings)
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                response, source = orchestrator.try_handle("plan run OFDMSim AWGN QPSK at 18 dB for 10 frames")

        payload = json.loads(response)
        self.assertEqual(source, "RuntimeOrchestrator")
        self.assertEqual(payload["source"], "ToolRegistry")
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["steps"][0]["action"], "resolve_capability")

    def test_orchestrator_records_execution_plan_history(self):
        settings = SimpleNamespace(
            ofdm_root=Path("missing-ofdm"),
            ofdm_emi_root=Path("missing-emi"),
            cadconverter_root=Path("missing-cad"),
            folderguardian_root=Path("missing-folder"),
        )
        registry = create_default_registry(settings)
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                response, source = orchestrator.try_handle("list tools")
                snapshot = orchestrator.snapshot()

        self.assertEqual(source, "RuntimeOrchestrator")
        self.assertIn("Available engineering capabilities", response)
        self.assertEqual(snapshot["recent_execution_plans"][0]["status"], "completed")

    def test_orchestrator_refreshes_project_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FolderGuardian"
            root.mkdir()
            (root / "README.md").write_text("FolderGuardian watchtower memory.", encoding="utf-8")
            (root / "FolderGuardian.csproj").write_text(
                "<Project><PropertyGroup><TargetFramework>net8.0-windows</TargetFramework></PropertyGroup></Project>",
                encoding="utf-8",
            )
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=root,
                cpp_workspace_root=Path("missing-cpp"),
                python_workspace_root=Path("missing-python"),
                csharp_workspace_root=Path("missing-csharp"),
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                response, source = orchestrator.try_handle("refresh project memory")
                hits = memory.search("watchtower net8.0-windows", project="folderguardian")

        self.assertEqual(source, "MemoryService")
        self.assertIn("folderguardian: 3", response)
        self.assertTrue(hits)

    def test_execution_plans_are_persisted_in_memory(self):
        settings = SimpleNamespace(
            ofdm_root=Path("missing-ofdm"),
            ofdm_emi_root=Path("missing-emi"),
            cadconverter_root=Path("missing-cad"),
            folderguardian_root=Path("missing-folder"),
        )
        registry = create_default_registry(settings)
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.sqlite3"
            with MemoryService(db_path) as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                orchestrator.try_handle("list tools")
                stats = memory.stats()

            with MemoryService(db_path) as memory:
                persisted = memory.recent_execution_plans(limit=1)
                replayed = RuntimeOrchestrator(memory, registry).snapshot()

        self.assertEqual(stats["execution_plan_count"], 1)
        self.assertEqual(persisted[0]["status"], "completed")
        self.assertEqual(persisted[0]["steps"][0]["action"], "inspect_registry")
        self.assertEqual(replayed["recent_execution_plans"][0]["id"], persisted[0]["id"])

    def test_cadconverter_inspect_records_runtime_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "CadConverter"
            root.mkdir()
            (root / "part.step").write_text("step", encoding="utf-8")
            (root / "scene.glb").write_bytes(b"glb")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=root,
                cadconverter_output_dir=Path(tmp) / "outputs",
                cadconverter_python="python",
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                response = registry.run_capability("cadconverter", "inspect", {}, memory=memory)
                runs = memory.recent_tool_runs()

        self.assertIn("STEP/STP files: 1", response)
        self.assertIn("GLB/GLTF files: 1", response)
        self.assertEqual(runs[0]["tool_name"], "cadconverter")
        self.assertIn("tool_runs", runs[0]["artifact_path"])

    def test_folderguardian_inspect_records_project_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FolderGuardian"
            (root / "Core").mkdir(parents=True)
            (root / "FolderGuardian.sln").write_text("solution", encoding="utf-8")
            (root / "FolderGuardian.csproj").write_text(
                """
                <Project Sdk="Microsoft.NET.Sdk">
                  <PropertyGroup>
                    <TargetFramework>net8.0-windows</TargetFramework>
                    <UseWPF>true</UseWPF>
                    <UseWindowsForms>true</UseWindowsForms>
                  </PropertyGroup>
                  <ItemGroup>
                    <PackageReference Include="System.Security.Cryptography.ProtectedData" Version="9.0.8" />
                  </ItemGroup>
                </Project>
                """,
                encoding="utf-8",
            )
            (root / "README.md").write_text(
                "AES-256 encryption with DPAPI and SecurityLog.txt monitoring.",
                encoding="utf-8",
            )
            (root / "Core" / "FolderMonitor.cs").write_text(
                "var watcher = new FileSystemWatcher(); var log = \"SecurityLog.txt\";",
                encoding="utf-8",
            )
            (root / "Core" / "EncryptionHelper.cs").write_text(
                "Aes.Create(); ProtectedData.Protect(bytes, null, DataProtectionScope.CurrentUser);",
                encoding="utf-8",
            )
            (root / "MainWindow.xaml").write_text("<Window />", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=root,
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                response = registry.run_capability("folderguardian", "inspect", {}, memory=memory)
                runs = memory.recent_tool_runs()

        metrics = json.loads(runs[0]["metrics_json"])
        self.assertIn("folderguardian inspect success", response)
        self.assertIn("Adapter safety: live encrypt/decrypt", response)
        self.assertEqual(runs[0]["tool_name"], "folderguardian")
        self.assertEqual(metrics["target_frameworks"], ["net8.0-windows"])
        self.assertTrue(metrics["uses_wpf"])
        self.assertTrue(metrics["uses_windows_forms"])
        self.assertIn("DPAPI key protection", metrics["detected_features"])
        self.assertIn("FileSystemWatcher monitoring", metrics["detected_features"])
        self.assertIn("tool_runs", runs[0]["artifact_path"])

    def test_cadconverter_convert_builds_cli_command_and_records_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "CadConverter"
            root.mkdir()
            step = root / "part.step"
            step.write_text("step", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=root,
                cadconverter_output_dir=Path(tmp) / "outputs",
                cadconverter_python="cad-python",
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            completed = subprocess.CompletedProcess(
                ["cad-python"],
                0,
                stdout="Exported: output.glb\nOK: output.glb\n  nodes=2, meshes=1, materials=1, primitives=1\n  colors=1/1, normals=1/1, named_nodes=2/2",
                stderr="",
            )

            with patch("tool_registry.subprocess.run", return_value=completed) as run:
                with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                    response = registry.run_capability(
                        "cadconverter",
                        "convert_step_to_glb",
                        {"input_path": "part.step"},
                        memory=memory,
                    )
                    runs = memory.recent_tool_runs()

        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["cad-python", "-m", "cadconverter.cli", "convert"])
        self.assertIn(str(step.resolve()), command)
        self.assertIn("--validate", command)
        self.assertIn("success", response)
        self.assertIn('"nodes": 2', runs[0]["metrics_json"])

    def test_ofdmsim_emi_compare_builds_cli_command_and_records_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "OFDMSim_EMI"
            exe_dir = root / "x64" / "Release"
            exe_dir.mkdir(parents=True)
            exe = exe_dir / "OFDMSim.exe"
            exe.write_text("placeholder", encoding="utf-8")
            (root / "examples").mkdir()
            (root / "examples" / "sim_default.ini").write_text("emi=false", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=root,
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            completed = subprocess.CompletedProcess(
                [str(exe)],
                0,
                stdout=(
                    "case,ber,errors,bits,emi_impulses,emi_samples_hit,clipped_samples\n"
                    "baseline,0,0,11200,0,0,0\n"
                    "emi,0.07446429,834,11200,20,20,0\n"
                    "emi_clipping,0.03053571,342,11200,20,20,20\n"
                ),
                stderr="",
            )

            with patch("tool_registry.subprocess.run", return_value=completed) as run:
                with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                    response = registry.run_capability(
                        "ofdmsim_emi",
                        "compare_emi",
                        {
                            "channel": "awgn",
                            "modulation": "qpsk",
                            "equalizer": "mmse",
                            "snr_db": 18,
                            "frames": 100,
                            "seed": 123,
                            "emi_probability": 0.003,
                            "emi_amplitude": 15,
                        },
                        memory=memory,
                    )
                    runs = memory.recent_tool_runs()

        command = run.call_args.args[0]
        self.assertEqual(command[:2], [str(exe), "compare"])
        self.assertIn("--emi-probability", command)
        self.assertIn("EMI comparison success", response)
        metrics = json.loads(runs[0]["metrics_json"])
        self.assertAlmostEqual(metrics["emi_ber"], 0.07446429)
        self.assertAlmostEqual(metrics["clipping_ber_improvement"], 0.04392858)

    def test_emi_compare_metric_parser_extracts_cases(self):
        metrics = _parse_emi_compare_metrics(
            "case,ber,errors,bits,emi_impulses,emi_samples_hit,clipped_samples\n"
            "baseline,0,0,100,0,0,0\n"
            "emi,0.2,20,100,4,4,0\n"
            "emi_clipping,0.1,10,100,4,4,4\n"
        )

        self.assertEqual(len(metrics["cases"]), 3)
        self.assertEqual(metrics["baseline_ber"], 0.0)
        self.assertEqual(metrics["emi_ber_delta"], 0.2)
        self.assertEqual(metrics["clipping_ber_improvement"], 0.1)

    def test_orchestrator_routes_ofdmsim_emi_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "OFDMSim_EMI"
            exe_dir = root / "x64" / "Release"
            exe_dir.mkdir(parents=True)
            exe = exe_dir / "OFDMSim.exe"
            exe.write_text("placeholder", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=root,
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            completed = subprocess.CompletedProcess(
                [str(exe)],
                0,
                stdout=(
                    "case,ber,errors,bits,emi_impulses,emi_samples_hit,clipped_samples\n"
                    "baseline,0,0,100,0,0,0\nemi,0.2,20,100,4,4,0\nemi_clipping,0.1,10,100,4,4,4\n"
                ),
                stderr="",
            )

            with patch("tool_registry.subprocess.run", return_value=completed):
                with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                    orchestrator = RuntimeOrchestrator(memory, registry)
                    response, source = orchestrator.try_handle(
                        "compare OFDMSim EMI awgn qpsk at 18 db for 100 frames emi probability 0.003 amplitude 15"
                    )

        self.assertEqual(source, "ToolRegistry")
        self.assertIn("EMI comparison success", response)

    def test_orchestrator_routes_natural_ofdm_run_with_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "OFDMSim"
            exe_dir = root / "x64" / "Release"
            exe_dir.mkdir(parents=True)
            exe = exe_dir / "OFDMSim.exe"
            exe.write_text("placeholder", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=root,
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            completed = subprocess.CompletedProcess(
                [str(exe)],
                0,
                stdout="0.01000000 1 100 12.50000000 9.48970004",
                stderr="",
            )

            with patch("tool_registry.subprocess.run", return_value=completed) as run:
                with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                    orchestrator = RuntimeOrchestrator(memory, registry)
                    response, source = orchestrator.try_handle(
                        "what BER do we get for OFDM over Rayleigh fading using 64 QAM "
                        "and zero forcing at SNR 12.5 dB for 250 frames seed 9"
                    )

        command = run.call_args.args[0]
        self.assertEqual(source, "ToolRegistry")
        self.assertIn("simulation success", response)
        self.assertEqual(command[command.index("--channel") + 1], "2")
        self.assertEqual(command[command.index("--modulation") + 1], "3")
        self.assertEqual(command[command.index("--eq") + 1], "2")
        self.assertEqual(command[command.index("--snr-db") + 1], "12.5")
        self.assertEqual(command[command.index("--frames") + 1], "250")
        self.assertEqual(command[command.index("--seed") + 1], "9")

    def test_orchestrator_routes_natural_ofdm_emi_without_compare_keyword(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "OFDMSim_EMI"
            exe_dir = root / "x64" / "Release"
            exe_dir.mkdir(parents=True)
            exe = exe_dir / "OFDMSim.exe"
            exe.write_text("placeholder", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=root,
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            completed = subprocess.CompletedProcess(
                [str(exe)],
                0,
                stdout=(
                    "case,ber,errors,bits,emi_impulses,emi_samples_hit,clipped_samples\n"
                    "baseline,0,0,100,0,0,0\nemi,0.2,20,100,4,4,0\nemi_clipping,0.1,10,100,4,4,4\n"
                ),
                stderr="",
            )

            with patch("tool_registry.subprocess.run", return_value=completed) as run:
                with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                    orchestrator = RuntimeOrchestrator(memory, registry)
                    response, source = orchestrator.try_handle(
                        "run an OFDM EMI case over AWGN QPSK at 18 dB for 100 frames "
                        "with impulse probability 0.003 amplitude 15 width 2 and clipping at 6"
                    )

        command = run.call_args.args[0]
        self.assertEqual(source, "ToolRegistry")
        self.assertEqual(command[:2], [str(exe), "compare"])
        self.assertIn("EMI comparison success", response)
        self.assertEqual(command[command.index("--emi-probability") + 1], "0.003")
        self.assertEqual(command[command.index("--emi-amplitude") + 1], "15.0")
        self.assertEqual(command[command.index("--emi-width") + 1], "2")
        self.assertEqual(command[command.index("--clip-threshold") + 1], "6.0")

    def test_orchestrator_routes_natural_cadconverter_export_by_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "CadConverter"
            root.mkdir()
            step = root / "python half.stp"
            step.write_text("step", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=root,
                cadconverter_output_dir=Path(tmp) / "outputs",
                cadconverter_python="cad-python",
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            completed = subprocess.CompletedProcess(
                ["cad-python"],
                0,
                stdout="Exported: output.glb\nOK: output.glb\n  nodes=2, meshes=1, materials=1, primitives=1",
                stderr="",
            )

            with patch("tool_registry.subprocess.run", return_value=completed) as run:
                with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                    orchestrator = RuntimeOrchestrator(memory, registry)
                    response, source = orchestrator.try_handle(
                        "use the CAD converter to export python half as GLB with linear deflection 0.1"
                    )

        command = run.call_args.args[0]
        self.assertEqual(source, "ToolRegistry")
        self.assertIn("convert_step_to_glb success", response)
        self.assertIn(str(step.resolve()), command)
        self.assertIn("--linear-deflection", command)
        self.assertEqual(command[command.index("--linear-deflection") + 1], "0.1")

    def test_orchestrator_routes_natural_cadconverter_validate_by_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "CadConverter"
            root.mkdir()
            glb = root / "scene.glb"
            glb.write_bytes(b"glb")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=root,
                cadconverter_output_dir=Path(tmp) / "outputs",
                cadconverter_python="cad-python",
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            completed = subprocess.CompletedProcess(
                ["cad-python"],
                0,
                stdout="OK: scene.glb\n  nodes=3, meshes=2, materials=1, primitives=4",
                stderr="",
            )

            with patch("tool_registry.subprocess.run", return_value=completed) as run:
                with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                    orchestrator = RuntimeOrchestrator(memory, registry)
                    response, source = orchestrator.try_handle("check the scene GLB file in the CAD workbench")

        command = run.call_args.args[0]
        self.assertEqual(source, "ToolRegistry")
        self.assertIn("validate_glb success", response)
        self.assertIn(str(glb.resolve()), command)

    def test_orchestrator_runs_cadconverter_and_updates_bayesian_when_probability_requested(self):
        class FakeBayesianAdapter:
            def add_tool_run_evidence(self, tool_name, artifact_path, ber_threshold=0.01):
                return {
                    "evidence": [
                        {"factor": "CAD operation executed successfully", "passed": True, "status": "added"},
                        {"factor": "CAD output is machine-readable", "passed": True, "status": "added"},
                    ],
                    "assessment": {
                        "status": "success",
                        "probability": 0.79,
                        "confidence": 0.56,
                        "credible_interval": [0.45, 0.9],
                        "warnings": [],
                    },
                    "config": {},
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "CadConverter"
            root.mkdir()
            glb = root / "scene.glb"
            glb.write_bytes(b"glb")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=root,
                cadconverter_output_dir=Path(tmp) / "outputs",
                cadconverter_python="cad-python",
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            registry.bayesian_adapter = FakeBayesianAdapter()
            completed = subprocess.CompletedProcess(
                ["cad-python"],
                0,
                stdout="OK: scene.glb\n  nodes=3, meshes=2, materials=1, primitives=4",
                stderr="",
            )

            with patch("tool_registry.subprocess.run", return_value=completed):
                with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                    orchestrator = RuntimeOrchestrator(memory, registry)
                    response, source = orchestrator.try_handle('validate CADConverter GLB "scene.glb" and give probability')
                    runs = memory.recent_tool_runs()

        self.assertEqual(source, "ToolRegistry")
        self.assertIn("Bayesian evidence updated", response)
        self.assertIn("79.0%", response)
        self.assertEqual(runs[0]["capability"], "validate_glb")
        self.assertEqual(runs[0]["evidence_status"], "updated")

    def test_orchestrator_routes_natural_bayesian_confidence_request(self):
        class FakeBayesianAdapter:
            def assess_tool_confidence(self, tool_name):
                self.tool_name = tool_name
                return SimpleNamespace(probability=0.88, confidence=0.66, report_text="")

        with tempfile.TemporaryDirectory() as tmp:
            bayesian_root = Path(tmp) / "Bayesian Engine"
            bayesian_root.mkdir()
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                bayesian_engine_root=bayesian_root,
                bayesian_python=Path(tmp) / "python.exe",
                bayesian_state_dir=Path(tmp) / "bayesian_state",
            )
            registry = create_default_registry(settings)
            registry.bayesian_adapter = FakeBayesianAdapter()
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                response, source = orchestrator.try_handle("is the OFDM tool output reliable?")
                folder_response, folder_source = orchestrator.try_handle("is FolderGuardian output reliable?")

        self.assertEqual(source, "BayesianEngine")
        self.assertIn("Bayesian confidence for ofdmsim: 88.0%", response)
        self.assertEqual(folder_source, "BayesianEngine")
        self.assertIn("Bayesian confidence for folderguardian: 88.0%", folder_response)

    def test_orchestrator_routes_folderguardian_inspect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FolderGuardian"
            root.mkdir()
            (root / "FolderGuardian.csproj").write_text(
                "<Project><PropertyGroup><TargetFramework>net8.0-windows</TargetFramework><UseWPF>true</UseWPF></PropertyGroup></Project>",
                encoding="utf-8",
            )
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=root,
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                response, source = orchestrator.try_handle("inspect FolderGuardian project")

        self.assertEqual(source, "ToolRegistry")
        self.assertIn("folderguardian inspect success", response)

    def test_orchestrator_routes_folderguardian_encrypt_to_live_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FolderGuardian"
            root.mkdir()
            target = Path(tmp) / "Temp" / "Demo"
            target.mkdir(parents=True)
            fake_bridge = Path(tmp) / "FolderGuardianBridge.exe"
            fake_bridge.write_text("bridge", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=root,
                folderguardian_bridge_exe=fake_bridge,
                folderguardian_live_actions=True,
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                bridge_payload = {
                    "status": "success",
                    "action": "encrypt",
                    "folder_path": str(target),
                    "summary": {
                        "processed_count": 1,
                        "skipped_count": 0,
                        "failed_count": 0,
                        "total_count": 1,
                        "duration_ms": 12,
                    },
                    "key_location": "test-key",
                }
                with patch("tool_registry.subprocess.run") as run_mock:
                    run_mock.return_value = subprocess.CompletedProcess(
                        [str(fake_bridge), "encrypt", "--folder", str(target), "--json"],
                        0,
                        stdout=json.dumps(bridge_payload),
                        stderr="",
                    )
                    response, source = orchestrator.try_handle(f'encrypt folder "{target}"')
                runs = memory.recent_tool_runs()

        self.assertEqual(source, "ToolRegistry")
        self.assertIn("live execution success", response)
        self.assertIn(str(target), response)
        self.assertEqual(runs[0]["tool_name"], "folderguardian")
        self.assertEqual(runs[0]["capability"], "encrypt_folder")

    def test_orchestrator_runs_folderguardian_and_updates_bayesian_when_confidence_requested(self):
        class FakeBayesianAdapter:
            def add_tool_run_evidence(self, tool_name, artifact_path, ber_threshold=0.01):
                return {
                    "evidence": [
                        {"factor": "FolderGuardian operation executed successfully", "passed": True, "status": "added"},
                        {"factor": "FolderGuardian bridge output is machine-readable", "passed": True, "status": "added"},
                    ],
                    "assessment": {
                        "status": "success",
                        "probability": 0.84,
                        "confidence": 0.62,
                        "credible_interval": [0.5, 0.93],
                        "warnings": [],
                    },
                    "config": {},
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FolderGuardian"
            root.mkdir()
            target = Path(tmp) / "Temp" / "Demo"
            target.mkdir(parents=True)
            fake_bridge = Path(tmp) / "FolderGuardianBridge.exe"
            fake_bridge.write_text("bridge", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=root,
                folderguardian_bridge_exe=fake_bridge,
                folderguardian_live_actions=True,
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            registry.bayesian_adapter = FakeBayesianAdapter()
            bridge_payload = {
                "status": "success",
                "action": "encrypt",
                "folder_path": str(target),
                "summary": {
                    "processed_count": 1,
                    "skipped_count": 0,
                    "failed_count": 0,
                    "total_count": 1,
                    "duration_ms": 12,
                },
                "key_location": "test-key",
            }
            with patch("tool_registry.subprocess.run") as run_mock:
                run_mock.return_value = subprocess.CompletedProcess(
                    [str(fake_bridge), "encrypt", "--folder", str(target), "--json"],
                    0,
                    stdout=json.dumps(bridge_payload),
                    stderr="",
                )
                with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                    orchestrator = RuntimeOrchestrator(memory, registry)
                    response, source = orchestrator.try_handle(f'encrypt folder "{target}" and give confidence')
                    runs = memory.recent_tool_runs()

        self.assertEqual(source, "ToolRegistry")
        self.assertIn("Bayesian evidence updated", response)
        self.assertIn("84.0%", response)
        self.assertEqual(runs[0]["capability"], "encrypt_folder")
        self.assertEqual(runs[0]["evidence_status"], "updated")

    def test_orchestrator_routes_folderguardian_dry_run_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FolderGuardian"
            root.mkdir()
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=root,
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                response, source = orchestrator.try_handle('dry run FolderGuardian protect "D:\\Demo"')

        self.assertEqual(source, "ToolRegistry")
        self.assertIn("action request prepared", response)
        self.assertIn("No live encryption", response)

    def test_orchestrator_parses_folderguardian_natural_windows_path(self):
        orchestrator = RuntimeOrchestrator(None, SimpleNamespace(list_tools=lambda: []))
        params = orchestrator._parse_folderguardian_path_request("check logs for Demo in Temp in D Drive")

        self.assertEqual(params["folder_path"], "D:\\Temp\\Demo")

    def test_orchestrator_treats_lock_unlock_as_folderguardian_crypto(self):
        orchestrator = RuntimeOrchestrator(None, SimpleNamespace(list_tools=lambda: []))

        self.assertTrue(orchestrator._mentions_folderguardian("lock Demo in Temp in D Drive"))
        self.assertTrue(orchestrator._mentions_folderguardian("unlock Demo in Temp in D Drive"))
        self.assertEqual(orchestrator._folderguardian_crypto_action("lock demo in temp in d drive"), "encrypt_folder")
        self.assertEqual(orchestrator._folderguardian_crypto_action("unlock demo in temp in d drive"), "decrypt_folder")

    def test_orchestrator_routes_folderguardian_log_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FolderGuardian"
            root.mkdir()
            target = Path(tmp) / "Temp" / "Demo"
            target.mkdir(parents=True)
            (target / "SecurityLog.txt").write_text("line one\nline two\n", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=root,
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                response, source = orchestrator.try_handle(f'check logs for "{target}"')

        self.assertEqual(source, "ToolRegistry")
        self.assertIn("log check complete", response)
        self.assertIn("line two", response)

    def test_folderguardian_start_and_stop_watch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FolderGuardian"
            root.mkdir()
            target = Path(tmp) / "WatchMe"
            target.mkdir()
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=root,
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                start = registry.run_capability("folderguardian", "start_watch", {"folder_path": str(target)}, memory=memory)
                status = registry.run_capability("folderguardian", "watch_status", {}, memory=memory)
                stop = registry.run_capability("folderguardian", "stop_watch", {"folder_path": str(target)}, memory=memory)

        self.assertIn("watch started", start)
        self.assertIn("Active FolderGuardian watches", status)
        self.assertIn("watch stopped", stop)

    def test_security_suite_clean_folder_returns_low_risk_and_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "clean"
            target.mkdir()
            (target / "notes.txt").write_text("ordinary project notes", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                response = registry.run_capability(
                    "security_suite",
                    "assess_path",
                    {"target_path": str(target), "include_defender_status": False},
                    memory=memory,
                )
                run = memory.recent_tool_runs(limit=1)[0]
            artifact = json.loads(Path(run["artifact_path"]).read_text(encoding="utf-8"))

        self.assertIn("local risk assessment complete", response)
        self.assertEqual(run["tool_name"], "security_suite")
        self.assertEqual(artifact["risk_level"], "low")
        self.assertEqual(artifact["target_path"], str(target.resolve()))
        self.assertIn("file_hashes", artifact)

    def test_security_suite_flags_suspicious_script_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "scripts"
            target.mkdir()
            (target / "setup.ps1").write_text(
                "powershell -EncodedCommand SQBFAFgA\nSet-MpPreference -DisableRealtimeMonitoring $true\n",
                encoding="utf-8",
            )
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            response = registry.run_capability(
                "security_suite",
                "assess_path",
                {"target_path": str(target), "include_defender_status": False},
            )

        self.assertIn("critical risk", response)
        self.assertIn("Microsoft Defender", response)
        self.assertIn("encoded command", response.lower())

    def test_security_suite_flags_double_extension_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "invoice.pdf.exe"
            target.write_bytes(b"MZ" + b"\x00" * 4096)
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            response = registry.run_capability(
                "security_suite",
                "assess_path",
                {"target_path": str(target), "include_defender_status": False},
            )

        self.assertIn("Suspicious double extension", response)
        self.assertIn("invoice.pdf.exe", response)

    def test_security_suite_flags_macro_shortcut_and_archive_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Downloads"
            target.mkdir()
            (target / "proposal.docm").write_bytes(b"PK\x03\x04macro")
            (target / "meeting.url").write_text("[InternetShortcut]\nURL=https://example.com/file.exe", encoding="utf-8")
            with zipfile.ZipFile(target / "bundle.zip", "w") as archive:
                archive.writestr("invoice.pdf.exe", b"MZ")
                archive.writestr("scripts/run.ps1", "powershell -EncodedCommand SQBFAFgA")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                response = registry.run_capability(
                    "security_suite",
                    "assess_path",
                    {"target_path": str(target), "include_defender_status": False},
                    memory=memory,
                )
                artifact = json.loads(Path(memory.recent_tool_runs(limit=1)[0]["artifact_path"]).read_text(encoding="utf-8"))

        categories = {finding["category"] for finding in artifact["findings"]}
        self.assertIn("macro_document", categories)
        self.assertIn("shortcut_file", categories)
        self.assertIn("archive_contents", categories)
        self.assertIn("Archive contains", response)

    def test_security_suite_missing_path_returns_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            response = registry.run_capability("security_suite", "assess_path", {})

        self.assertIn("target_path is required", response)

    def test_security_suite_respects_max_files_and_reports_partial_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "many"
            target.mkdir()
            for index in range(3):
                (target / f"file{index}.txt").write_text(f"file {index}", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                response = registry.run_capability(
                    "security_suite",
                    "assess_path",
                    {"target_path": str(target), "max_files": 1, "include_defender_status": False},
                    memory=memory,
                )
                run = memory.recent_tool_runs(limit=1)[0]
            artifact = json.loads(Path(run["artifact_path"]).read_text(encoding="utf-8"))

        self.assertIn("max_files=1", response)
        self.assertEqual(artifact["summary"]["scanned_file_count"], 1)
        self.assertTrue(any(item["name"] == "file_collection" for item in artifact["skipped_checks"]))

    def test_security_suite_defender_unavailable_is_skipped_not_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "clean.txt"
            target.write_text("clean", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with patch("security_adapter.os.name", "nt"):
                with patch("security_adapter.subprocess.run", side_effect=FileNotFoundError("missing powershell")):
                    with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                        response = registry.run_capability(
                            "security_suite",
                            "assess_path",
                            {"target_path": str(target)},
                            memory=memory,
                        )
                        run = memory.recent_tool_runs(limit=1)[0]
            artifact = json.loads(Path(run["artifact_path"]).read_text(encoding="utf-8"))

        self.assertIn("local risk assessment complete", response)
        self.assertTrue(any(item["name"] == "windows_defender_status" for item in artifact["skipped_checks"]))
        self.assertEqual(run["status"], "success")

    def test_orchestrator_routes_natural_security_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "Downloads"
            target.mkdir()
            (target / "readme.txt").write_text("ordinary download", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                response, source = orchestrator.try_handle(f'scan "{target}" for suspicious files without defender')

        self.assertEqual(source, "ToolRegistry")
        self.assertIn("security_suite local risk assessment complete", response)

    def test_security_suite_history_and_explanation_use_recorded_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "clean"
            target.mkdir()
            (target / "notes.txt").write_text("ordinary project notes", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            registry.run_capability(
                "security_suite",
                "assess_path",
                {"target_path": str(target), "include_defender_status": False},
            )
            history = registry.run_capability("security_suite", "security_history", {"limit": 1})
            explanation = registry.run_capability("security_suite", "explain_assessment", {})

        self.assertIn("Recent local security assessments", history)
        self.assertIn(str(target.resolve()), history)
        self.assertIn("Local security assessment explanation", explanation)
        self.assertIn("low risk", explanation)

    def test_security_suite_project_audit_flags_sensitive_files_and_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "Project"
            project.mkdir()
            (project / "package.json").write_text('{"dependencies":{"left-pad":"1.0.0"}}', encoding="utf-8")
            (project / ".env").write_text("API_KEY='abcdefghijklmnop123456'\n", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                response = registry.run_capability(
                    "security_suite",
                    "audit_project",
                    {"target_path": str(project)},
                    memory=memory,
                )
                run = memory.recent_tool_runs(limit=1)[0]
            artifact = json.loads(Path(run["artifact_path"]).read_text(encoding="utf-8"))

        self.assertIn("project audit complete", response)
        self.assertIn("Sensitive configuration", response)
        self.assertEqual(artifact["metrics"]["manifest_count"], 1)
        self.assertGreaterEqual(artifact["metrics"]["risk_score"], 15)

    def test_security_suite_defender_scan_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "clean.txt"
            target.write_text("clean", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with patch("security_adapter.subprocess.run") as run_mock:
                response = registry.run_capability(
                    "security_suite",
                    "defender_scan",
                    {"target_path": str(target)},
                )

        self.assertIn("prepared, not executed", response)
        self.assertFalse(run_mock.called)

    def test_security_suite_defender_scan_runs_when_confirmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "clean.txt"
            target.write_text("clean", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with patch("security_adapter.subprocess.run") as run_mock:
                run_mock.return_value = subprocess.CompletedProcess(["powershell"], 0, stdout="", stderr="")
                response = registry.run_capability(
                    "security_suite",
                    "defender_scan",
                    {"target_path": str(target), "confirm_scan": True},
                )

        self.assertIn("custom scan completed", response)
        self.assertTrue(run_mock.called)

    def test_security_suite_quarantine_plan_does_not_move_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "suspicious.txt"
            target.write_text("do not move me", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            response = registry.run_capability(
                "security_suite",
                "quarantine_plan",
                {"target_path": str(target)},
            )
            self.assertIn("No files were moved or deleted", response)
            self.assertTrue(target.exists())

    def test_security_suite_quarantine_requires_confirmation_then_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "suspicious.txt"
            target.write_text("move me safely", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                planned = registry.run_capability(
                    "security_suite",
                    "quarantine_path",
                    {"target_path": str(target)},
                    memory=memory,
                )
                exists_after_plan = target.exists()
                moved = registry.run_capability(
                    "security_suite",
                    "quarantine_path",
                    {"target_path": str(target), "confirm_quarantine": True},
                    memory=memory,
                )
                exists_after_move = target.exists()
                quarantine_run = memory.recent_tool_runs(limit=1)[0]
                quarantine_artifact = json.loads(Path(quarantine_run["artifact_path"]).read_text(encoding="utf-8"))
                metadata_path = quarantine_artifact["metrics"]["metadata_path"]
                restore_planned = registry.run_capability(
                    "security_suite",
                    "restore_quarantine",
                    {"metadata_path": metadata_path},
                    memory=memory,
                )
                restored = registry.run_capability(
                    "security_suite",
                    "restore_quarantine",
                    {"metadata_path": metadata_path, "confirm_restore": True},
                    memory=memory,
                )
                exists_after_restore = target.exists()
                content_after_restore = target.read_text(encoding="utf-8") if target.exists() else ""

        self.assertIn("prepared, not executed", planned)
        self.assertTrue(exists_after_plan)
        self.assertIn("quarantine complete", moved.lower())
        self.assertFalse(exists_after_move)
        self.assertIn("restore prepared", restore_planned)
        self.assertIn("restore complete", restored)
        self.assertTrue(exists_after_restore)
        self.assertEqual(content_after_restore, "move me safely")

    def test_security_suite_rule_scan_uses_builtin_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "scripts"
            target.mkdir()
            (target / "install.ps1").write_text("powershell -EncodedCommand SQBFAFgA", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            response = registry.run_capability(
                "security_suite",
                "rule_scan",
                {"target_path": str(target)},
            )

        self.assertIn("local rule scan complete", response)
        self.assertIn("PowerShell encoded command", response)

    def test_security_suite_defender_history_parses_available_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            payload = json.dumps({"ThreatName": "DemoThreat", "ActionSuccess": True, "Resources": ["file:test"]})
            with patch("security_adapter.os.name", "nt"):
                with patch("security_adapter.subprocess.run") as run_mock:
                    run_mock.return_value = subprocess.CompletedProcess(["powershell"], 0, stdout=payload, stderr="")
                    response = registry.run_capability("security_suite", "defender_history", {"limit": 3})

        self.assertIn("Defender threat history read complete", response)
        self.assertIn("DemoThreat", response)

    def test_orchestrator_routes_project_security_audit_and_defender_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "Project"
            project.mkdir()
            (project / "package.json").write_text("{}", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                audit_response, audit_source = orchestrator.try_handle(f'audit project security for "{project}"')
                defender_response, defender_source = orchestrator.try_handle(f'prepare defender scan for "{project}"')

        self.assertEqual(audit_source, "ToolRegistry")
        self.assertIn("project audit complete", audit_response)
        self.assertEqual(defender_source, "ToolRegistry")
        self.assertIn("prepared, not executed", defender_response)

    def test_orchestrator_routes_quarantine_rule_scan_and_defender_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "suspicious.txt"
            target.write_text("powershell -EncodedCommand SQBFAFgA", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                quarantine_response, quarantine_source = orchestrator.try_handle(f'quarantine "{target}"')
                rule_response, rule_source = orchestrator.try_handle(f'run local security rules on "{target}"')
                history_response, history_source = orchestrator.try_handle("show defender threat history")

        self.assertEqual(quarantine_source, "ToolRegistry")
        self.assertIn("prepared, not executed", quarantine_response)
        self.assertEqual(rule_source, "ToolRegistry")
        self.assertIn("local rule scan complete", rule_response)
        self.assertEqual(history_source, "ToolRegistry")
        self.assertIn("Defender", history_response)


    def test_security_suite_updates_bayesian_when_probability_requested(self):
        class FakeBayesianAdapter:
            def add_tool_run_evidence(self, tool_name, artifact_path, ber_threshold=0.01):
                return {
                    "evidence": [
                        {"factor": "Security assessment completed successfully", "passed": True, "status": "added"},
                        {"factor": "Security assessment output is machine-readable", "passed": True, "status": "skipped", "reason": "duplicate evidence"},
                        {
                            "factor": "Security assessment found no high-risk indicators",
                            "passed": False,
                            "status": "added",
                            "description": "Risk score stayed below the high-risk threshold.",
                            "observed_value": 61,
                        },
                    ],
                    "assessment": {
                        "status": "success",
                        "probability": 0.71,
                        "confidence": 0.55,
                        "credible_interval": [0.4, 0.9],
                        "warnings": [],
                    },
                    "config": {},
                }

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "clean.txt"
            target.write_text("clean", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            registry.bayesian_adapter = FakeBayesianAdapter()
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                response, source = orchestrator.try_handle(f'scan "{target}" for suspicious files and give probability without defender')
                run = memory.recent_tool_runs(limit=1)[0]

        self.assertEqual(source, "ToolRegistry")
        self.assertIn("Bayesian security evidence updated", response)
        self.assertIn("Evidence added: 2; skipped: 1", response)
        self.assertIn("Evidence against reliability", response)
        self.assertEqual(run["evidence_status"], "updated")

    def test_memory_refresh_indexes_tool_artifacts_for_rag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Repo"
            root.mkdir()
            target = Path(tmp) / "scan"
            target.mkdir()
            (target / "invoice.pdf.exe").write_bytes(b"MZ" + b"\0" * 32)
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                registry.run_capability(
                    "security_suite",
                    "assess_path",
                    {"target_path": str(target), "include_defender_status": False},
                    memory=memory,
                )
                service = RuntimeServiceFacade(memory, registry)
                result = service.index_project_memory(project_names=["security_suite"], max_files_per_project=10)
                hits = memory.search("security risk invoice double extension detector recommended actions", project="tool:security_suite", limit=3)

        self.assertGreaterEqual(result["total_indexed_document_count"], 1)
        self.assertTrue(any({"double", "extension"} <= set(hit.get("matched_tokens", [])) for hit in hits))
        self.assertTrue(any(hit["source_kind"] == "tool_run" for hit in hits))

    def test_orchestrator_routes_cadconverter_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "CadConverter"
            root.mkdir()
            (root / "part.step").write_text("solid", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=root,
                cadconverter_output_dir=Path(tmp) / "outputs",
                cadconverter_python="python",
                folderguardian_root=Path("missing-folder"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                orchestrator = RuntimeOrchestrator(memory, registry)
                response, source = orchestrator.try_handle('recommend CADConverter workflow for "part.step" for AR viewing')

        self.assertEqual(source, "ToolRegistry")
        self.assertIn("Next action: convert_step_to_glb", response)

    def test_workspace_inspector_records_candidate_projects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "CPP"
            project = root / "RadioSim"
            project.mkdir(parents=True)
            (project / "CMakeLists.txt").write_text("project(RadioSim)", encoding="utf-8")
            (project / "main.cpp").write_text("int main(){return 0;}", encoding="utf-8")
            settings = SimpleNamespace(
                ofdm_root=Path("missing-ofdm"),
                ofdm_emi_root=Path("missing-emi"),
                cadconverter_root=Path("missing-cad"),
                folderguardian_root=Path("missing-folder"),
                cpp_workspace_root=root,
                python_workspace_root=Path("missing-python"),
                csharp_workspace_root=Path("missing-csharp"),
                tool_runs_dir=Path(tmp) / "tool_runs",
            )
            registry = create_default_registry(settings)
            with MemoryService(Path(tmp) / "memory.sqlite3") as memory:
                response = registry.run_capability("cpp_workspace", "inspect", {"max_projects": 5}, memory=memory)

        self.assertIn("RadioSim", response)
        self.assertIn("CMakeLists.txt", response)

    def test_cadconverter_metric_parser_extracts_validation_counts(self):
        metrics = _parse_cadconverter_metrics(
            "OK: file.glb\n  nodes=3, meshes=2, materials=1, primitives=4\n  colors=4/4, normals=4/4, named_nodes=3/3"
        )

        self.assertEqual(metrics["nodes"], 3)
        self.assertEqual(metrics["primitives"], 4)
        self.assertEqual(metrics["colored_primitives"], 4)


if __name__ == "__main__":
    unittest.main()
