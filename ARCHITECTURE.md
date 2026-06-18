# Hivemind Runtime Architecture

Hivemind is moving toward a local-first engineering runtime. The current app still exposes the existing `AIService` gRPC surface for UI compatibility, while new runtime responsibilities are being split into focused Python modules.

## Current Runtime Shape

- Electron/React remains the desktop shell and process launcher.
- Python still hosts the compatibility `AIService`, but now delegates runtime concerns to smaller services.
- VoiceBridge stays a separate C# process and receives the actual backend URL from Electron.
- Electron can start either the full `ai_server.py` backend or the lightweight `runtime_lite_server.py` compatibility backend. The lite backend keeps runtime tools, memory, adapters, and gRPC contracts testable even when the heavy local model/voice Python stack is not installed.
- Local LLMs should not all be startup blockers. WizardMath is disabled by default; math now routes to deterministic `MathService` first. Phi-3 can be preloaded with `HIVEMIND_EAGER_LOAD_PHI3=true` when local conversational reasoning is desired.
- Local model loading can pin revisions and fall back to a complete cached Hugging Face snapshot when `refs/main` is present but missing weight shards. This keeps a partial download from blocking the whole desktop runtime.
- The compatibility backend now exposes model lifecycle commands through the runtime command bridge: `model status`, `load phi`, and `unload phi`. This is a bridge toward real model-management gRPC methods.
- External projects remain standalone and are integrated through adapters instead of imports.

## Runtime Modules

- `runtime_config.py`: environment-backed configuration, secrets, ports, and external project paths.
- `memory_service.py`: SQLite-backed conversation memory, summaries, project/tool document indexing, chunked local vector retrieval, and tool-run history.
- `tool_registry.py`: capability metadata and safe adapters for OFDMSim, OFDMSim EMI, CADConverter, FolderGuardian, Security Suite, and Bayesian Engine.
- `security_adapter.py`: dependency-light local defensive file/folder risk assessment, artifact history, and evidence explanation.
- `intent_router.py`: deterministic first-pass classification for math, tool/runtime, engineering, long-form, and general conversation requests.
- `math_service.py`: deterministic arithmetic and optional SymPy-backed symbolic math before any math LLM is considered.
- `model_registry.py`: model metadata, availability, and compatibility routing for local chat models, deterministic services, cloud fallback, and embeddings.
- `model_runtime_service.py`: direct runtime boundary for model listing, status, and lifecycle controls.
- `bayesian_adapter.py`: invokes the external Bayesian Engine CLI as an evidence broker while keeping generated state under Hivemind.
- `runtime_artifacts.py`: writes and updates machine-readable tool-run artifacts for later memory and Bayesian evidence.
- `runtime_plans.py`: structured runtime execution plan records for routed actions.
- `runtime_service.py`: service-boundary facade for capability schemas, parameter validation, execution, memory search, project indexing, recent runs, and snapshots; future generated gRPC servicers should delegate here.
- `orchestrator.py`: compatibility orchestrator that can plan and answer tool, memory, health, safe-probe, OFDMSim simulation, CADConverter workflow, and Bayesian confidence requests before falling back to model routing.
- `voice_utils.py`: speech text normalization and chunking, including decimal-safe weather/TTS handling.

## Current Tool Routes

The compatibility orchestrator can currently respond to:

- `list tools`
- `schema for cadconverter.convert_step_to_glb`
- `list models`
- `model health`
- `plan run OFDMSim AWGN QPSK at 18 dB for 10 frames`
- `recent execution plans`
- `tool health`
- `search project memory ...`
- `refresh project memory`
- `run OFDMSim self test`
- `run OFDMSim AWGN QPSK MMSE at 18 dB for 100 frames seed 123`
- `compare OFDMSim EMI AWGN QPSK at 18 dB for 100 frames EMI probability 0.003 amplitude 15`
- `run OFDMSim AWGN QPSK MMSE at 18 dB for 100 frames and update Bayesian confidence`
- `inspect CADConverter files`
- `validate CADConverter GLB "python half.glb"`
- `validate CADConverter GLB "python half.glb" and give probability`
- `convert CADConverter STEP to GLB "python half.stp"`
- `convert CADConverter GLB to STEP "python half.glb" mode reconstructed`
- `inspect FolderGuardian project`
- `encrypt folder "D:\Temp\Demo" and give confidence`
- `scan "D:\Temp\Demo" for suspicious files`
- `check if "D:\Temp\invoice.pdf.exe" is malicious`
- `scan "D:\Temp\Demo" for suspicious files and give probability`
- `audit project security for "."`
- `prepare defender scan for "D:\Temp\Demo"`
- `prepare quarantine plan for "D:\Temp\invoice.pdf.exe"`
- `show recent security scans`
- `explain the last security assessment`
- `assess OFDMSim confidence`
- `Bayesian health`

OFDMSim simulation requests are normalized into explicit CLI flags, executed without a shell, captured, summarized, and stored in `tool_runs`.
When Bayesian confidence is requested, the tool artifact is attached as local evidence to a Hivemind-owned Bayesian factor-graph mission.
The adapter updates the same artifact after evidence processing so `evidence_status` moves from `pending` to `updated` or a failure/skipped reason. A compact `bayesian_assessment` is stored in the artifact without copying the full Bayesian report text. Evidence factors are tool-aware: OFDM uses BER quality checks, CADConverter uses execution/geometry/output-artifact checks, FolderGuardian uses bridge execution/summary/error-count checks, and EMI uses baseline BER checks.

OFDMSim EMI comparison requests run the standalone simulator's `compare` command with normalized EMI probability, amplitude, width, clipping threshold, SNR, frame, and seed parameters. The adapter parses the baseline/EMI/clipping CSV into structured BER metrics and stores them as runtime artifacts and searchable memory.

CADConverter requests are routed through `python -m cadconverter.cli` from the standalone CADConverter project root. Hivemind records stdout, stderr, parsed validation/reconstruction metrics, and a machine-readable artifact under its own runtime state. Default generated CAD outputs go under `AICoreClient/runtime_state/cadconverter_outputs` so adapter tests and routine conversions do not write into the external project unless an explicit allowed output path is provided.
CADConverter validation and conversion requests can now update Bayesian tool-confidence evidence when the user asks for probability, confidence, evidence, or reliability in the same natural-language request.

FolderGuardian is integrated through a Hivemind-owned `FolderGuardianBridge` sidecar. The bridge links FolderGuardian's standalone C# core files at build time and exposes headless `encrypt` and `decrypt` commands, so the external WPF app remains independent while Jarvis can run live folder protection workflows. Hivemind records every live bridge call as a runtime artifact with target path, command, stdout/stderr, parsed summary counts, and status. Dry-run planning remains available for non-mutating review requests, and watch/log helpers are still handled by the Hivemind adapter.
FolderGuardian live encrypt/decrypt requests can also update Bayesian confidence evidence when reliability/probability is requested.

Security Suite is a local-only defensive scanner for file, folder, and project risk assessment. It records SHA-256 inventories, extension mismatch checks, suspicious double extensions, script-pattern findings, entropy hints, executable/script counts, recent modification summaries, FolderGuardian log correlation, sensitive project config, simple secret patterns, dependency manifests, local rule-pack matches, Defender threat-history reads when available, and optional read-only Microsoft Defender status. Defender custom scan execution requires explicit confirmation because local Defender policy may remediate threats. Quarantine defaults to planning only; confirmed quarantine moves the target into Hivemind-owned quarantine storage, writes restore metadata with original path and hashes, and confirmed restore can move it back. It never deletes files or kills processes.
Security Suite artifacts can update Bayesian evidence when the user asks for confidence, probability, reliability, or Bayesian evidence. The Bayesian factors are security-specific: assessment completed, output is machine-readable, and local risk score stayed below the high-risk threshold.

## Memory And RAG

Memory now has three layers:

- structured records: conversations, summaries, tool runs, and project documents in SQLite
- chunked retrieval: READMEs/docs/tool outputs are split into retrieval chunks
- swappable local embeddings: deterministic hash embeddings are the dependency-free default; `HIVEMIND_EMBEDDING_PROVIDER=transformers` can point at a local Hugging Face model path later

Tool-run records are also indexed back into project memory under `tool:<name>`, which lets later questions retrieve previous metrics, artifacts, errors, and outputs without duplicating memory inside each tool.
Project memory refresh also indexes recent JSON artifacts for each registered tool, including Security Suite risk scans and project audits, as compact searchable records instead of dumping raw artifact JSON into RAG context.
Each chunk stores embedding provider, dimensions, and content hash metadata so the runtime can migrate providers without changing the rest of the memory interface.
Project memory refresh indexes each registered tool's README, docs folder, and common project manifests such as `.csproj`, `.sln`, `CMakeLists.txt`, `package.json`, `pyproject.toml`, and `.proto` files. It also writes a compact `__source_summary__.md` per project when source files are present, listing source paths, extension counts, and detected classes/functions/services. This gives RAG enough structure to understand standalone projects without copying their implementation into Hivemind.
Retrieval ranking now combines lexical IDF, local embedding similarity, phrase/title/project bonuses, source-kind bonuses, and per-source diversification so one noisy file or artifact does not flood the context. Runtime panel memory search exposes explicit project-doc and `tool:<name>` tool-run scopes.

## Model And Intent Routing

Model routing is represented as registry metadata plus deterministic intent decisions instead of keyword checks inside `AIService`.
Current routes are:

- `runtime_orchestrator`: rule/registry router for runtime, tools, and memory
- `math_service`: deterministic arithmetic, algebra, calculus, and equation solving; uses SymPy when available
- `phi3`: local short general reasoning route
- `mistral`: cloud fallback when configured
- `wizardmath`: legacy local math model, disabled by default
- `embedding`: active memory/RAG embedding provider

The registry does not make every external program run its own LLM. It describes available model roles centrally so specialist models can be added deliberately where they help. Phi-3 is useful for explanation and general engineering reasoning, but it is not trusted as the primary router because it can misclassify tool requests. Tool and math routing stay deterministic.

Runtime command bridge model controls:

- `model status`
- `load phi`
- `unload phi`

## Runtime Plans

Routed runtime actions now create lightweight structured execution plans. A plan records intent, source, steps, status, timestamps, and result summary. Completed/failed plans are persisted in SQLite and surfaced in runtime snapshots. This is intentionally smaller than a full workflow engine for now: it gives the UI and future gRPC runtime service a stable trace of what the orchestrator decided to do without forcing every feature into a heavy agent loop.

Plan steps currently cover registry inspection, memory retrieval, capability resolution, facade-level parameter validation, adapter execution, artifact writing, and memory storage. Tool execution now routes through `RuntimeServiceFacade.run_capability` so compatibility chat routes and future gRPC runtime routes share the same validation boundary.

The Runtime panel can run registered capabilities from their schemas. It renders basic controls for enum, boolean, number, integer, JSON/list, and text/path fields, then sends capability execution through the direct `RuntimeService.RunCapability` gRPC client. Runtime snapshot data, capability metadata, tool history, execution-plan history, memory stats, memory indexing, memory search, model listing, and model lifecycle controls now use direct JS clients for `RuntimeService`, `RuntimeMemoryService`, and `RuntimeModelService`.

## Contract Direction

`AICoreClient/Protos/runtime_contract.proto` defines the service split for `RuntimeService`, `RuntimeMemoryService`, `RuntimeModelService`, and future `VoiceService`. Python bindings are generated into `AICoreClient/PythonServer/runtime_contract_pb2.py` and `runtime_contract_pb2_grpc.py`, and the Python server now registers real `RuntimeService`, `RuntimeMemoryService`, and `RuntimeModelService` implementations next to the compatibility `AIService`.

Electron initializes direct JS gRPC clients for `RuntimeService`, `RuntimeMemoryService`, and `RuntimeModelService` next to the existing compatibility `AIService` client. The runtime gRPC capability contract includes tool root, runtime type, availability, examples, and tool description metadata so the UI can build its runtime snapshot without asking the chat bridge. Direct runtime gRPC clients can now call capability listing, schema lookup, capability execution, memory search, memory indexing, recent tool runs, recent plans, memory stats, model listing, Phi-3 load/unload, and runtime health.

## Voice Direction

VoiceBridge now captures microphone input as live PCM chunks with silence detection instead of a fixed five-second WAV capture. In streaming mode it sends chunks through `VoiceService.StreamRecognizeSpeech`, which emits lifecycle partials such as `listening`, `audio_started`, `speech_detected`, and `recognizing`, followed by a final/error event. Recognition is still final-result based because the current Google recognizer path does not provide true token-level partial transcription. Stop/interrupt polling has been tightened for faster speech cancellation. Punjabi improvements are intentionally parked for now.

## Configuration

Secrets are read from environment variables instead of source:

- `HIVEMIND_MISTRAL_API_KEY`
- `HIVEMIND_OPENWEATHER_API_KEY`
- `HIVEMIND_WIZARDMATH_MODEL`
- `HIVEMIND_EAGER_LOAD_WIZARDMATH`
- `HIVEMIND_PHI3_MODEL`
- `HIVEMIND_EAGER_LOAD_PHI3`
- `HIVEMIND_EMBEDDING_PROVIDER`
- `HIVEMIND_EMBEDDING_MODEL`
- `HIVEMIND_BAYESIAN_ENGINE_ROOT`
- `HIVEMIND_BAYESIAN_PYTHON`
- `HIVEMIND_BAYESIAN_STATE_DIR`

See `.env.example` for the full list of runtime settings.

Generated runtime state is ignored by git and lives under `AICoreClient/runtime_state/` by default.
Bayesian Engine is invoked from its standalone project root, but Hivemind passes absolute model, mission, and evidence paths under `AICoreClient/runtime_state/bayesian`. If the configured Bayesian venv Python cannot be launched, the adapter falls back to the Python currently running Hivemind, which works for the current dependency-free Bayesian Engine package.

## Verified Smoke Path

The current verified end-to-end runtime path is:

1. `run OFDMSim self test`
2. `run OFDMSim AWGN QPSK MMSE at 18 dB for 10 frames seed 42`
3. `run OFDMSim AWGN QPSK MMSE at 18 dB for 10 frames seed 42 and update Bayesian confidence`
4. `compare OFDMSim EMI AWGN QPSK at 18 dB for 100 frames EMI probability 0.003 amplitude 15`
5. `assess OFDMSim confidence`
6. Runtime panel memory search for `OFDM BER quiet run`

This produces:

- tool artifacts under `AICoreClient/runtime_state/tool_runs/ofdmsim`
- EMI comparison artifacts under `AICoreClient/runtime_state/tool_runs/ofdmsim_emi`
- searchable `tool_runs` records in SQLite
- Bayesian model/mission/evidence files under `AICoreClient/runtime_state/bayesian`
- execution plan records visible through `recent execution plans`
