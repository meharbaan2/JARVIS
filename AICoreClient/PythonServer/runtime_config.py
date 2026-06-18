import os
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]


def _env_path(name, default):
    value = os.getenv(name) or default
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _env_executable(name, default):
    value = os.getenv(name, default)
    if not any(separator in value for separator in ("\\", "/")) and ":" not in value:
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else REPO_ROOT / path)


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeConfig:
    mistral_api_key: str = field(default_factory=lambda: os.getenv("HIVEMIND_MISTRAL_API_KEY", ""))
    mistral_model: str = field(default_factory=lambda: os.getenv("HIVEMIND_MISTRAL_MODEL", "mistral-large-latest"))
    wizardmath_model: str = field(default_factory=lambda: os.getenv("HIVEMIND_WIZARDMATH_MODEL", "WizardLMTeam/WizardMath-7B-V1.1"))
    wizardmath_revision: str = field(default_factory=lambda: os.getenv("HIVEMIND_WIZARDMATH_REVISION", ""))
    phi3_model: str = field(default_factory=lambda: os.getenv("HIVEMIND_PHI3_MODEL", "microsoft/Phi-3-mini-4k-instruct"))
    phi3_revision: str = field(default_factory=lambda: os.getenv("HIVEMIND_PHI3_REVISION", ""))
    eager_load_wizardmath: bool = field(default_factory=lambda: _env_bool("HIVEMIND_EAGER_LOAD_WIZARDMATH", False))
    eager_load_phi3: bool = field(default_factory=lambda: _env_bool("HIVEMIND_EAGER_LOAD_PHI3", False))
    transformers_local_files_only: bool = field(default_factory=lambda: _env_bool("HIVEMIND_TRANSFORMERS_LOCAL_FILES_ONLY", False))
    local_snapshot_fallback: bool = field(default_factory=lambda: _env_bool("HIVEMIND_LOCAL_SNAPSHOT_FALLBACK", True))
    weather_api_key: str = field(default_factory=lambda: os.getenv("HIVEMIND_OPENWEATHER_API_KEY", ""))
    punjabi_reply_mode: str = field(default_factory=lambda: os.getenv("HIVEMIND_PUNJABI_REPLY_MODE", "local"))
    database_path: Path = field(default_factory=lambda: _env_path("HIVEMIND_MEMORY_DB", "AICoreClient/runtime_state/ai_memory.db"))
    embedding_provider: str = field(default_factory=lambda: os.getenv("HIVEMIND_EMBEDDING_PROVIDER", "hash"))
    embedding_model: str = field(default_factory=lambda: os.getenv("HIVEMIND_EMBEDDING_MODEL", ""))
    embedding_device: str = field(default_factory=lambda: os.getenv("HIVEMIND_EMBEDDING_DEVICE", "auto"))
    embedding_max_length: int = field(default_factory=lambda: int(os.getenv("HIVEMIND_EMBEDDING_MAX_LENGTH", "512")))
    embedding_hash_dimensions: int = field(default_factory=lambda: int(os.getenv("HIVEMIND_EMBEDDING_HASH_DIMENSIONS", "96")))
    host: str = field(default_factory=lambda: os.getenv("HIVEMIND_GRPC_HOST", "::"))
    port: int = field(default_factory=lambda: int(os.getenv("HIVEMIND_GRPC_PORT", "50051")))
    port_scan_limit: int = field(default_factory=lambda: int(os.getenv("HIVEMIND_GRPC_PORT_SCAN_LIMIT", "100")))
    runtime_state_dir: Path = field(default_factory=lambda: _env_path("HIVEMIND_RUNTIME_STATE_DIR", "AICoreClient/runtime_state"))
    tool_runs_dir: Path = field(default_factory=lambda: _env_path("HIVEMIND_TOOL_RUNS_DIR", "AICoreClient/runtime_state/tool_runs"))
    cadconverter_output_dir: Path = field(default_factory=lambda: _env_path("HIVEMIND_CADCONVERTER_OUTPUT_DIR", "AICoreClient/runtime_state/cadconverter_outputs"))

    ofdm_root: Path = field(default_factory=lambda: _env_path("HIVEMIND_OFDMSIM_ROOT", "external/OFDMSim"))
    ofdm_emi_root: Path = field(default_factory=lambda: _env_path("HIVEMIND_OFDMSIM_EMI_ROOT", "external/OFDMSim_EMI"))
    cadconverter_root: Path = field(default_factory=lambda: _env_path("HIVEMIND_CADCONVERTER_ROOT", "external/CadConverter"))
    cadconverter_python: str = field(default_factory=lambda: _env_executable("HIVEMIND_CADCONVERTER_PYTHON", "python"))
    folderguardian_root: Path = field(default_factory=lambda: _env_path("HIVEMIND_FOLDERGUARDIAN_ROOT", "external/FolderGuardian"))
    folderguardian_bridge_exe: Path = field(default_factory=lambda: _env_path("HIVEMIND_FOLDERGUARDIAN_BRIDGE_EXE", r"FolderGuardianBridge\bin\Debug\net8.0-windows\FolderGuardianBridge.exe"))
    folderguardian_live_actions: bool = field(default_factory=lambda: _env_bool("HIVEMIND_FOLDERGUARDIAN_LIVE_ACTIONS", True))
    cpp_workspace_root: Path = field(default_factory=lambda: _env_path("HIVEMIND_CPP_WORKSPACE_ROOT", "external/cpp"))
    python_workspace_root: Path = field(default_factory=lambda: _env_path("HIVEMIND_PYTHON_WORKSPACE_ROOT", "external/python"))
    csharp_workspace_root: Path = field(default_factory=lambda: _env_path("HIVEMIND_CSHARP_WORKSPACE_ROOT", "external/csharp"))
    bayesian_engine_root: Path = field(default_factory=lambda: _env_path("HIVEMIND_BAYESIAN_ENGINE_ROOT", "external/BayesianEngine"))
    bayesian_python: str = field(default_factory=lambda: _env_executable("HIVEMIND_BAYESIAN_PYTHON", "python"))
    bayesian_state_dir: Path = field(default_factory=lambda: _env_path("HIVEMIND_BAYESIAN_STATE_DIR", "AICoreClient/runtime_state/bayesian"))
    bayesian_ber_threshold: float = field(default_factory=lambda: float(os.getenv("HIVEMIND_BAYESIAN_BER_THRESHOLD", "0.01")))
    bayesian_evidence_dedupe: bool = field(default_factory=lambda: _env_bool("HIVEMIND_BAYESIAN_EVIDENCE_DEDUPE", True))
    bayesian_max_evidence_per_factor: int = field(default_factory=lambda: int(os.getenv("HIVEMIND_BAYESIAN_MAX_EVIDENCE_PER_FACTOR", "12")))
    bayesian_success_strength: str = field(default_factory=lambda: os.getenv("HIVEMIND_BAYESIAN_SUCCESS_STRENGTH", "medium"))
    bayesian_machine_readable_strength: str = field(default_factory=lambda: os.getenv("HIVEMIND_BAYESIAN_MACHINE_READABLE_STRENGTH", "medium"))
    bayesian_quality_strength: str = field(default_factory=lambda: os.getenv("HIVEMIND_BAYESIAN_QUALITY_STRENGTH", "weak"))
    bayesian_tool_confidence_factors_json: str = field(default_factory=lambda: os.getenv("HIVEMIND_BAYESIAN_TOOL_CONFIDENCE_FACTORS_JSON", ""))


settings = RuntimeConfig()
