import logging
import os
import socket
import sys
from concurrent import futures

import grpc

import ai_service_pb2
import ai_service_pb2_grpc
from embedding_service import create_embedding_provider
from memory_service import MemoryService
from model_runtime_service import RuntimeModelServiceFacade
from model_registry import create_default_model_registry
from orchestrator import RuntimeOrchestrator
from runtime_config import settings as runtime_settings
from runtime_grpc_services import register_runtime_grpc_services
from tool_registry import create_default_registry


if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiteAIService(ai_service_pb2_grpc.AIServiceServicer):
    """Compatibility AIService that exposes the runtime without heavy model/voice deps."""

    def __init__(self):
        self.embedding_provider = create_embedding_provider(runtime_settings)
        self.model_registry = create_default_model_registry(runtime_settings, self.embedding_provider)
        self.memory = MemoryService(runtime_settings.database_path, embedding_provider=self.embedding_provider)
        self.tool_registry = create_default_registry(runtime_settings)
        self.runtime_model_service = RuntimeModelServiceFacade(self.model_registry)
        self.runtime_orchestrator = RuntimeOrchestrator(self.memory, self.tool_registry, self.model_registry)
        self.runtime_orchestrator.bootstrap_project_memory()

    def ProcessQuery(self, request, context):
        raw_input = request.input_text or ""
        runtime_prefix = "__hivemind_runtime__:"
        silent_runtime_request = raw_input.startswith(runtime_prefix)
        user_input = raw_input[len(runtime_prefix):].strip() if silent_runtime_request else raw_input

        response_text, source = self.runtime_orchestrator.try_handle(user_input)
        if not response_text:
            if silent_runtime_request:
                response_text = "Unsupported runtime command."
            else:
                response_text = (
                    "Runtime-lite backend is active. I can list tools, search project memory, "
                    "run registered adapters, and show runtime health. Full chat/model/voice "
                    "responses need the full Python AI environment."
                )
            source = "RuntimeLite"

        if not silent_runtime_request:
            self.memory.add_conversation("runtime", "user", user_input)
            self.memory.add_conversation("runtime", "assistant", response_text)

        return ai_service_pb2.QueryResponse(response_text=response_text, ai_source=source)

    def GetSystemStatus(self, request, context):
        return ai_service_pb2.SystemStatus(
            greeting="Runtime-lite backend is running.",
            weather_report="Weather is unavailable in runtime-lite mode.",
            location="",
            temperature=0.0,
            conditions="unavailable",
            wind_speed=0.0,
            humidity=0,
            local_time="",
        )

    def RecognizeSpeech(self, request, context):
        return ai_service_pb2.QueryResponse(
            response_text="Speech recognition is unavailable in runtime-lite mode.",
            ai_source="RuntimeLite",
        )

    def StreamSpeechStatus(self, request, context):
        if False:
            yield ai_service_pb2.SpeechStatus(status="idle", message="runtime-lite")

    def InterruptSpeech(self, request, context):
        return ai_service_pb2.InterruptResponse(
            success=True,
            message="No speech is active in runtime-lite mode.",
        )


def find_available_port(start_port=50051, max_tries=100):
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
                probe.bind(("::", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_tries - 1}")


def serve():
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=(("grpc.so_reuseport", 1),),
    )
    ai_service = LiteAIService()
    ai_service_pb2_grpc.add_AIServiceServicer_to_server(ai_service, server)
    register_runtime_grpc_services(
        server,
        ai_service.runtime_orchestrator.runtime_service,
        ai_service.runtime_model_service,
    )

    port = find_available_port(runtime_settings.port, runtime_settings.port_scan_limit)
    bind_host = f"[{runtime_settings.host}]" if ":" in runtime_settings.host else runtime_settings.host
    server.add_insecure_port(f"{bind_host}:{port}")

    try:
        server.start()
        print(f"Server started successfully on port {port}", flush=True)
        logger.info("Runtime-lite gRPC server is running.")
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down runtime-lite server.")
        server.stop(0)
    finally:
        logger.info("Runtime-lite server stopped.")


if __name__ == "__main__":
    serve()
