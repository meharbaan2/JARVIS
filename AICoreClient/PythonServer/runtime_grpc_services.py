import json
import uuid

import grpc

import runtime_contract_pb2
import runtime_contract_pb2_grpc
import ai_service_pb2


def register_runtime_grpc_services(server, runtime_service, model_service=None, voice_service=None):
    runtime_contract_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeGrpcService(runtime_service),
        server,
    )
    runtime_contract_pb2_grpc.add_RuntimeMemoryServiceServicer_to_server(
        RuntimeMemoryGrpcService(runtime_service),
        server,
    )
    if model_service:
        runtime_contract_pb2_grpc.add_RuntimeModelServiceServicer_to_server(
            RuntimeModelGrpcService(model_service),
            server,
        )
    if voice_service:
        runtime_contract_pb2_grpc.add_VoiceServiceServicer_to_server(
            VoiceGrpcService(voice_service),
            server,
        )


class RuntimeGrpcService(runtime_contract_pb2_grpc.RuntimeServiceServicer):
    def __init__(self, runtime_service):
        self.runtime_service = runtime_service

    def ListCapabilities(self, request, context):
        return runtime_contract_pb2.ListCapabilitiesResponse(
            capabilities=[
                _capability_message(capability)
                for capability in self.runtime_service.list_capabilities()
            ]
        )

    def GetCapabilitySchema(self, request, context):
        try:
            schema = self.runtime_service.get_capability_schema(
                request.tool_name,
                request.capability_name,
            )
        except KeyError as exc:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(exc))
            return runtime_contract_pb2.CapabilitySchema()
        return runtime_contract_pb2.CapabilitySchema(capability=_capability_message(schema))

    def RunCapability(self, request, context):
        try:
            params = json.loads(request.input_json or "{}")
        except json.JSONDecodeError as exc:
            return runtime_contract_pb2.RunCapabilityResponse(
                run=runtime_contract_pb2.RunRef(run_id=uuid.uuid4().hex[:12]),
                status="failed",
                output_json=json.dumps({"text": "input_json must be a JSON object"}),
                error_message=f"input_json must be a JSON object: {exc}",
            )
        if not isinstance(params, dict):
            return runtime_contract_pb2.RunCapabilityResponse(
                run=runtime_contract_pb2.RunRef(run_id=uuid.uuid4().hex[:12]),
                status="failed",
                output_json=json.dumps({"text": "input_json must be a JSON object"}),
                error_message="input_json must be a JSON object.",
            )

        result = self.runtime_service.run_capability(
            request.capability.tool_name,
            request.capability.capability_name,
            params,
        )
        return runtime_contract_pb2.RunCapabilityResponse(
            run=runtime_contract_pb2.RunRef(run_id=uuid.uuid4().hex[:12]),
            status=result["status"],
            output_json=result["output_json"],
            error_message=result.get("error_message", ""),
        )

    def StreamCapabilityEvents(self, request, context):
        yield runtime_contract_pb2.CapabilityEvent(
            run=request,
            type="status",
            message="Capability event streaming is not implemented for compatibility adapters yet.",
            progress=1.0,
        )

    def CancelRun(self, request, context):
        return runtime_contract_pb2.CancelRunResponse(
            success=False,
            message="Cancellation is not implemented for compatibility adapters yet.",
        )

    def HealthCheck(self, request, context):
        snapshot = self.runtime_service.runtime_snapshot()
        return runtime_contract_pb2.HealthCheckResponse(
            status="ok",
            details=[
                f"tools={snapshot['tool_count']}",
                f"available_tools={snapshot['available_tool_count']}",
                f"tool_runs={snapshot['memory'].get('tool_run_count', 0)}",
                f"plans={snapshot['memory'].get('execution_plan_count', 0)}",
            ],
        )


class RuntimeMemoryGrpcService(runtime_contract_pb2_grpc.RuntimeMemoryServiceServicer):
    def __init__(self, runtime_service):
        self.runtime_service = runtime_service

    def SearchMemory(self, request, context):
        hits = self.runtime_service.search_memory(
            request.query,
            project=request.project or None,
            limit=request.limit or 5,
        )
        return runtime_contract_pb2.MemorySearchResponse(
            hits=[
                runtime_contract_pb2.MemoryHit(
                    project=hit.get("project", ""),
                    source_path=str(hit.get("source_path", "")),
                    title=hit.get("title") or "",
                    chunk_index=int(hit.get("chunk_index") or 0),
                    snippet=hit.get("snippet", ""),
                    score=float(hit.get("score") or 0.0),
                )
                for hit in hits
            ]
        )

    def IndexProjectMemory(self, request, context):
        result = self.runtime_service.index_project_memory(
            project_names=list(request.project_names),
            max_files_per_project=request.max_files_per_project or 24,
            include_source_summary=True,
        )
        return runtime_contract_pb2.IndexProjectMemoryResponse(
            results=[
                runtime_contract_pb2.ProjectIndexResult(
                    project=item["project"],
                    root=item["root"],
                    indexed_document_count=int(item["indexed_document_count"]),
                    status=item["status"],
                )
                for item in result["results"]
            ],
            total_indexed_document_count=int(result["total_indexed_document_count"]),
        )

    def RecentToolRuns(self, request, context):
        runs = self.runtime_service.recent_tool_runs(limit=request.limit or 8)
        return runtime_contract_pb2.RecentToolRunsResponse(
            runs=[
                runtime_contract_pb2.ToolRunRecord(
                    tool_name=run.get("tool_name", ""),
                    capability_name=run.get("capability", ""),
                    status=run.get("status", ""),
                    input_json=run.get("input_json") or "",
                    artifact_path=run.get("artifact_path") or "",
                    metrics_json=run.get("metrics_json") or "",
                    evidence_status=run.get("evidence_status") or "",
                    started_at=run.get("started_at") or "",
                    finished_at=run.get("finished_at") or "",
                )
                for run in runs
            ]
        )

    def RecentExecutionPlans(self, request, context):
        plans = self.runtime_service.recent_execution_plans(limit=request.limit or 8)
        return runtime_contract_pb2.RecentExecutionPlansResponse(
            plans=[
                runtime_contract_pb2.ExecutionPlanRecord(
                    id=plan.get("id", ""),
                    intent=plan.get("intent", ""),
                    source=plan.get("source", ""),
                    status=plan.get("status", ""),
                    created_at=plan.get("created_at", ""),
                    finished_at=plan.get("finished_at", ""),
                    result_summary=plan.get("result_summary", ""),
                    error=plan.get("error", ""),
                    steps_json=json.dumps(plan.get("steps", []), sort_keys=True),
                )
                for plan in plans
            ]
        )

    def GetMemoryStats(self, request, context):
        return runtime_contract_pb2.MemoryStatsResponse(
            stats_json=json.dumps(self.runtime_service.memory_stats(), sort_keys=True)
        )


class RuntimeModelGrpcService(runtime_contract_pb2_grpc.RuntimeModelServiceServicer):
    def __init__(self, model_service):
        self.model_service = model_service

    def ListModels(self, request, context):
        snapshot = self.model_service.list_models()
        return runtime_contract_pb2.ListModelsResponse(
            models=[_model_message(model) for model in snapshot.get("models", [])],
            model_count=int(snapshot.get("model_count") or 0),
            available_model_count=int(snapshot.get("available_model_count") or 0),
        )

    def GetModelStatus(self, request, context):
        result = self.model_service.get_model_status(request.model_name)
        return runtime_contract_pb2.ModelStatusResponse(
            model=_model_message(result.get("model")),
            found=bool(result.get("found")),
            message=result.get("message", ""),
        )

    def LoadModel(self, request, context):
        result = self.model_service.load_model(request.model_name)
        return runtime_contract_pb2.ModelActionResponse(
            model=_model_message(result.get("model")),
            success=bool(result.get("success")),
            status=result.get("status", ""),
            message=result.get("message", ""),
        )

    def UnloadModel(self, request, context):
        result = self.model_service.unload_model(request.model_name)
        return runtime_contract_pb2.ModelActionResponse(
            model=_model_message(result.get("model")),
            success=bool(result.get("success")),
            status=result.get("status", ""),
            message=result.get("message", ""),
        )

    def ProbeModel(self, request, context):
        result = self.model_service.probe_model(
            request.model_name,
            prompts=[
                {
                    "category": prompt.category,
                    "prompt": prompt.prompt,
                    "expected_hint": prompt.expected_hint,
                }
                for prompt in request.prompts
            ],
        )
        return runtime_contract_pb2.ModelProbeResponse(
            model=_model_message(result.get("model")),
            success=bool(result.get("success")),
            status=result.get("status", ""),
            message=result.get("message", ""),
            results=[
                runtime_contract_pb2.ModelProbeResult(
                    category=item.get("category", ""),
                    prompt=item.get("prompt", ""),
                    response_text=item.get("response_text", ""),
                    expected_hint=item.get("expected_hint", ""),
                    status=item.get("status", ""),
                    warning=item.get("warning", ""),
                    latency_ms=float(item.get("latency_ms") or 0.0),
                )
                for item in result.get("results", [])
            ],
        )


class VoiceGrpcService(runtime_contract_pb2_grpc.VoiceServiceServicer):
    def __init__(self, ai_service):
        self.ai_service = ai_service

    def StreamRecognizeSpeech(self, request_iterator, context):
        language_code = "en-US"
        sample_rate = 16000
        chunks = []
        chunk_count = 0
        total_bytes = 0
        speech_detected = False
        yield runtime_contract_pb2.SpeechRecognitionEvent(
            type="partial",
            text="listening",
            language_code=language_code,
            confidence=0.0,
        )
        try:
            for chunk in request_iterator:
                if chunk.language_code:
                    language_code = chunk.language_code
                if chunk.sample_rate:
                    sample_rate = chunk.sample_rate
                if chunk.audio_data:
                    audio_data = bytes(chunk.audio_data)
                    chunks.append(audio_data)
                    chunk_count += 1
                    total_bytes += len(audio_data)
                    if chunk_count == 1:
                        yield runtime_contract_pb2.SpeechRecognitionEvent(
                            type="partial",
                            text="audio_started",
                            language_code=language_code,
                            confidence=0.1,
                        )
                    if not speech_detected and _pcm16_peak(audio_data) >= 500:
                        speech_detected = True
                        yield runtime_contract_pb2.SpeechRecognitionEvent(
                            type="partial",
                            text="speech_detected",
                            language_code=language_code,
                            confidence=0.35,
                        )
                    elif chunk_count % 10 == 0:
                        captured_seconds = total_bytes / max(1, sample_rate * 2)
                        yield runtime_contract_pb2.SpeechRecognitionEvent(
                            type="partial",
                            text=f"capturing {captured_seconds:.1f}s",
                            language_code=language_code,
                            confidence=0.2,
                        )
                if chunk.end_of_utterance:
                    break
            if not chunks:
                yield runtime_contract_pb2.SpeechRecognitionEvent(
                    type="error",
                    text="No audio captured.",
                    language_code=language_code,
                    confidence=0.0,
                )
                return

            yield runtime_contract_pb2.SpeechRecognitionEvent(
                type="partial",
                text="recognizing",
                language_code=language_code,
                confidence=0.45 if speech_detected else 0.2,
            )
            request = ai_service_pb2.AudioRequest(
                audio_data=b"".join(chunks),
                language_code=language_code,
                sample_rate=sample_rate,
            )
            response = self.ai_service.RecognizeSpeech(request, context)
            event_type = "final" if response.ai_source != "System" else "error"
            yield runtime_contract_pb2.SpeechRecognitionEvent(
                type=event_type,
                text=response.response_text,
                language_code=language_code,
                confidence=1.0 if event_type == "final" else 0.0,
            )
        except Exception as exc:
            yield runtime_contract_pb2.SpeechRecognitionEvent(
                type="error",
                text=str(exc),
                language_code=language_code,
                confidence=0.0,
            )

    def Speak(self, request, context):
        text = (request.text or "").strip()
        if not text:
            return runtime_contract_pb2.SpeakResponse(accepted=False, message="No text supplied.")
        self.ai_service._speak_response(text)
        return runtime_contract_pb2.SpeakResponse(accepted=True, message="Speech started.")

    def InterruptSpeech(self, request, context):
        response = self.ai_service.InterruptSpeech(ai_service_pb2.Empty(), context)
        return runtime_contract_pb2.InterruptSpeechResponse(
            success=bool(response.success),
            message=response.message,
        )


def _pcm16_peak(audio_data):
    peak = 0
    for index in range(0, len(audio_data) - 1, 2):
        sample = int.from_bytes(audio_data[index : index + 2], byteorder="little", signed=True)
        peak = max(peak, abs(sample))
    return peak


def _capability_message(capability):
    return runtime_contract_pb2.Capability(
        tool_name=capability["tool_name"],
        capability_name=capability["capability_name"],
        description=capability.get("description", ""),
        input_schema_json=json.dumps(capability.get("input_schema", {}), sort_keys=True),
        safety_level=capability.get("safety_level", ""),
        supports_progress=bool(capability.get("supports_progress")),
        supports_cancel=bool(capability.get("supports_cancel")),
        available=bool(capability.get("available")),
        runtime=capability.get("runtime", ""),
        root=capability.get("root", ""),
        examples_json=json.dumps(capability.get("examples", []), sort_keys=True),
        tool_description=capability.get("tool_description", ""),
    )


def _model_message(model):
    model = model or {}
    return runtime_contract_pb2.ModelRecord(
        name=model.get("name", ""),
        provider=model.get("provider", ""),
        kind=model.get("kind", ""),
        purpose=model.get("purpose", ""),
        model_name=model.get("model_name", ""),
        available=bool(model.get("available")),
        status=model.get("status", ""),
        priority=int(model.get("priority") or 0),
        metadata_json=json.dumps(model.get("metadata", {}), sort_keys=True),
    )
