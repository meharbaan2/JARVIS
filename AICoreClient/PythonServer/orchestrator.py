import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from runtime_plans import RuntimePlanStep, make_plan, summarize_result
from runtime_service import RuntimeServiceFacade


class RuntimeOrchestrator:
    """Small compatibility orchestrator used by the existing AIService."""

    def __init__(self, memory, tool_registry, model_registry=None):
        self.memory = memory
        self.tool_registry = tool_registry
        self.model_registry = model_registry
        self.runtime_service = RuntimeServiceFacade(memory, tool_registry, model_registry)
        self.recent_plans = []

    def bootstrap_project_memory(self):
        indexed = self.runtime_service.index_project_memory()
        return {
            result["project"]: result["indexed_document_count"]
            for result in indexed["results"]
        }

    def try_handle(self, text):
        lowered = (text or "").lower()
        stripped = lowered.strip()

        if stripped in {"runtime snapshot", "runtime snapshot json", "runtime status json"}:
            return json.dumps(self.snapshot(), indent=2), "RuntimeOrchestrator"

        if any(phrase in lowered for phrase in ["capability schema", "schema for"]):
            schema = self._capability_schema_text(text)
            if schema:
                plan = self._metadata_plan(text, "RuntimeServiceFacade", "Return one registered capability schema.")
                return self._run_with_plan(plan, lambda: schema), "RuntimeOrchestrator"

        if stripped.startswith(("run capability ", "execute capability ")):
            request = self._parse_generic_capability_request(text)
            if request.get("error"):
                return request["error"], "RuntimeOrchestrator"
            plan = self._tool_plan(
                text,
                request["tool_name"],
                request["capability_name"],
                request["params"],
                include_memory=True,
            )
            return self._run_with_plan(
                plan,
                lambda: self._run_capability_text(
                    request["tool_name"],
                    request["capability_name"],
                    request["params"],
                ),
            ), "ToolRegistry"

        if "recent execution plans" in lowered or "runtime plans" in lowered:
            plans = self.memory.recent_execution_plans(limit=8)
            if not plans and self.recent_plans:
                plans = [plan.to_dict() for plan in self.recent_plans[-8:]]
            if not plans:
                return "No runtime execution plans have been recorded yet.", "RuntimeOrchestrator"
            lines = ["Recent runtime execution plans:"]
            lines.extend(
                f"- {plan['id']}: {plan['intent']} [{plan['status']}] via {plan['source']}"
                for plan in plans
            )
            return "\n".join(lines), "RuntimeOrchestrator"

        if stripped.startswith("plan "):
            plan = self.plan_for(text[5:].strip())
            if not plan:
                return "I do not have a structured runtime plan for that request yet.", "RuntimeOrchestrator"
            return json.dumps(plan.to_dict(), indent=2), "RuntimeOrchestrator"

        if any(phrase in lowered for phrase in ["list tools", "available tools", "what tools", "capabilities"]):
            plan = self._metadata_plan(text, "ToolRegistry", "List registered tool capabilities.")
            return self._run_with_plan(plan, self._capability_list_text), "RuntimeOrchestrator"

        if any(phrase in lowered for phrase in ["list models", "model health", "model status", "available models"]):
            if not self.model_registry:
                return "Model registry is not configured yet.", "RuntimeOrchestrator"
            plan = self._metadata_plan(text, "ModelRegistry", "List configured model routes.")
            return self._run_with_plan(plan, self.model_registry.describe), "RuntimeOrchestrator"

        if "tool health" in lowered or "runtime health" in lowered:
            plan = self._metadata_plan(text, "ToolRegistry", "Check registered tool health.")
            return self._run_with_plan(plan, self._tool_health_text), "RuntimeOrchestrator"

        if "bayesian" in lowered and "health" in lowered:
            plan = self._tool_plan(text, "bayesian_engine", "health", {}, include_memory=False)
            return self._run_with_plan(
                plan,
                lambda: self._run_capability_text("bayesian_engine", "health", {}),
            ), "BayesianEngine"

        if any(phrase in lowered for phrase in ["refresh project memory", "index project memory", "bootstrap project memory"]):
            plan = self._memory_plan(text, action="index")
            return self._run_with_plan(plan, self._refresh_project_memory_text), "MemoryService"

        if "search memory" in lowered or "project memory" in lowered:
            plan = self._memory_plan(text)
            return self._run_with_plan(plan, lambda: self._memory_search_text(text)), "MemoryService"

        if "recent tool runs" in lowered or "tool run history" in lowered:
            runs = self.memory.recent_tool_runs(limit=8)
            if not runs:
                return "No tool runs have been recorded yet.", "MemoryService"
            lines = ["Recent tool runs:"]
            for run in runs:
                lines.append(
                    f"- {run['tool_name']}.{run['capability']}: {run['status']} "
                    f"at {run['finished_at'] or run['started_at']}"
                )
            return "\n".join(lines), "MemoryService"

        tool_route = self._route_tool_request(text)
        if tool_route:
            if tool_route.get("error"):
                return tool_route["error"], tool_route.get("source", "RuntimeOrchestrator")
            plan = self._tool_plan(
                text,
                tool_route["tool_name"],
                tool_route["capability_name"],
                tool_route["params"],
                include_memory=tool_route.get("include_memory", True),
            )
            return self._run_with_plan(
                plan,
                lambda: self._run_capability_text(
                    tool_route["tool_name"],
                    tool_route["capability_name"],
                    tool_route["params"],
                ),
            ), tool_route["source"]

        if "ofdmsim" in lowered and "self" in lowered and "test" in lowered:
            plan = self._tool_plan(text, "ofdmsim", "selftest", {}, include_memory=True)
            return self._run_with_plan(
                plan,
                lambda: self._run_capability_text("ofdmsim", "selftest", {}),
            ), "ToolRegistry"

        if "emi" in lowered and "self" in lowered and "test" in lowered:
            plan = self._tool_plan(text, "ofdmsim_emi", "selftest", {}, include_memory=True)
            return self._run_with_plan(
                plan,
                lambda: self._run_capability_text("ofdmsim_emi", "selftest", {}),
            ), "ToolRegistry"

        if "emi" in lowered and "ofdmsim" in lowered and any(word in lowered for word in ["compare", "comparison"]):
            params = self._parse_emi_compare_request(lowered)
            plan = self._tool_plan(text, "ofdmsim_emi", "compare_emi", params, include_memory=True)
            return self._run_with_plan(
                plan,
                lambda: self._run_capability_text("ofdmsim_emi", "compare_emi", params),
            ), "ToolRegistry"

        if "cadconverter" in lowered and any(word in lowered for word in ["inspect", "files", "list"]):
            plan = self._tool_plan(text, "cadconverter", "inspect", {}, include_memory=True)
            return self._run_with_plan(
                plan,
                lambda: self._run_capability_text("cadconverter", "inspect", {}),
            ), "ToolRegistry"

        if "folderguardian" in lowered and any(word in lowered for word in ["inspect", "files", "list", "project"]):
            plan = self._tool_plan(text, "folderguardian", "inspect", {}, include_memory=True)
            return self._run_with_plan(
                plan,
                lambda: self._run_capability_text("folderguardian", "inspect", {}),
            ), "ToolRegistry"

        if "cadconverter" in lowered and "validate" in lowered and any(ext in lowered for ext in [".glb", ".gltf"]):
            params = self._parse_cadconverter_request(text)
            if not params.get("input_path"):
                return 'Please include the GLB path in quotes, for example: validate CADConverter GLB "python half.glb"', "ToolRegistry"
            plan = self._tool_plan(text, "cadconverter", "validate_glb", params, include_memory=True)
            return self._run_with_plan(
                plan,
                lambda: self._run_capability_text("cadconverter", "validate_glb", params),
            ), "ToolRegistry"

        if "cadconverter" in lowered and any(phrase in lowered for phrase in ["step to glb", "stp to glb", "convert step", "convert stp"]):
            params = self._parse_cadconverter_request(text)
            if not params.get("input_path"):
                return 'Please include the STEP/STP input path in quotes, for example: convert CADConverter STEP to GLB "python half.stp"', "ToolRegistry"
            plan = self._tool_plan(text, "cadconverter", "convert_step_to_glb", params, include_memory=True)
            return self._run_with_plan(
                plan,
                lambda: self._run_capability_text("cadconverter", "convert_step_to_glb", params),
            ), "ToolRegistry"

        if "cadconverter" in lowered and any(phrase in lowered for phrase in ["glb to step", "gltf to step", "convert glb", "convert gltf"]):
            params = self._parse_cadconverter_request(text)
            if not params.get("input_path"):
                return 'Please include the GLB/GLTF input path in quotes, for example: convert CADConverter GLB to STEP "python half.glb" mode reconstructed', "ToolRegistry"
            plan = self._tool_plan(text, "cadconverter", "convert_glb_to_step", params, include_memory=True)
            return self._run_with_plan(
                plan,
                lambda: self._run_capability_text("cadconverter", "convert_glb_to_step", params),
            ), "ToolRegistry"

        if "ofdmsim" in lowered and "confidence" in lowered and any(word in lowered for word in ["assess", "probability", "reliable"]):
            params = {"tool_name": "ofdmsim"}
            plan = self._tool_plan(text, "bayesian_engine", "assess_tool_confidence", params, include_memory=True)
            return self._run_with_plan(
                plan,
                lambda: self._run_capability_text("bayesian_engine", "assess_tool_confidence", params),
            ), "BayesianEngine"

        if "probability" in lowered and any(phrase in lowered for phrase in ["tool output", "output reliable", "output is reliable"]):
            params = {"tool_name": "ofdmsim"}
            plan = self._tool_plan(text, "bayesian_engine", "assess_tool_confidence", params, include_memory=True)
            return self._run_with_plan(
                plan,
                lambda: self._run_capability_text("bayesian_engine", "assess_tool_confidence", params),
            ), "BayesianEngine"

        if "ofdmsim" in lowered and any(word in lowered for word in ["run", "simulate", "simulation"]):
            params = self._parse_ofdm_request(lowered)
            if any(word in lowered for word in ["bayesian", "confidence", "probability", "evidence"]):
                params["update_bayesian"] = True
            plan = self._tool_plan(text, "ofdmsim", "run_simulation", params, include_memory=True)
            return self._run_with_plan(
                plan,
                lambda: self._run_capability_text("ofdmsim", "run_simulation", params),
            ), "ToolRegistry"

        return None, None

    def snapshot(self):
        snapshot = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **self.runtime_service.runtime_snapshot(),
        }
        if not snapshot["recent_execution_plans"]:
            snapshot["recent_execution_plans"] = [plan.to_dict() for plan in self.recent_plans[-8:]]
        return snapshot

    def _parse_ofdm_request(self, lowered_text):
        params = {}
        if any(phrase in lowered_text for phrase in ["rayleigh", "fading", "multipath"]):
            params["channel"] = "rayleigh"
        elif any(phrase in lowered_text for phrase in ["ideal", "no channel", "clean channel"]):
            params["channel"] = "ideal"
        elif any(phrase in lowered_text for phrase in ["awgn", "white gaussian", "white noise", "additive noise"]):
            params["channel"] = "awgn"

        compact_text = re.sub(r"[\s_-]+", "", lowered_text)
        if any(marker in compact_text for marker in ["64qam", "qam64"]):
            params["modulation"] = "64-qam"
        elif any(marker in compact_text for marker in ["16qam", "qam16"]):
            params["modulation"] = "16-qam"
        elif "qpsk" in compact_text or "quadrature phase" in lowered_text:
            params["modulation"] = "qpsk"

        if (
            re.search(r"\bzf\b", lowered_text)
            or "zero-forcing" in lowered_text
            or "zero forcing" in lowered_text
        ):
            params["equalizer"] = "zf"
        elif "mmse" in lowered_text or "minimum mean square" in lowered_text:
            params["equalizer"] = "mmse"

        snr = self._first_number(
            lowered_text,
            [
                rf"\bsnr\s*(?:=|is|at|of|to)?\s*({_NUMBER_PATTERN})",
                rf"\b(?:es/n0|eb/n0)\s*(?:=|is|at|of|to)?\s*({_NUMBER_PATTERN})",
                rf"\b({_NUMBER_PATTERN})\s*(?:db|decibels?)\s*(?:snr|es/n0|eb/n0)?\b",
            ],
        )
        if snr is not None:
            params["snr_db"] = snr

        frames = self._first_number(
            lowered_text,
            [
                rf"\bframes?\s*(?:=|is|at|of|to)?\s*(\d+)",
                r"\bfor\s+(\d+)\s+frames?\b",
                r"\b(\d+)\s+frames?\b",
            ],
            cast=int,
        )
        if frames is not None:
            params["frames"] = frames

        seed = self._first_number(
            lowered_text,
            [
                r"\b(?:random\s+)?seed\s*(?:=|is|at|of|to)?\s*(\d+)",
            ],
            cast=int,
        )
        if seed is not None:
            params["seed"] = seed

        threshold = self._first_number(
            lowered_text,
            [
                rf"\bber\s+threshold\s*(?:=|is|at|of|to)?\s*({_NUMBER_PATTERN})",
                rf"\bthreshold\s*(?:=|is|at|of|to)?\s*({_NUMBER_PATTERN})",
            ],
        )
        if threshold is not None:
            params["ber_threshold"] = threshold

        return params

    def _parse_emi_compare_request(self, lowered_text):
        params = self._parse_ofdm_request(lowered_text)

        probability = self._first_number(
            lowered_text,
            [
                rf"\b(?:emi|impulse|noise)?\s*(?:probability|prob|rate)\s*(?:=|is|at|of|to)?\s*({_NUMBER_PATTERN})",
                rf"\bp\s*=\s*({_NUMBER_PATTERN})",
            ],
        )
        if probability is not None:
            params["emi_probability"] = probability

        amplitude = self._first_number(
            lowered_text,
            [
                rf"\b(?:emi|impulse|noise)?\s*(?:amplitude|amp|rms)\s*(?:=|is|at|of|to)?\s*({_NUMBER_PATTERN})",
            ],
        )
        if amplitude is not None:
            params["emi_amplitude"] = amplitude

        width = self._first_number(
            lowered_text,
            [
                r"\b(?:emi|impulse|noise)?\s*width\s*(?:=|is|at|of|to)?\s*(\d+)",
                r"\b(\d+)\s+samples?\s+wide\b",
            ],
            cast=int,
        )
        if width is not None:
            params["emi_width"] = width

        clip = self._first_number(
            lowered_text,
            [
                rf"\b(?:clip|clipping)(?:\s+threshold)?\s*(?:=|is|at|of|to)?\s*({_NUMBER_PATTERN})",
                rf"\bclip\s+at\s*({_NUMBER_PATTERN})",
            ],
        )
        if clip is not None:
            params["clip_threshold"] = clip

        return params

    def plan_for(self, text):
        lowered = (text or "").lower()
        stripped = lowered.strip()
        if stripped in {"runtime snapshot", "runtime snapshot json", "runtime status json"}:
            return self._metadata_plan(text, "RuntimeOrchestrator", "Build runtime snapshot.")
        if any(phrase in lowered for phrase in ["capability schema", "schema for"]):
            return self._metadata_plan(text, "RuntimeServiceFacade", "Return one registered capability schema.")
        if stripped.startswith(("run capability ", "execute capability ")):
            request = self._parse_generic_capability_request(text)
            if request.get("error"):
                return None
            return self._tool_plan(text, request["tool_name"], request["capability_name"], request["params"], include_memory=True)
        if any(phrase in lowered for phrase in ["list tools", "available tools", "what tools", "capabilities"]):
            return self._metadata_plan(text, "ToolRegistry", "List registered tool capabilities.")
        if any(phrase in lowered for phrase in ["list models", "model health", "model status", "available models"]):
            return self._metadata_plan(text, "ModelRegistry", "List configured model routes.")
        if any(phrase in lowered for phrase in ["refresh project memory", "index project memory", "bootstrap project memory"]):
            return self._memory_plan(text, action="index")
        if "search memory" in lowered or "project memory" in lowered:
            return self._memory_plan(text)
        tool_route = self._route_tool_request(text)
        if tool_route and not tool_route.get("error"):
            return self._tool_plan(
                text,
                tool_route["tool_name"],
                tool_route["capability_name"],
                tool_route["params"],
                include_memory=tool_route.get("include_memory", True),
            )
        if "cadconverter" in lowered and any(word in lowered for word in ["inspect", "files", "list"]):
            return self._tool_plan(text, "cadconverter", "inspect", {}, include_memory=True)
        if "cadconverter" in lowered and "validate" in lowered:
            return self._tool_plan(text, "cadconverter", "validate_glb", self._parse_cadconverter_request(text), include_memory=True)
        if "cadconverter" in lowered and any(phrase in lowered for phrase in ["step to glb", "stp to glb", "convert step", "convert stp"]):
            return self._tool_plan(text, "cadconverter", "convert_step_to_glb", self._parse_cadconverter_request(text), include_memory=True)
        if "cadconverter" in lowered and any(phrase in lowered for phrase in ["glb to step", "gltf to step", "convert glb", "convert gltf"]):
            return self._tool_plan(text, "cadconverter", "convert_glb_to_step", self._parse_cadconverter_request(text), include_memory=True)
        if "folderguardian" in lowered and any(word in lowered for word in ["inspect", "files", "list", "project"]):
            return self._tool_plan(text, "folderguardian", "inspect", {}, include_memory=True)
        if "ofdmsim" in lowered and "self" in lowered and "test" in lowered:
            return self._tool_plan(text, "ofdmsim", "selftest", {}, include_memory=True)
        if "emi" in lowered and "ofdmsim" in lowered and any(word in lowered for word in ["compare", "comparison"]):
            return self._tool_plan(text, "ofdmsim_emi", "compare_emi", self._parse_emi_compare_request(lowered), include_memory=True)
        if "ofdmsim" in lowered and any(word in lowered for word in ["run", "simulate", "simulation"]):
            params = self._parse_ofdm_request(lowered)
            if any(word in lowered for word in ["bayesian", "confidence", "probability", "evidence"]):
                params["update_bayesian"] = True
            return self._tool_plan(text, "ofdmsim", "run_simulation", params, include_memory=True)
        if "ofdmsim" in lowered and "confidence" in lowered:
            return self._tool_plan(text, "bayesian_engine", "assess_tool_confidence", {"tool_name": "ofdmsim"}, include_memory=True)
        return None

    def _run_with_plan(self, plan, executor):
        plan.start()
        try:
            result = executor()
            plan.complete(summarize_result(result))
            self._record_plan(plan)
            return result
        except Exception as exc:
            plan.fail(exc)
            self._record_plan(plan)
            raise

    def _run_capability_text(self, tool_name, capability_name, params=None):
        result = self.runtime_service.run_capability(tool_name, capability_name, params or {})
        return result["output_text"]

    def _record_plan(self, plan):
        self.recent_plans.append(plan)
        self.recent_plans = self.recent_plans[-25:]
        if hasattr(self.memory, "record_execution_plan"):
            self.memory.record_execution_plan(plan)

    def _metadata_plan(self, intent, target, details):
        return make_plan(
            intent=intent,
            source="RuntimeOrchestrator",
            steps=[
                RuntimePlanStep("inspect_registry", target, details),
                RuntimePlanStep("format_response", "RuntimeOrchestrator", "Return registry information to the caller."),
            ],
        )

    def _memory_plan(self, intent, action="search"):
        if action == "index":
            steps = [
                RuntimePlanStep("list_projects", "ToolRegistry", "List registered projects and roots."),
                RuntimePlanStep("scan_documents", "MemoryService", "Find README, docs, and project manifest files."),
                RuntimePlanStep("index_documents", "MemoryService", "Store chunks and embeddings for retrieval."),
                RuntimePlanStep("format_response", "RuntimeOrchestrator", "Summarize indexed document counts."),
            ]
        else:
            steps = [
                RuntimePlanStep("embed_query", "MemoryService", "Embed or tokenize the user query for retrieval."),
                RuntimePlanStep("retrieve_context", "MemoryService", "Search project and tool-run memory chunks."),
                RuntimePlanStep("format_response", "RuntimeOrchestrator", "Summarize relevant memory hits."),
            ]
        return make_plan(
            intent=intent,
            source="MemoryService",
            steps=steps,
        )

    def _tool_plan(self, intent, tool_name, capability, params, include_memory=True):
        steps = [
            RuntimePlanStep("resolve_capability", "ToolRegistry", f"Select {tool_name}.{capability}."),
            RuntimePlanStep(
                "validate_params",
                "RuntimeServiceFacade",
                "Validate structured adapter parameters.",
                metadata={"params": params},
            ),
            RuntimePlanStep("execute_adapter", "RuntimeServiceFacade", f"Run {tool_name}.{capability} through the service facade."),
        ]
        if include_memory:
            steps.extend(
                [
                    RuntimePlanStep("write_artifact", "runtime_artifacts", "Store machine-readable run details when available."),
                    RuntimePlanStep("store_memory", "MemoryService", "Record the run and index output for retrieval."),
                ]
            )
        return make_plan(intent=intent, source="ToolRegistry", steps=steps)

    def _tool_health_text(self):
        health = self.tool_registry.health()
        lines = ["Runtime tool health:"]
        for item in health:
            state = "available" if item["available"] else "missing"
            lines.append(f"- {item['name']}: {state} at {item['root']}")
        return "\n".join(lines)

    def _capability_list_text(self):
        lines = ["Available engineering capabilities:"]
        current_tool = None
        for capability in self.runtime_service.list_capabilities():
            if capability["tool_name"] != current_tool:
                state = "available" if capability["available"] else "missing"
                lines.append(
                    f"- {capability['tool_name']} ({state}, {capability['runtime']}): "
                    f"{capability['root']}"
                )
                current_tool = capability["tool_name"]
            lines.append(f"  - {capability['capability_name']}: {capability['description']}")
        return "\n".join(lines)

    def _capability_schema_text(self, text):
        lowered = (text or "").lower()
        for capability in self.runtime_service.list_capabilities():
            full_name = f"{capability['tool_name']}.{capability['capability_name']}".lower()
            loose_name = f"{capability['tool_name']} {capability['capability_name']}".lower()
            if full_name in lowered or loose_name in lowered:
                schema = self.runtime_service.get_capability_schema(
                    capability["tool_name"],
                    capability["capability_name"],
                )
                return json.dumps(schema, indent=2)
        return None

    def _parse_generic_capability_request(self, text):
        match = re.match(
            r"\s*(?:run|execute)\s+capability\s+([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\s*(.*)\s*$",
            text or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return {"error": 'Use: run capability tool.capability {"param": "value"}'}

        params_text = (match.group(3) or "").strip()
        if not params_text:
            params = {}
        else:
            try:
                params = json.loads(params_text)
            except json.JSONDecodeError as exc:
                return {"error": f"Capability parameters must be a JSON object: {exc}"}
            if not isinstance(params, dict):
                return {"error": "Capability parameters must be a JSON object."}

        return {
            "tool_name": match.group(1),
            "capability_name": match.group(2),
            "params": params,
        }

    def _memory_search_text(self, text):
        hits = self.memory.search(text, limit=5)
        if not hits:
            return "I did not find relevant project memory yet. Indexing READMEs is active, and richer tool-result memory will build up as adapters run."
        lines = ["Relevant project memory:"]
        for hit in hits:
            lines.append(f"- {hit['project']}: {hit['snippet']}")
        return "\n".join(lines)

    def _refresh_project_memory_text(self):
        counts = self.bootstrap_project_memory()
        indexed_total = sum(counts.values())
        lines = [f"Project memory refresh complete. Indexed {indexed_total} documents:"]
        for name, count in counts.items():
            lines.append(f"- {name}: {count}")
        return "\n".join(lines)

    def _route_tool_request(self, text):
        lowered = (text or "").lower()
        if not lowered.strip():
            return None

        if self._wants_bayesian_health(lowered):
            return self._tool_route("bayesian_engine", "health", source="BayesianEngine", include_memory=False)

        if self._wants_bayesian_breakdown(lowered):
            return self._tool_route(
                "bayesian_engine",
                "tool_confidence_breakdown",
                {"tool_name": self._target_tool_from_text(lowered)},
                source="BayesianEngine",
            )

        if self._wants_bayesian_confidence(lowered) and not self._wants_tool_execution(lowered):
            return self._tool_route(
                "bayesian_engine",
                "assess_tool_confidence",
                {"tool_name": self._target_tool_from_text(lowered)},
                source="BayesianEngine",
            )

        security_route = self._route_security_suite_request(text)
        if security_route:
            return security_route

        if self._mentions_cadconverter(lowered):
            cad_route = self._route_cadconverter_request(text)
            if cad_route:
                return cad_route

        if self._mentions_folderguardian(lowered):
            if self._wants_folderguardian_review(lowered):
                return self._tool_route("folderguardian", "security_review")
            if self._wants_folderguardian_logs(lowered):
                params = self._parse_folderguardian_path_request(text)
                return self._tool_route(
                    "folderguardian",
                    "check_logs",
                    params,
                )
            if self._wants_folderguardian_watch_status(lowered):
                return self._tool_route("folderguardian", "watch_status")
            if self._wants_folderguardian_stop_watch(lowered):
                return self._tool_route(
                    "folderguardian",
                    "stop_watch",
                    self._parse_folderguardian_path_request(text),
                )
            if self._wants_folderguardian_start_watch(lowered):
                return self._tool_route(
                    "folderguardian",
                    "start_watch",
                    self._parse_folderguardian_path_request(text),
                )
            if self._wants_folderguardian_dry_run(lowered):
                return self._tool_route(
                    "folderguardian",
                    "dry_run_protection_plan",
                    self._parse_folderguardian_plan_request(text),
                )
            crypto_action = self._folderguardian_crypto_action(lowered)
            if crypto_action:
                params = self._parse_folderguardian_path_request(text)
                self._add_bayesian_update_if_requested(lowered, params)
                return self._tool_route(
                    "folderguardian",
                    crypto_action,
                    params,
                )
            if self._mentions_folderguardian_blocked_action(lowered):
                return self._tool_route(
                    "folderguardian",
                    "dry_run_protection_plan",
                    self._parse_folderguardian_plan_request(text),
                )
            if self._wants_inspection(lowered) or "folder guardian" in lowered or "folderguardian" in lowered:
                return self._tool_route("folderguardian", "inspect")

        workspace_route = self._route_workspace_request(text)
        if workspace_route:
            return workspace_route

        if self._mentions_ofdm(lowered):
            if self._wants_selftest(lowered):
                tool_name = "ofdmsim_emi" if self._mentions_emi(lowered) else "ofdmsim"
                return self._tool_route(tool_name, "selftest")
            if self._wants_emi_compare(lowered):
                return self._tool_route("ofdmsim_emi", "compare_emi", self._parse_emi_compare_request(lowered))
            if self._wants_ofdm_run(lowered):
                params = self._parse_ofdm_request(lowered)
                if self._wants_bayesian_update(lowered):
                    params["update_bayesian"] = True
                return self._tool_route("ofdmsim", "run_simulation", params)

        generic_route = self._route_registered_tool_request(text)
        if generic_route:
            return generic_route

        return None

    def _route_cadconverter_request(self, text):
        lowered = (text or "").lower()
        if self._wants_cad_recommendation(lowered):
            return self._tool_route("cadconverter", "recommend_conversion", self._parse_cadconverter_request(text))

        if self._wants_cad_batch_validate(lowered):
            params = {}
            max_files = self._first_number(
                lowered,
                [r"\b(?:first|up to|max(?:imum)?)\s+(\d+)\b", r"\b(\d+)\s+(?:glb|gltf|files)\b"],
                int,
            )
            if max_files is not None:
                params["max_files"] = max_files
            self._add_bayesian_update_if_requested(lowered, params)
            return self._tool_route("cadconverter", "batch_validate_glb", params)

        if self._wants_cad_validate(lowered):
            params = self._parse_cadconverter_request(text, input_suffixes={".glb", ".gltf"})
            if not params.get("input_path"):
                return {
                    "error": "I can validate a GLB/GLTF file, but I need the file name or path from the CADConverter workspace.",
                    "source": "ToolRegistry",
                }
            self._add_bayesian_update_if_requested(lowered, params)
            return self._tool_route("cadconverter", "validate_glb", params)

        conversion = self._cad_conversion_kind(text)
        if conversion == "step_to_glb":
            params = self._parse_cadconverter_request(text, input_suffixes={".step", ".stp"})
            if not params.get("input_path"):
                return {
                    "error": "I can convert STEP/STP to GLB, but I need the input file name or path from the CADConverter workspace.",
                    "source": "ToolRegistry",
                }
            self._add_bayesian_update_if_requested(lowered, params)
            return self._tool_route("cadconverter", "convert_step_to_glb", params)

        if conversion == "glb_to_step":
            params = self._parse_cadconverter_request(text, input_suffixes={".glb", ".gltf"})
            if not params.get("input_path"):
                return {
                    "error": "I can convert GLB/GLTF to STEP, but I need the input file name or path from the CADConverter workspace.",
                    "source": "ToolRegistry",
                }
            self._add_bayesian_update_if_requested(lowered, params)
            return self._tool_route("cadconverter", "convert_glb_to_step", params)

        if self._wants_inspection(lowered):
            return self._tool_route("cadconverter", "inspect")
        return None

    def _route_security_suite_request(self, text):
        lowered = (text or "").lower()
        if self._wants_defender_history(lowered):
            return self._tool_route("security_suite", "defender_history", self._parse_security_history_request(lowered))
        if self._wants_security_history(lowered):
            return self._tool_route("security_suite", "security_history", self._parse_security_history_request(lowered))
        if self._wants_security_explanation(lowered):
            return self._tool_route("security_suite", "explain_assessment", self._parse_security_artifact_request(text))
        if self._wants_security_restore(lowered):
            params = self._parse_security_restore_request(text)
            if any(phrase in lowered for phrase in ["confirm restore", "confirm_restore=true", "restore now"]):
                params["confirm_restore"] = True
            return self._tool_route("security_suite", "restore_quarantine", params)
        if self._wants_security_quarantine_live(lowered):
            params = self._parse_security_path_request(text)
            if any(phrase in lowered for phrase in ["confirm quarantine", "confirm_quarantine=true", "quarantine now"]):
                params["confirm_quarantine"] = True
            if not params.get("target_path"):
                return {
                    "error": "I can quarantine a file or folder, but I need a target path.",
                    "source": "ToolRegistry",
                }
            return self._tool_route("security_suite", "quarantine_path", params)
        if self._wants_security_quarantine_plan(lowered):
            params = self._parse_security_path_request(text)
            if not params.get("target_path"):
                return {
                    "error": "I can prepare a quarantine plan, but I need a file or folder path.",
                    "source": "ToolRegistry",
                }
            return self._tool_route("security_suite", "quarantine_plan", params)
        if self._wants_defender_scan(lowered):
            params = self._parse_security_path_request(text)
            if "confirm defender scan" in lowered or "confirm_scan=true" in lowered or "confirm scan" in lowered:
                params["confirm_scan"] = True
            timeout_seconds = self._first_number(lowered, [r"\btimeout\s+(\d+)\s*(?:seconds?|s)\b"], int)
            if timeout_seconds is not None:
                params["timeout_seconds"] = timeout_seconds
            if not params.get("target_path"):
                return {
                    "error": "I can prepare a Defender scan, but I need a file or folder path.",
                    "source": "ToolRegistry",
                }
            return self._tool_route("security_suite", "defender_scan", params)
        if self._wants_project_security_audit(lowered):
            params = self._parse_security_path_request(text)
            if not params.get("target_path"):
                return {
                    "error": "I can audit a project, but I need the project folder path.",
                    "source": "ToolRegistry",
                }
            return self._tool_route("security_suite", "audit_project", params)
        if self._wants_rule_scan(lowered):
            params = self._parse_security_path_request(text)
            rule_path = self._extract_rule_path(text)
            if rule_path:
                params["rules_path"] = rule_path
            if not params.get("target_path"):
                return {
                    "error": "I can run local security rules, but I need a file or folder path.",
                    "source": "ToolRegistry",
                }
            return self._tool_route("security_suite", "rule_scan", params)
        if not self._wants_security_assessment(lowered):
            return None
        if self._mentions_folderguardian(lowered) and not self._explicit_malicious_scan(lowered):
            return None
        params = self._parse_security_path_request(text)
        if not params.get("target_path"):
            return {
                "error": "I can run a local security assessment, but I need a file or folder path.",
                "source": "ToolRegistry",
            }
        self._add_bayesian_update_if_requested(lowered, params)
        return self._tool_route("security_suite", "assess_path", params)

    def _tool_route(self, tool_name, capability_name, params=None, source="ToolRegistry", include_memory=True):
        return {
            "tool_name": tool_name,
            "capability_name": capability_name,
            "params": params or {},
            "source": source,
            "include_memory": include_memory,
        }

    def _add_bayesian_update_if_requested(self, lowered_text, params):
        if self._wants_bayesian_update(lowered_text):
            params["update_bayesian"] = True
        return params

    def _mentions_ofdm(self, lowered_text):
        return (
            "ofdmsim" in lowered_text
            or re.search(r"\bofdm\b", lowered_text) is not None
            or "ber" in lowered_text and any(marker in lowered_text for marker in ["snr", "db", "channel", "modulation"])
        )

    def _mentions_emi(self, lowered_text):
        return any(
            marker in lowered_text
            for marker in ["emi", "emc", "impulse noise", "impulse", "transient", "clipping", "clip threshold"]
        )

    def _mentions_cadconverter(self, lowered_text):
        return (
            "cadconverter" in lowered_text
            or "cad converter" in lowered_text
            or re.search(r"\bcad\b", lowered_text) is not None
            or any(ext in lowered_text for ext in [".step", ".stp", ".glb", ".gltf"])
            or any(word in lowered_text for word in ["step file", "stp file", "glb file", "gltf file"])
        )

    def _mentions_folderguardian(self, lowered_text):
        return (
            "folderguardian" in lowered_text
            or "folder guardian" in lowered_text
            or "folder encryption" in lowered_text
            or "watchtower" in lowered_text
            or "security log" in lowered_text
            or (
                re.search(r"\b(encrypt|decrypt|lock|unlock|restore|protect|start watch|stop watch|watch|monitor|check logs|show logs|read logs)\b", lowered_text) is not None
                and any(marker in lowered_text for marker in ["folder", "drive", ":\\", " in "])
            )
        )

    def _mentions_folderguardian_blocked_action(self, lowered_text):
        return re.search(
            r"\b(encrypt|decrypt|lock|unlock|protect)\b|\bstart\s+(watch|monitor)\b|\bwatch\s+this\s+folder\b",
            lowered_text,
        ) is not None

    def _folderguardian_crypto_action(self, lowered_text):
        if re.search(r"\b(decrypt|unlock|restore)\b", lowered_text):
            return "decrypt_folder"
        if re.search(r"\b(encrypt|lock|protect)\b", lowered_text):
            return "encrypt_folder"
        return None

    def _wants_folderguardian_review(self, lowered_text):
        return any(word in lowered_text for word in ["review", "security", "risk", "audit", "architecture"])

    def _wants_folderguardian_dry_run(self, lowered_text):
        return any(phrase in lowered_text for phrase in ["dry run", "dry-run", "plan"]) and any(
            word in lowered_text for word in ["protect", "encrypt", "decrypt", "lock", "unlock", "watch", "monitor", "folderguardian", "folder guardian"]
        )

    def _wants_folderguardian_logs(self, lowered_text):
        return any(phrase in lowered_text for phrase in ["check logs", "show logs", "read logs", "security log", "securitylog"])

    def _wants_folderguardian_start_watch(self, lowered_text):
        return (
            any(phrase in lowered_text for phrase in ["start watch", "start watchtower", "watch this folder", "monitor this folder"])
            or re.search(r"\bwatch\s+(?:folder\s+)?", lowered_text) is not None
        ) and not self._wants_folderguardian_stop_watch(lowered_text)

    def _wants_folderguardian_stop_watch(self, lowered_text):
        return any(phrase in lowered_text for phrase in ["stop watch", "stop watchtower", "pause watch", "pause monitoring", "stop monitoring"])

    def _wants_folderguardian_watch_status(self, lowered_text):
        return any(phrase in lowered_text for phrase in ["watch status", "watchtower status", "monitor status", "active watches"])

    def _parse_folderguardian_plan_request(self, text):
        params = self._parse_folderguardian_path_request(text)
        lowered = (text or "").lower()
        for action in ("encrypt", "decrypt", "lock", "unlock", "watch", "monitor", "inspect", "protect"):
            if action in lowered:
                params["action"] = {"lock": "encrypt", "unlock": "decrypt"}.get(action, action)
                break
        return params

    def _parse_folderguardian_path_request(self, text):
        params = {}
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text or "")
        paths = [first or second for first, second in quoted]
        if paths:
            params["folder_path"] = paths[0]
        else:
            match = re.search(r"([A-Za-z]:\\[^\r\n\"']+)", text or "")
            if match:
                params["folder_path"] = match.group(1).strip()
            else:
                natural_path = self._extract_natural_windows_folder_path(text)
                if natural_path:
                    params["folder_path"] = natural_path
        max_lines = self._first_number((text or "").lower(), [r"\blast\s+(\d+)\s+lines?\b", r"\b(\d+)\s+log\s+lines?\b"], int)
        if max_lines is not None:
            params["max_lines"] = max_lines
        return params

    def _extract_natural_windows_folder_path(self, text):
        raw = text or ""
        patterns = [
            r"(?P<leaf>[A-Za-z0-9_. -]+?)\s+in\s+(?P<parent>[A-Za-z0-9_. -]+?)\s+in\s+(?P<drive>[A-Za-z])\s+(?:drive|dr)\b",
            r"(?P<drive>[A-Za-z])\s+(?:drive|dr)\s+(?:in\s+)?(?P<parent>[A-Za-z0-9_. -]+?)\s+(?:folder\s+)?(?:named|called)?\s*(?P<leaf>[A-Za-z0-9_. -]+)\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, raw, flags=re.IGNORECASE):
                drive = match.group("drive").upper()
                parent = self._clean_folder_phrase(match.group("parent"))
                leaf = self._clean_folder_phrase(match.group("leaf"))
                if drive and parent and leaf:
                    return f"{drive}:\\{parent}\\{leaf}"
        return None

    def _clean_folder_phrase(self, value):
        text = re.sub(
            r"\b(folderguardian|folder guardian|encrypt|decrypt|lock|unlock|protect|watch|monitor|check|logs?|security|safe|malicious|suspicious|infected|scan|review|is|for|the|a|an|particular|folder|named|called)\b",
            " ",
            value or "",
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s+", " ", text).strip(" .,:;")
        return text

    def _wants_inspection(self, lowered_text):
        return any(
            phrase in lowered_text
            for phrase in [
                "inspect",
                "check",
                "look at",
                "scan",
                "summarize",
                "what files",
                "which files",
                "list files",
                "show files",
                "inventory",
                "what can",
                "capabilities",
            ]
        )

    def _wants_selftest(self, lowered_text):
        if any(phrase in lowered_text for phrase in ["simulation test", "sim test", "test simulation"]):
            return False
        return (
            "selftest" in lowered_text
            or "self-test" in lowered_text
            or "self test" in lowered_text
            or re.search(r"\btest\s+(?:the\s+)?(?:adapter|binary|executable|build|self\s*test)\b", lowered_text) is not None
            or re.search(r"\b(?:adapter|binary|executable|build)\s+test\b", lowered_text) is not None
            or "health check" in lowered_text
        )

    def _wants_ofdm_run(self, lowered_text):
        return any(
            phrase in lowered_text
            for phrase in [
                "run",
                "simulate",
                "simulation",
                "try",
                "sweep",
                "what ber",
                "ber do we get",
                "ber for",
                "at snr",
                "values",
                "general conditions",
                "default conditions",
                "normal conditions",
                "basic ofdm",
            ]
        ) or re.search(r"\b\d+(?:\.\d+)?\s*(?:db|decibels?)\b", lowered_text) is not None

    def _wants_emi_compare(self, lowered_text):
        if not self._mentions_emi(lowered_text):
            return False
        return any(
            phrase in lowered_text
            for phrase in [
                "compare",
                "comparison",
                "baseline",
                "clipping",
                "clip",
                "with emi",
                "emi probability",
                "impulse probability",
                "impulse noise",
                "transient",
            ]
        )

    def _wants_bayesian_update(self, lowered_text):
        return any(word in lowered_text for word in ["bayesian", "confidence", "probability", "evidence", "reliable"])

    def _wants_tool_execution(self, lowered_text):
        return any(
            word in lowered_text
            for word in [
                "run",
                "simulate",
                "simulation",
                "compare",
                "validate",
                "convert",
                "export",
                "encrypt",
                "decrypt",
                "lock",
                "unlock",
                "start",
                "stop",
                "watch",
                "monitor",
                "scan",
            ]
        )

    def _wants_bayesian_health(self, lowered_text):
        return ("bayesian" in lowered_text or "evidence broker" in lowered_text) and "health" in lowered_text

    def _wants_bayesian_confidence(self, lowered_text):
        if "bayesian" in lowered_text and any(word in lowered_text for word in ["assess", "confidence", "probability", "reliable"]):
            return True
        if any(word in lowered_text for word in ["confidence", "reliable", "trustworthy", "quality"]) and any(
            marker in lowered_text for marker in ["tool", "output", "result", "ofdm", "cad", "folderguardian", "folder guardian", "security suite", "security scan"]
        ):
            return True
        return False

    def _wants_cad_validate(self, lowered_text):
        return any(word in lowered_text for word in ["validate", "verify", "check"]) and any(
            ext in lowered_text for ext in [".glb", ".gltf", "glb file", "gltf file"]
        )

    def _wants_cad_recommendation(self, lowered_text):
        return any(word in lowered_text for word in ["recommend", "suggest", "advise", "workflow", "settings"]) and any(
            marker in lowered_text for marker in ["cad", "step", "stp", "glb", "gltf", "conversion"]
        )

    def _wants_cad_batch_validate(self, lowered_text):
        return any(phrase in lowered_text for phrase in ["validate all", "batch validate", "check all"]) and any(
            marker in lowered_text for marker in ["glb", "gltf", "cadconverter", "cad converter"]
        )

    def _route_workspace_request(self, text):
        lowered = (text or "").lower()
        if not any(word in lowered for word in ["workspace", "projects", "programs", "repos", "repositories"]):
            return None
        max_projects = self._first_number(lowered, [r"\b(?:first|up to|max(?:imum)?)\s+(\d+)\b"], int)
        params = {"max_projects": max_projects} if max_projects is not None else {}
        if "c++" in lowered or "cpp" in lowered:
            return self._tool_route("cpp_workspace", "inspect", params)
        if "c#" in lowered or "csharp" in lowered or "c sharp" in lowered:
            return self._tool_route("csharp_workspace", "inspect", params)
        if "python" in lowered:
            return self._tool_route("python_workspace", "inspect", params)
        if "coding" in lowered or "external" in lowered:
            return self._tool_route("python_workspace", "inspect", params)
        return None

    def _route_registered_tool_request(self, text):
        lowered = (text or "").lower()
        for tool in self.tool_registry.list_tools():
            if not self._mentions_registered_tool(tool, lowered):
                continue
            capability = self._select_registered_capability(tool, lowered)
            if not capability:
                continue
            if tool.name == "cadconverter":
                params = self._parse_cadconverter_request(text)
            elif tool.name == "folderguardian":
                params = (
                    self._parse_folderguardian_plan_request(text)
                    if capability.name == "dry_run_protection_plan"
                    else self._parse_folderguardian_path_request(text)
                )
            elif tool.name in {"cpp_workspace", "python_workspace", "csharp_workspace"}:
                params = {}
                max_projects = self._first_number(lowered, [r"\b(?:first|up to|max(?:imum)?)\s+(\d+)\b"], int)
                if max_projects is not None:
                    params["max_projects"] = max_projects
            elif tool.name == "security_suite":
                if capability.name == "security_history":
                    params = self._parse_security_history_request(lowered)
                elif capability.name == "explain_assessment":
                    params = self._parse_security_artifact_request(text)
                else:
                    params = self._parse_security_path_request(text)
            else:
                params = {}
            return self._tool_route(tool.name, capability.name, params)
        return None

    def _mentions_registered_tool(self, tool, lowered_text):
        aliases = {
            "cadconverter": ["cadconverter", "cad converter"],
            "folderguardian": ["folderguardian", "folder guardian", "watchtower"],
            "bayesian_engine": ["bayesian engine", "evidence broker"],
            "ofdmsim": ["ofdmsim", "ofdm"],
            "ofdmsim_emi": ["ofdmsim emi", "ofdm emi", "emi simulator"],
            "security_suite": ["security suite", "security scanner", "malware scanner", "local security scan"],
            "cpp_workspace": ["cpp workspace", "c++ workspace", "c++ projects"],
            "python_workspace": ["python workspace", "python projects"],
            "csharp_workspace": ["csharp workspace", "c# workspace", "c# projects", "c sharp projects"],
        }.get(tool.name, [])
        normalized_name = tool.name.replace("_", " ")
        return tool.name in lowered_text or normalized_name in lowered_text or any(alias in lowered_text for alias in aliases)

    def _select_registered_capability(self, tool, lowered_text):
        scored = []
        for capability in tool.capabilities:
            score = self._capability_text_score(capability, lowered_text)
            if score:
                scored.append((score, capability))
        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            return scored[0][1]
        for preferred in ("inspect", "health", "selftest"):
            for capability in tool.capabilities:
                if capability.name == preferred and self._wants_inspection(lowered_text):
                    return capability
        return None

    def _capability_text_score(self, capability, lowered_text):
        score = 0
        name_words = capability.name.replace("_", " ").split()
        score += sum(3 for word in name_words if word in lowered_text)
        description_words = set(re.findall(r"[a-z0-9]+", capability.description.lower()))
        request_words = set(re.findall(r"[a-z0-9]+", lowered_text))
        score += min(4, len(description_words & request_words))
        for example in capability.examples or []:
            example_words = set(re.findall(r"[a-z0-9]+", example.lower()))
            score += min(3, len(example_words & request_words))
        return score

    def _cad_conversion_kind(self, text):
        lowered = (text or "").lower()
        if not any(word in lowered for word in ["convert", "export", "make", "turn", "reconstruct"]):
            return None
        step_markers = [".step", ".stp", " step ", " stp ", "step file", "stp file"]
        glb_markers = [".glb", ".gltf", " glb ", " gltf ", "glb file", "gltf file"]

        if any(phrase in lowered for phrase in ["step to glb", "stp to glb", "step into glb", "stp into glb"]):
            return "step_to_glb"
        if any(phrase in lowered for phrase in ["glb to step", "gltf to step", "glb into step", "gltf into step"]):
            return "glb_to_step"
        if "to glb" in lowered or "as glb" in lowered or "export" in lowered:
            if any(marker in lowered for marker in step_markers):
                return "step_to_glb"
            if self._mentions_cadconverter(lowered):
                return "step_to_glb"
        if "to step" in lowered or "as step" in lowered or "reconstruct" in lowered:
            if any(marker in lowered for marker in glb_markers):
                return "glb_to_step"
            if self._mentions_cadconverter(lowered):
                return "glb_to_step"

        step_index = self._first_marker_index(lowered, step_markers)
        glb_index = self._first_marker_index(lowered, glb_markers)
        if step_index is not None and glb_index is not None:
            return "step_to_glb" if step_index < glb_index else "glb_to_step"
        if step_index is not None:
            return "step_to_glb"
        if glb_index is not None and ("step" in lowered or "reconstruct" in lowered):
            return "glb_to_step"
        return None

    def _first_marker_index(self, text, markers):
        found = [text.find(marker) for marker in markers if marker in text]
        return min(found) if found else None

    def _target_tool_from_text(self, lowered_text):
        if self._mentions_cadconverter(lowered_text):
            return "cadconverter"
        if self._mentions_folderguardian(lowered_text):
            return "folderguardian"
        if "security" in lowered_text or "malware" in lowered_text or "scan" in lowered_text:
            return "security_suite"
        if self._mentions_emi(lowered_text):
            return "ofdmsim_emi"
        if self._mentions_ofdm(lowered_text):
            return "ofdmsim"
        if hasattr(self.memory, "recent_tool_runs"):
            recent = self.memory.recent_tool_runs(limit=1)
            if recent:
                return recent[0]["tool_name"]
        return "ofdmsim"

    def _first_number(self, text, patterns, cast=float):
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            try:
                return cast(match.group(1))
            except (TypeError, ValueError):
                continue
        return None

    def _parse_cadconverter_request(self, text, input_suffixes=None):
        params = {}
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text or "")
        paths = [first or second for first, second in quoted]
        if paths:
            params["input_path"] = paths[0]
        if len(paths) > 1:
            params["output_path"] = paths[1]

        lowered = (text or "").lower()
        if not params.get("input_path"):
            simple_path = self._extract_unquoted_cad_path(text, input_suffixes=input_suffixes)
            if simple_path:
                params["input_path"] = simple_path
        if not params.get("input_path") and input_suffixes:
            inferred = self._infer_cadconverter_input_path(text, input_suffixes)
            if inferred:
                params["input_path"] = inferred

        for mode in ("advanced", "reconstructed", "faceted"):
            if mode in lowered:
                params["mode"] = mode
                break
        if "no validate" in lowered or "without validation" in lowered:
            params["validate"] = False
        elif "validate" in lowered:
            params["validate"] = True
        if "single sided" in lowered or "single-sided" in lowered:
            params["single_sided"] = True

        linear = re.search(r"linear(?:\s+deflection)?\s+([0-9.]+)", lowered)
        if linear:
            params["linear_deflection"] = float(linear.group(1))
        angular = re.search(r"angular(?:\s+deflection)?\s+([0-9.]+)", lowered)
        if angular:
            params["angular_deflection"] = float(angular.group(1))
        unit_scale = re.search(r"unit(?:\s+scale)?\s+([0-9.]+)", lowered)
        if unit_scale:
            params["unit_scale"] = float(unit_scale.group(1))

        return params

    def _extract_unquoted_cad_path(self, text, input_suffixes=None):
        suffixes = input_suffixes or {".step", ".stp", ".glb", ".gltf"}
        suffix_pattern = "|".join(re.escape(suffix.lstrip(".")) for suffix in sorted(suffixes))
        match = re.search(
            rf"([A-Za-z]:\\[^\s\"']+?\.(?:{suffix_pattern})|[\w./\\-]+?\.(?:{suffix_pattern}))",
            text or "",
            flags=re.IGNORECASE,
        )
        return match.group(1) if match else None

    def _infer_cadconverter_input_path(self, text, suffixes):
        root = self._tool_root("cadconverter")
        if not root or not root.exists():
            return None
        ignored = {".git", ".venv", "venv", "__pycache__", "node_modules"}
        candidates = []
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if set(path.relative_to(root).parts[:-1]) & ignored:
                continue
            candidates.append(path)
        if not candidates:
            return None

        lowered = (text or "").lower()
        for path in candidates:
            if path.name.lower() in lowered or str(path.relative_to(root)).lower() in lowered:
                return str(path.relative_to(root))

        text_tokens = set(re.findall(r"[a-z0-9]+", lowered))
        scored = []
        for path in candidates:
            stem_tokens = set(re.findall(r"[a-z0-9]+", path.stem.lower()))
            score = len(stem_tokens & text_tokens)
            if score:
                scored.append((score, path))
        if scored:
            scored.sort(key=lambda item: (-item[0], len(str(item[1]))))
            if len(scored) == 1 or scored[0][0] > scored[1][0]:
                return str(scored[0][1].relative_to(root))

        generic_file_request = any(
            phrase in lowered
            for phrase in ["the step file", "the stp file", "the glb file", "the gltf file", "the cad file"]
        )
        if generic_file_request and len(candidates) == 1:
            return str(candidates[0].relative_to(root))
        return None

    def _tool_root(self, tool_name):
        for tool in self.tool_registry.list_tools():
            if tool.name == tool_name:
                return tool.root
        return None

    def _wants_security_history(self, lowered_text):
        return any(
            phrase in lowered_text
            for phrase in [
                "security history",
                "security scan history",
                "recent security scans",
                "recent security assessments",
                "malware scan history",
            ]
        )

    def _wants_security_explanation(self, lowered_text):
        return any(
            phrase in lowered_text
            for phrase in [
                "explain security assessment",
                "explain the security assessment",
                "explain last security",
                "explain that security scan",
                "explain malware scan",
            ]
        )

    def _wants_security_assessment(self, lowered_text):
        if (
            self._wants_security_history(lowered_text)
            or self._wants_security_explanation(lowered_text)
            or self._wants_project_security_audit(lowered_text)
            or self._wants_defender_scan(lowered_text)
            or self._wants_security_quarantine_plan(lowered_text)
        ):
            return False
        if not self._has_security_target_hint(lowered_text):
            return False
        strong_markers = [
            "malicious",
            "malware",
            "virus",
            "trojan",
            "infected",
            "suspicious",
            "threat",
            "security scan",
            "scan for threats",
            "local risk",
            "is this file safe",
            "is this folder safe",
            "is it safe",
            "check if",
            "check this file",
            "check this folder",
            "review this folder's security",
            "review this folders security",
            "review folder security",
            "scan downloads",
            "scan download",
        ]
        if any(marker in lowered_text for marker in strong_markers):
            return True
        return "safe" in lowered_text and any(word in lowered_text for word in ["file", "folder", "download", "drive", ":\\"])

    def _wants_project_security_audit(self, lowered_text):
        return any(
            phrase in lowered_text
            for phrase in [
                "project security audit",
                "audit project security",
                "security audit for project",
                "check project for secrets",
                "scan project for secrets",
                "dependency audit",
                "audit dependencies",
                "repo security audit",
                "repository security audit",
            ]
        )

    def _wants_defender_scan(self, lowered_text):
        if any(phrase in lowered_text for phrase in ["without defender", "no defender", "skip defender"]):
            return False
        return "defender" in lowered_text and any(word in lowered_text for word in ["scan", "check", "run", "prepare"])

    def _wants_defender_history(self, lowered_text):
        return "defender" in lowered_text and any(word in lowered_text for word in ["history", "detections", "threats", "alerts"])

    def _wants_rule_scan(self, lowered_text):
        return any(phrase in lowered_text for phrase in ["rule scan", "security rules", "local rules", "rule pack", "yara-style"])

    def _wants_security_quarantine_plan(self, lowered_text):
        return "quarantine" in lowered_text and any(word in lowered_text for word in ["plan", "prepare", "how", "review"])

    def _wants_security_quarantine_live(self, lowered_text):
        return "quarantine" in lowered_text and not self._wants_security_quarantine_plan(lowered_text) and "restore" not in lowered_text

    def _wants_security_restore(self, lowered_text):
        return "quarantine" in lowered_text and any(word in lowered_text for word in ["restore", "recover", "put back"])

    def _wants_bayesian_breakdown(self, lowered_text):
        return "bayesian" in lowered_text and any(word in lowered_text for word in ["breakdown", "factors", "evidence"])

    def _explicit_malicious_scan(self, lowered_text):
        return any(word in lowered_text for word in ["malicious", "malware", "virus", "trojan", "infected", "suspicious", "threat"])

    def _has_security_target_hint(self, lowered_text):
        return (
            ":\\"
            in lowered_text
            or bool(re.search(r'"[^"]+"|\'[^\']+\'', lowered_text))
            or any(word in lowered_text for word in ["downloads", "download folder", "desktop", "documents", "folder", "file", "drive"])
            or self._extract_natural_windows_folder_path(lowered_text) is not None
        )

    def _parse_security_path_request(self, text):
        params = {}
        target = self._extract_quoted_path(text) or self._extract_absolute_windows_path(text)
        if not target:
            natural_path = self._extract_natural_windows_folder_path(text)
            if natural_path:
                target = natural_path
        if not target:
            target = self._known_user_folder_from_text(text)
        if target:
            params["target_path"] = target
        max_files = self._first_number((text or "").lower(), [r"\b(?:first|up to|max(?:imum)?)\s+(\d+)\s+files?\b"], int)
        if max_files is not None:
            params["max_files"] = max_files
        lowered = (text or "").lower()
        if any(phrase in lowered for phrase in ["without defender", "no defender", "skip defender"]):
            params["include_defender_status"] = False
        return params

    def _parse_security_history_request(self, lowered_text):
        params = {}
        limit = self._first_number(lowered_text, [r"\blast\s+(\d+)\b", r"\b(\d+)\s+(?:scans?|assessments?)\b"], int)
        if limit is not None:
            params["limit"] = limit
        return params

    def _parse_security_artifact_request(self, text):
        params = {}
        path = self._extract_quoted_path(text) or self._extract_absolute_windows_path(text)
        if path:
            params["artifact_path"] = path
        return params

    def _parse_security_restore_request(self, text):
        params = self._parse_security_artifact_request(text)
        lowered = (text or "").lower()
        id_match = re.search(r"\b(?:quarantine\s+id|id)\s*[:=]?\s*([A-Za-z0-9_-]{8,})", lowered)
        if id_match:
            params["quarantine_id"] = id_match.group(1)
        restore_path = self._extract_restore_path(text)
        if restore_path:
            params["restore_path"] = restore_path
        if "overwrite" in lowered:
            params["allow_overwrite"] = True
        return params

    def _extract_rule_path(self, text):
        lowered = (text or "").lower()
        if "rules" not in lowered and "rule pack" not in lowered:
            return None
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text or "")
        paths = [first or second for first, second in quoted]
        if len(paths) >= 2:
            return paths[-1]
        return None

    def _extract_restore_path(self, text):
        match = re.search(r"\brestore(?:\s+to)?\s+(?:path\s+)?([A-Za-z]:\\[^\r\n\"']+)", text or "", flags=re.IGNORECASE)
        if match:
            return self._clean_unquoted_path_tail(match.group(1))
        return None

    def _extract_quoted_path(self, text):
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text or "")
        paths = [first or second for first, second in quoted]
        return paths[0] if paths else None

    def _extract_absolute_windows_path(self, text):
        match = re.search(r"([A-Za-z]:\\[^\r\n\"']+)", text or "")
        if not match:
            return None
        return self._clean_unquoted_path_tail(match.group(1))

    def _clean_unquoted_path_tail(self, value):
        text = (value or "").strip().strip(" .,:;")
        text = re.sub(r"\s+(?:and|for|to)\s+.*$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(
            r"\s+\b(?:safe|malicious|suspicious|infected|security|risk|scan|check|review|please)\b.*$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        return text.strip(" .,:;")

    def _known_user_folder_from_text(self, text):
        lowered = (text or "").lower()
        home = Path(os.path.expanduser("~"))
        if "downloads" in lowered or "download folder" in lowered:
            return str(home / "Downloads")
        if "desktop" in lowered:
            return str(home / "Desktop")
        if "documents" in lowered or "document folder" in lowered:
            return str(home / "Documents")
        return None


_NUMBER_PATTERN = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
