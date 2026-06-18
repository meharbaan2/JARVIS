const { app, BrowserWindow, ipcMain } = require('electron');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');
const { readRuntimeArtifact } = require('./runtimeArtifacts');

const repoRoot = path.resolve(__dirname, '..', '..');

let pyProc = null;
let grpcClient = null;
let runtimeClient = null;
let runtimeMemoryClient = null;
let runtimeModelClient = null;
let win = null;
let voiceProc = null;

function trimShellQuotes(value) {
    return (value || '').trim().replace(/^["']|["']$/g, '');
}

function splitOptionalArgs(value) {
    const trimmed = (value || '').trim();
    return trimmed ? trimmed.split(/\s+/).map(trimShellQuotes).filter(Boolean) : [];
}

function resolvePythonLaunch() {
    const command = trimShellQuotes(process.env.HIVEMIND_PYTHON) || 'python';
    const args = splitOptionalArgs(process.env.HIVEMIND_PYTHON_ARGS);
    return { command, args };
}

function resolveVoiceBridgeLaunch() {
    const overrideExe = trimShellQuotes(process.env.HIVEMIND_VOICEBRIDGE_EXE);
    if (overrideExe) {
        return { command: overrideExe, args: [], source: 'HIVEMIND_VOICEBRIDGE_EXE' };
    }

    const exeCandidates = [
        path.join(repoRoot, 'VoiceBridge', 'bin', 'Release', 'net8.0', 'VoiceBridge.exe'),
        path.join(repoRoot, 'VoiceBridge', 'bin', 'Debug', 'net8.0', 'VoiceBridge.exe'),
    ];
    const exeCandidate = newestExistingPath(exeCandidates);
    if (exeCandidate) {
        return { command: exeCandidate, args: [], source: exeCandidate };
    }

    const dllCandidates = [
        path.join(repoRoot, 'VoiceBridge', 'bin', 'Release', 'net8.0', 'VoiceBridge.dll'),
        path.join(repoRoot, 'VoiceBridge', 'bin', 'Debug', 'net8.0', 'VoiceBridge.dll'),
    ];
    const dllCandidate = newestExistingPath(dllCandidates);
    if (dllCandidate) {
        return { command: 'dotnet', args: [dllCandidate], source: dllCandidate };
    }

    return null;
}

function newestExistingPath(candidates) {
    return candidates
        .filter(candidate => fs.existsSync(candidate))
        .map(candidate => ({ candidate, mtimeMs: fs.statSync(candidate).mtimeMs }))
        .sort((left, right) => right.mtimeMs - left.mtimeMs)[0]?.candidate || null;
}

// --- Start Python Backend ---
function startPythonServer() {
    return new Promise((resolve, reject) => {
        const backendScript = trimShellQuotes(process.env.HIVEMIND_BACKEND_SCRIPT);
        const script = backendScript
            ? (path.isAbsolute(backendScript) ? backendScript : path.join(repoRoot, backendScript))
            : path.join(__dirname, '../PythonServer/ai_server.py');
        const pythonLaunch = resolvePythonLaunch();

        console.log('[ELECTRON] Starting Python server at:', script);
        console.log('[ELECTRON] Python command:', pythonLaunch.command, pythonLaunch.args.join(' '));

        const env = {
            ...process.env,
            PYTHONIOENCODING: 'utf-8',
            PYTHONUTF8: '1',
            LANG: 'en_US.UTF-8'  // Unix-style, but helps on some systems
        };

        pyProc = spawn(pythonLaunch.command, [...pythonLaunch.args, '-X', 'utf8', script], {
            stdio: ['pipe', 'pipe', 'pipe'],
            shell: false,
            env: env
        });

        let resolved = false;

        pyProc.stdout.on('data', (data) => {
            // Use Buffer to ensure proper UTF-8 handling
            const buffer = Buffer.from(data);
            const output = buffer.toString('utf8');
            const visibleOutput = output
                .split(/\r?\n/)
                .filter(line => !line.trim().startsWith('[SPEECH_EVENT]'))
                .join('\n')
                .trimEnd();
            if (visibleOutput.trim()) {
                console.log(`[PYTHON]: ${visibleOutput}`);
            }

            // Detect speech events from Python
            if (output.includes('[SPEECH_EVENT] start')) {
                console.log('[PYTHON] JARVIS speaking started');
                win?.webContents?.send('speech-event', { type: 'jarvis_speak_start' });
            }
            if (output.includes('[SPEECH_EVENT] end')) {
                console.log('[PYTHON] JARVIS speaking ended');
                win?.webContents?.send('speech-event', { type: 'jarvis_speak_end' });
            }

            // Detect port
            const portMatch = output.match(/Server started successfully on port\s+(\d+)/i);
            if (portMatch && !resolved) {
                resolved = true;
                const port = parseInt(portMatch[1]);
                console.log(`[ELECTRON] Python backend detected on port ${port}`);
                resolve(port);
            }
        });

        pyProc.stderr.on('data', (data) => {
            const output = data.toString();
            // Check if this is actually an error or just normal logging
            if (output.includes('ERROR') || output.includes('Error') || output.includes('Exception') || output.includes('Traceback')) {
                console.error(`[PYTHON ACTUAL ERROR]: ${output}`);
            } else {
                console.log(`[PYTHON LOG]: ${output}`);
            }
        });

        pyProc.on('close', (code) => {
            console.log(`[PYTHON] process exited with code ${code}`);
            if (!resolved) reject(new Error('Python process closed before server start'));
        });

        pyProc.on('error', (error) => {
            console.error('[PYTHON] failed to start:', error);
            if (!resolved) reject(error);
        });
    });
}

// --- Launch C# Bridge ---
function startVoiceBridge(port) {
    const launch = resolveVoiceBridgeLaunch();
    if (!launch) {
        console.error('[ELECTRON ERROR] VoiceBridge build not found. Run dotnet build VoiceBridge\\VoiceBridge.csproj, or set HIVEMIND_VOICEBRIDGE_EXE.');
        return;
    }

    console.log('[ELECTRON] Launching C# Voice Bridge:', launch.source);

    voiceProc = spawn(launch.command, [...launch.args, `http://localhost:${port}`], {
        shell: false,
        env: {
            ...process.env,
            HIVEMIND_GRPC_URL: `http://localhost:${port}`
        }
    });

    voiceProc.stdout.on('data', data => {
        const output = data.toString().trim();
        console.log(`[VOICE] ${output}`);

        // Detect RECORDING events (user voice input)
        if (output.includes('[RECORDING_EVENT] start')) {
            console.log('[VOICE] User recording started');
            win?.webContents?.send('speech-event', { type: 'recording_start' });
        }
        if (output.includes('[RECORDING_EVENT] end')) {
            console.log('[VOICE] User recording ended');
            win?.webContents?.send('speech-event', { type: 'recording_end' });
        }
        if (output.includes('[RECORDING_EVENT] cancelled')) {
            console.log('[VOICE] User recording cancelled');
            win?.webContents?.send('speech-event', { type: 'recording_cancelled' });
        }
        const partialMatch = output.match(/\[VoiceBridge\] Partial:\s*(.+)/);
        if (partialMatch && partialMatch[1]) {
            win?.webContents?.send('speech-event', {
                type: 'recognition_partial',
                text: partialMatch[1].trim(),
            });
        }
        if (output.includes('[VoiceBridge] Error:')) {
            const message = output.split('[VoiceBridge] Error:').pop()?.trim() || 'Voice capture failed.';
            console.error(`[VOICE] Capture error: ${message}`);
            win?.webContents?.send('speech-event', { type: 'voice_error', message });
        }

        // Detect recognized speech from VoiceBridge
        const recognizedMatch = output.match(/\[VoiceBridge\] Recognized:\s*(.+)/);
        if (recognizedMatch && recognizedMatch[1]) {
            const base64Text = recognizedMatch[1].trim();

            try {
                // Decode from Base64
                const recognizedText = Buffer.from(base64Text, 'base64').toString('utf8');
                console.log(`[VOICE] Decoded recognized text: "${recognizedText}"`);

                win?.webContents?.send('speech-event', {
                    type: 'recognized',
                    text: recognizedText
                });
            } catch (error) {
                console.error('[VOICE] Failed to decode Base64:', error);
            }
        }
    });

    voiceProc.stderr.on('data', data => console.error(`[VOICE ERR] ${data}`));

    voiceProc.on('exit', code => {
        console.log(`[VOICE] Bridge exited with code ${code}`);
        voiceProc = null;
    });

    voiceProc.on('error', error => {
        console.error('[VOICE ERR] Failed to start VoiceBridge:', error);
        voiceProc = null;
    });
}


// --- Initialize gRPC Client ---
function initGrpcClient(port) {
    console.log(`[ELECTRON] Setting up gRPC client for port ${port}...`);

    const PROTO_PATH = path.join(__dirname, '../Protos/ai_service.proto');
    const packageDef = protoLoader.loadSync(PROTO_PATH, {
        keepCase: true,
        longs: String,
        enums: String,
        defaults: true,
        oneofs: true,
    });

    const aiProto = grpc.loadPackageDefinition(packageDef).AIService;
    grpcClient = new aiProto(`localhost:${port}`, grpc.credentials.createInsecure());
    console.log('[ELECTRON] gRPC client initialized successfully.');
    initRuntimeGrpcClients(port);
}

function initRuntimeGrpcClients(port) {
    try {
        const PROTO_PATH = path.join(__dirname, '../Protos/runtime_contract.proto');
        const packageDef = protoLoader.loadSync(PROTO_PATH, {
            keepCase: true,
            longs: String,
            enums: String,
            defaults: true,
            oneofs: true,
        });

        const runtimeProto = grpc.loadPackageDefinition(packageDef);
        runtimeClient = new runtimeProto.RuntimeService(`localhost:${port}`, grpc.credentials.createInsecure());
        runtimeMemoryClient = new runtimeProto.RuntimeMemoryService(`localhost:${port}`, grpc.credentials.createInsecure());
        runtimeModelClient = new runtimeProto.RuntimeModelService(`localhost:${port}`, grpc.credentials.createInsecure());
        console.log('[ELECTRON] Runtime gRPC clients initialized successfully.');
    } catch (error) {
        runtimeClient = null;
        runtimeMemoryClient = null;
        runtimeModelClient = null;
        console.error('[ELECTRON ERROR] Runtime gRPC client initialization failed:', error);
    }
}

function processQuery(inputText) {
    return new Promise((resolve, reject) => {
        if (!grpcClient) {
            console.error('[ELECTRON ERROR] gRPC client not initialized!');
            return reject(new Error('gRPC client not connected'));
        }

        grpcClient.ProcessQuery({ input_text: inputText }, (err, response) => {
            if (err) {
                console.error('[ELECTRON ERROR] gRPC request failed:', err);
                reject(err);
            } else {
                resolve(response);
            }
        });
    });
}

function runRuntimeCapability(toolName, capabilityName, params) {
    return new Promise((resolve, reject) => {
        if (!runtimeClient) {
            return reject(new Error('Runtime gRPC client not connected'));
        }

        runtimeClient.RunCapability({
            capability: {
                tool_name: toolName,
                capability_name: capabilityName,
            },
            input_json: JSON.stringify(params || {}),
        }, (err, response) => {
            if (err) {
                console.error('[ELECTRON ERROR] Runtime capability request failed:', err);
                reject(err);
                return;
            }

            let parsed = {};
            try {
                parsed = JSON.parse(response.output_json || '{}');
            } catch {
                parsed = { text: response.output_json || '' };
            }

            resolve({
                status: response.status || 'unknown',
                run_id: response.run?.run_id || '',
                output_json: response.output_json || '',
                error_message: response.error_message || '',
                response_text: parsed.text || response.error_message || response.output_json || 'No response.',
                ai_source: 'RuntimeService',
            });
        });
    });
}

function grpcUnary(client, method, request = {}) {
    return new Promise((resolve, reject) => {
        if (!client) {
            return reject(new Error('Runtime gRPC client not connected'));
        }
        client[method](request, (err, response) => {
            if (err) {
                reject(err);
                return;
            }
            resolve(response || {});
        });
    });
}

function parseJsonOrDefault(text, fallback) {
    if (!text) return fallback;
    try {
        return JSON.parse(text);
    } catch {
        return fallback;
    }
}

function capabilityFromGrpc(capability) {
    return {
        name: capability.capability_name || '',
        description: capability.description || '',
        input_schema: parseJsonOrDefault(capability.input_schema_json, {}),
        safety_level: capability.safety_level || '',
        supports_progress: Boolean(capability.supports_progress),
        supports_cancel: Boolean(capability.supports_cancel),
        examples: parseJsonOrDefault(capability.examples_json, []),
    };
}

function toolsFromCapabilities(capabilities) {
    const byName = new Map();
    for (const capability of capabilities || []) {
        const toolName = capability.tool_name || 'unknown';
        if (!byName.has(toolName)) {
            byName.set(toolName, {
                name: toolName,
                description: capability.tool_description || '',
                root: capability.root || '',
                runtime: capability.runtime || '',
                available: Boolean(capability.available),
                capabilities: [],
            });
        }
        byName.get(toolName).capabilities.push(capabilityFromGrpc(capability));
    }
    return Array.from(byName.values());
}

function mapToolRun(run) {
    return {
        tool_name: run.tool_name || '',
        capability: run.capability_name || '',
        status: run.status || '',
        input_json: run.input_json || '',
        artifact_path: run.artifact_path || '',
        metrics_json: run.metrics_json || '',
        evidence_status: run.evidence_status || '',
        started_at: run.started_at || '',
        finished_at: run.finished_at || '',
    };
}

function mapExecutionPlan(plan) {
    return {
        id: plan.id || '',
        intent: plan.intent || '',
        source: plan.source || '',
        status: plan.status || '',
        created_at: plan.created_at || '',
        finished_at: plan.finished_at || '',
        result_summary: plan.result_summary || '',
        error: plan.error || '',
        steps: parseJsonOrDefault(plan.steps_json, []),
    };
}

function mapMemoryHit(hit) {
    return {
        project: hit.project || '',
        source_path: hit.source_path || '',
        title: hit.title || '',
        chunk_index: Number(hit.chunk_index || 0),
        snippet: hit.snippet || '',
        score: Number(hit.score || 0),
    };
}

function mapModel(model) {
    return {
        name: model.name || '',
        provider: model.provider || '',
        kind: model.kind || '',
        purpose: model.purpose || '',
        model_name: model.model_name || '',
        available: Boolean(model.available),
        status: model.status || '',
        priority: Number(model.priority || 0),
        metadata: parseJsonOrDefault(model.metadata_json, {}),
    };
}

function summarizeResponseForLog(response) {
    const text = response?.response_text || '';
    const source = response?.ai_source || 'Unknown';
    return `${source}, ${text.length} chars`;
}

async function listRuntimeModelsDirect() {
    const response = await grpcUnary(runtimeModelClient, 'ListModels', {});
    return {
        model_count: Number(response.model_count || 0),
        available_model_count: Number(response.available_model_count || 0),
        models: (response.models || []).map(mapModel),
        ai_source: 'RuntimeModelService',
    };
}

async function buildRuntimeSnapshotDirect() {
    const [capabilityResponse, healthResponse, statsResponse, runsResponse, plansResponse, modelSnapshot] = await Promise.all([
        grpcUnary(runtimeClient, 'ListCapabilities', {}),
        grpcUnary(runtimeClient, 'HealthCheck', {}),
        grpcUnary(runtimeMemoryClient, 'GetMemoryStats', {}),
        grpcUnary(runtimeMemoryClient, 'RecentToolRuns', { limit: 8 }),
        grpcUnary(runtimeMemoryClient, 'RecentExecutionPlans', { limit: 8 }),
        listRuntimeModelsDirect(),
    ]);

    const tools = toolsFromCapabilities(capabilityResponse.capabilities || []);
    const memory = parseJsonOrDefault(statsResponse.stats_json, {});
    return {
        generated_at: new Date().toISOString(),
        tool_count: tools.length,
        available_tool_count: tools.filter(tool => tool.available).length,
        tools,
        memory,
        recent_tool_runs: (runsResponse.runs || []).map(mapToolRun),
        recent_execution_plans: (plansResponse.plans || []).map(mapExecutionPlan),
        health: {
            status: healthResponse.status || 'unknown',
            details: healthResponse.details || [],
        },
        models: modelSnapshot,
        ai_source: 'RuntimeService',
    };
}

async function searchRuntimeMemoryDirect(query, project = '', limit = 5) {
    const response = await grpcUnary(runtimeMemoryClient, 'SearchMemory', {
        query: query || '',
        project: project || '',
        limit: limit || 5,
    });
    return {
        hits: (response.hits || []).map(mapMemoryHit),
        ai_source: 'RuntimeMemoryService',
    };
}

async function indexRuntimeMemoryDirect(projectNames = [], maxFilesPerProject = 24) {
    const response = await grpcUnary(runtimeMemoryClient, 'IndexProjectMemory', {
        project_names: projectNames,
        include_manifests: true,
        max_files_per_project: maxFilesPerProject || 24,
    });
    return {
        results: response.results || [],
        total_indexed_document_count: Number(response.total_indexed_document_count || 0),
        ai_source: 'RuntimeMemoryService',
    };
}

async function loadRuntimeModelDirect(modelName) {
    const response = await grpcUnary(runtimeModelClient, 'LoadModel', { model_name: modelName || '' });
    return {
        success: Boolean(response.success),
        status: response.status || '',
        message: response.message || '',
        model: mapModel(response.model || {}),
        ai_source: 'RuntimeModelService',
    };
}

async function unloadRuntimeModelDirect(modelName) {
    const response = await grpcUnary(runtimeModelClient, 'UnloadModel', { model_name: modelName || '' });
    return {
        success: Boolean(response.success),
        status: response.status || '',
        message: response.message || '',
        model: mapModel(response.model || {}),
        ai_source: 'RuntimeModelService',
    };
}

async function probeRuntimeModelDirect(modelName, prompts = []) {
    const response = await grpcUnary(runtimeModelClient, 'ProbeModel', {
        model_name: modelName || '',
        prompts: prompts.map((prompt) => ({
            category: prompt.category || '',
            prompt: prompt.prompt || '',
            expected_hint: prompt.expected_hint || '',
        })),
    });
    return {
        success: Boolean(response.success),
        status: response.status || '',
        message: response.message || '',
        model: mapModel(response.model || {}),
        results: (response.results || []).map((item) => ({
            category: item.category || '',
            prompt: item.prompt || '',
            response_text: item.response_text || '',
            expected_hint: item.expected_hint || '',
            status: item.status || '',
            warning: item.warning || '',
            latency_ms: Number(item.latency_ms || 0),
        })),
        ai_source: 'RuntimeModelService',
    };
}

// --- Create Browser Window ---
async function createWindow() {
    console.log('[ELECTRON] App ready, starting backend...');
    let port = 50051;

    try {
        port = await startPythonServer();
        console.log(`[ELECTRON] Backend running on port ${port}`);
        initGrpcClient(port);
        startVoiceBridge(port);
    } catch (err) {
        console.error('[ELECTRON ERROR] Failed to start Python backend:', err);
    }

    console.log('[ELECTRON] Creating BrowserWindow...');
    win = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false, // Make sure this is false
            enableRemoteModule: false,
        },
    });

    const indexPath = path.join(__dirname, '../ui/dist/index.html');
    console.log('[ELECTRON] Loading file:', indexPath);

    win.loadFile(indexPath).catch((err) => console.error('[ELECTRON ERROR] Failed to load UI:', err));

    win.webContents.on('did-finish-load', () => {
        console.log('[ELECTRON] UI loaded successfully!');
        win.webContents.send('backend-port', port);
    });

    win.on('closed', () => {
        console.log('[ELECTRON] Window closed, stopping backend...');
        if (pyProc) pyProc.kill();
        if (voiceProc) voiceProc.kill();
        pyProc = null;
        voiceProc = null;
    });
}

// --- IPC to trigger voice input ---
ipcMain.handle('start-voice', async (event, lang) => {
    if (!voiceProc) {
        console.error('[ELECTRON ERROR] VoiceBridge not running!');
        return { success: false, message: 'VoiceBridge is not running.' };
    }

    console.log(`[ELECTRON] Sending voice trigger: ${lang}`);
    voiceProc.stdin.write(`${lang}\n`);
    return { success: true, message: 'Voice capture started.' };
});

ipcMain.handle('stop-voice', async () => {
    if (!voiceProc) {
        return { success: false, message: 'VoiceBridge is not running.' };
    }

    console.log('[ELECTRON] Sending voice stop request');
    voiceProc.stdin.write('stop\n');
    return { success: true, message: 'Voice capture stop requested.' };
});

// --- IPC: React → Electron → gRPC → Python ---
ipcMain.handle('send-message', async (event, message) => {
    console.log(`[ELECTRON] Received message from UI: "${message}"`);

    const response = await processQuery(message);
    console.log(`[ELECTRON] gRPC response received (${summarizeResponseForLog(response)})`);
    return {
        response_text: response.response_text,
        ai_source: response.ai_source || 'Unknown',
    };
});

ipcMain.handle('runtime-snapshot', async () => {
    console.log('[ELECTRON] Runtime snapshot requested');

    try {
        return await buildRuntimeSnapshotDirect();
    } catch (error) {
        console.warn('[ELECTRON] Direct runtime snapshot failed; falling back to compatibility bridge.', error.message);
        const response = await processQuery('__hivemind_runtime__:runtime snapshot json');
        try {
            const snapshot = JSON.parse(response.response_text);
            return {
                ...snapshot,
                ai_source: response.ai_source,
            };
        } catch (parseError) {
            console.error('[ELECTRON ERROR] Runtime snapshot parse failed:', parseError);
            return {
                generated_at: new Date().toISOString(),
                tool_count: 0,
                available_tool_count: 0,
                tools: [],
                memory: {},
                recent_tool_runs: [],
                recent_execution_plans: [],
                error: response.response_text || parseError.message,
                ai_source: response.ai_source || 'RuntimeOrchestrator',
            };
        }
    }
});

ipcMain.handle('runtime-command', async (event, command) => {
    console.log(`[ELECTRON] Runtime command requested: "${command}"`);

    const response = await processQuery(`__hivemind_runtime__:${command}`);
    return {
        response_text: response.response_text,
        ai_source: response.ai_source,
    };
});

ipcMain.handle('runtime-run-capability', async (event, request) => {
    const toolName = request?.toolName;
    const capabilityName = request?.capabilityName;
    const params = request?.params || {};
    console.log(`[ELECTRON] Runtime gRPC capability requested: ${toolName}.${capabilityName}`);

    if (!toolName || !capabilityName) {
        return {
            status: 'failed',
            response_text: 'toolName and capabilityName are required.',
            error_message: 'toolName and capabilityName are required.',
            ai_source: 'RuntimeService',
        };
    }

    try {
        return await runRuntimeCapability(toolName, capabilityName, params);
    } catch (error) {
        console.warn('[ELECTRON] Runtime gRPC capability failed; falling back to compatibility command bridge.', error.message);
        const bridgeCommand = `run capability ${toolName}.${capabilityName} ${JSON.stringify(params)}`;
        const response = await processQuery(`__hivemind_runtime__:${bridgeCommand}`);
        return {
            status: 'completed',
            response_text: response.response_text,
            error_message: '',
            ai_source: response.ai_source || 'RuntimeOrchestrator',
        };
    }
});

ipcMain.handle('runtime-read-artifact', async (event, request) => {
    const artifactPath = request?.artifactPath || '';
    console.log(`[ELECTRON] Runtime artifact read requested: ${artifactPath}`);

    try {
        return readRuntimeArtifact(repoRoot, artifactPath);
    } catch (error) {
        console.warn('[ELECTRON] Runtime artifact read failed.', error.message);
        return {
            artifact_path: '',
            artifact: null,
            raw_text: '',
            error_message: error.message || 'Runtime artifact read failed.',
            ai_source: 'RuntimeArtifactStore',
        };
    }
});

ipcMain.handle('runtime-search-memory', async (event, request) => {
    const query = request?.query || '';
    const project = request?.project || '';
    const limit = request?.limit || 5;
    console.log(`[ELECTRON] Runtime memory search requested: "${query}"`);

    try {
        return await searchRuntimeMemoryDirect(query, project, limit);
    } catch (error) {
        console.warn('[ELECTRON] Direct runtime memory search failed; falling back to command bridge.', error.message);
        const command = project
            ? `search memory ${query} project ${project}`
            : `search memory ${query}`;
        const response = await processQuery(`__hivemind_runtime__:${command}`);
        return {
            hits: [],
            response_text: response.response_text,
            error_message: '',
            ai_source: response.ai_source || 'RuntimeOrchestrator',
        };
    }
});

ipcMain.handle('runtime-index-memory', async (event, request) => {
    const projectNames = Array.isArray(request?.projectNames) ? request.projectNames : [];
    const maxFilesPerProject = request?.maxFilesPerProject || 24;
    console.log(`[ELECTRON] Runtime memory index requested: ${projectNames.join(', ') || 'all projects'}`);

    try {
        return await indexRuntimeMemoryDirect(projectNames, maxFilesPerProject);
    } catch (error) {
        console.warn('[ELECTRON] Direct runtime memory indexing failed; falling back to command bridge.', error.message);
        const response = await processQuery('__hivemind_runtime__:refresh project memory');
        return {
            results: [],
            total_indexed_document_count: 0,
            response_text: response.response_text,
            error_message: '',
            ai_source: response.ai_source || 'RuntimeOrchestrator',
        };
    }
});

ipcMain.handle('runtime-list-models', async () => {
    console.log('[ELECTRON] Runtime model list requested');

    try {
        return await listRuntimeModelsDirect();
    } catch (error) {
        console.warn('[ELECTRON] Direct runtime model list failed; falling back to snapshot bridge.', error.message);
        const response = await processQuery('__hivemind_runtime__:runtime snapshot json');
        const snapshot = JSON.parse(response.response_text || '{}');
        return {
            ...(snapshot.models || { model_count: 0, available_model_count: 0, models: [] }),
            ai_source: response.ai_source || 'RuntimeOrchestrator',
        };
    }
});

ipcMain.handle('runtime-load-model', async (event, request) => {
    const modelName = request?.modelName || '';
    console.log(`[ELECTRON] Runtime model load requested: ${modelName}`);

    try {
        return await loadRuntimeModelDirect(modelName);
    } catch (error) {
        console.warn('[ELECTRON] Direct runtime model load failed; falling back to command bridge.', error.message);
        const response = await processQuery(`__hivemind_runtime__:load ${modelName}`);
        return {
            success: !/failed|unsupported|unknown/i.test(response.response_text || ''),
            status: 'bridge',
            message: response.response_text || '',
            model: {},
            ai_source: response.ai_source || 'RuntimeOrchestrator',
        };
    }
});

ipcMain.handle('runtime-unload-model', async (event, request) => {
    const modelName = request?.modelName || '';
    console.log(`[ELECTRON] Runtime model unload requested: ${modelName}`);

    try {
        return await unloadRuntimeModelDirect(modelName);
    } catch (error) {
        console.warn('[ELECTRON] Direct runtime model unload failed; falling back to command bridge.', error.message);
        const response = await processQuery(`__hivemind_runtime__:unload ${modelName}`);
        return {
            success: !/failed|unsupported|unknown/i.test(response.response_text || ''),
            status: 'bridge',
            message: response.response_text || '',
            model: {},
            ai_source: response.ai_source || 'RuntimeOrchestrator',
        };
    }
});

ipcMain.handle('runtime-probe-model', async (event, request) => {
    const modelName = request?.modelName || '';
    const prompts = Array.isArray(request?.prompts) ? request.prompts : [];
    console.log(`[ELECTRON] Runtime model probe requested: ${modelName}`);

    try {
        return await probeRuntimeModelDirect(modelName, prompts);
    } catch (error) {
        console.warn('[ELECTRON] Direct runtime model probe failed.', error.message);
        return {
            success: false,
            status: 'failed',
            message: error.message || 'Runtime model probe failed.',
            model: {},
            results: [],
            ai_source: 'RuntimeModelService',
        };
    }
});

// --- App lifecycle ---
app.whenReady().then(createWindow);
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('will-quit', () => {
    if (pyProc) pyProc.kill();
    if (voiceProc) voiceProc.kill();
});

// --- IPC: Interrupt Speech ---
ipcMain.handle('interrupt-speech', async (event) => {
    return new Promise((resolve, reject) => {
        if (!grpcClient) {
            console.error('[ELECTRON ERROR] gRPC client not initialized!');
            return reject(new Error('gRPC client not connected'));
        }

        grpcClient.InterruptSpeech({}, (err, response) => {
            if (err) {
                console.error('[ELECTRON ERROR] Interrupt request failed:', err);
                reject(err);
            } else {
                if (response.message !== 'No active speech to interrupt') {
                    console.log(`[ELECTRON] Interrupt response: ${response.message}`);
                }
                resolve(response.success);
            }
        });
    });
});
