const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');

let pyProc = null;
let grpcClient = null;
let win = null;
let voiceProc = null;

// --- Start Python Backend ---
function startPythonServer() {
    return new Promise((resolve, reject) => {
        const script = path.join(__dirname, '../PythonServer/ai_server.py');
        //console.log('[ELECTRON] Starting Python server at:', script);
        //pyProc = spawn('python', [script], { stdio: ['pipe', 'pipe', 'pipe'], shell: false });

        console.log('[ELECTRON] Starting Python server at:', script);

        const env = {
            ...process.env,
            PYTHONIOENCODING: 'utf-8',
            PYTHONUTF8: '1',
            LANG: 'en_US.UTF-8'  // Unix-style, but helps on some systems
        };

        pyProc = spawn('python', ['-X', 'utf8', script], {
            stdio: ['pipe', 'pipe', 'pipe'],
            shell: false,
            env: env
        });

        let resolved = false;

        pyProc.stdout.on('data', (data) => {
            // Use Buffer to ensure proper UTF-8 handling
            const buffer = Buffer.from(data);
            const output = buffer.toString('utf8');
            console.log(`[PYTHON]: ${output}`);

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
    });
}

// --- Launch C# Bridge ---
function startVoiceBridge() {
    const voiceExe = 'D:/Coding/Hivemind/VoiceBridge/bin/Release/net8.0/VoiceBridge.exe';
    console.log('[ELECTRON] Launching C# Voice Bridge:', voiceExe);

    voiceProc = spawn(voiceExe, [], { shell: false });

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
}

// --- Create Browser Window ---
async function createWindow() {
    console.log('[ELECTRON] App ready, starting backend...');
    let port = 50051;

    try {
        port = await startPythonServer();
        console.log(`[ELECTRON] Backend running on port ${port}`);
        initGrpcClient(port);
        startVoiceBridge();
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
        pyProc = null;
        voiceProc = null;
    });
}

// --- IPC to trigger voice input ---
ipcMain.handle('start-voice', async (event, lang) => {
    if (!voiceProc) {
        console.error('[ELECTRON ERROR] VoiceBridge not running!');
        return;
    }

    console.log(`[ELECTRON] Sending voice trigger: ${lang}`);
    voiceProc.stdin.write(`${lang}\n`);
});

// --- IPC: React → Electron → gRPC → Python ---
ipcMain.handle('send-message', async (event, message) => {
    console.log(`[ELECTRON] Received message from UI: "${message}"`);

    return new Promise((resolve, reject) => {
        if (!grpcClient) {
            console.error('[ELECTRON ERROR] gRPC client not initialized!');
            return reject(new Error('gRPC client not connected'));
        }

        grpcClient.ProcessQuery({ input_text: message }, (err, response) => {
            if (err) {
                console.error('[ELECTRON ERROR] gRPC request failed:', err);
                reject(err);
            } else {
                console.log(`[ELECTRON] gRPC response: "${response.response_text}"`);
                resolve(response.response_text);
            }
        });
    });
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
    console.log('[ELECTRON] Received speech interrupt request');

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
                console.log(`[ELECTRON] Interrupt response: ${response.message}`);
                resolve(response.success);
            }
        });
    });
});