const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    sendMessage: async (text) => ipcRenderer.invoke('send-message', text),
    getRuntimeSnapshot: async () => ipcRenderer.invoke('runtime-snapshot'),
    sendRuntimeCommand: async (command) => ipcRenderer.invoke('runtime-command', command),
    runRuntimeCapability: async (request) => ipcRenderer.invoke('runtime-run-capability', request),
    readRuntimeArtifact: async (request) => ipcRenderer.invoke('runtime-read-artifact', request),
    searchRuntimeMemory: async (request) => ipcRenderer.invoke('runtime-search-memory', request),
    indexRuntimeMemory: async (request) => ipcRenderer.invoke('runtime-index-memory', request),
    listRuntimeModels: async () => ipcRenderer.invoke('runtime-list-models'),
    loadRuntimeModel: async (request) => ipcRenderer.invoke('runtime-load-model', request),
    unloadRuntimeModel: async (request) => ipcRenderer.invoke('runtime-unload-model', request),
    probeRuntimeModel: async (request) => ipcRenderer.invoke('runtime-probe-model', request),
    interruptSpeech: async () => ipcRenderer.invoke('interrupt-speech'),
    stopVoice: async () => ipcRenderer.invoke('stop-voice'),

    onSpeechEvent: (callback) => {
        // Remove any existing listeners to prevent duplicates
        ipcRenderer.removeAllListeners('speech-event');
        const listener = (event, data) => {
            console.log('[PRELOAD] Forwarding speech event to UI:', data);
            callback(data);
        };
        ipcRenderer.on('speech-event', listener);
        return () => ipcRenderer.removeListener('speech-event', listener);
    },
    triggerVoice: (language) => ipcRenderer.invoke('start-voice', language)

});
