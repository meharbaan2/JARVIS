const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    sendMessage: async (text) => ipcRenderer.invoke('send-message', text),
    interruptSpeech: async () => ipcRenderer.invoke('interrupt-speech'),

    onSpeechEvent: (callback) => {
        // Remove any existing listeners to prevent duplicates
        ipcRenderer.removeAllListeners('speech-event');
        ipcRenderer.on('speech-event', (event, data) => {
            console.log('[PRELOAD] Forwarding speech event to UI:', data);
            callback(data);
        });
    },
    triggerVoice: (language) => ipcRenderer.invoke('start-voice', language)

});