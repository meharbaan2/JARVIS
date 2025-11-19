# JARVIS AI Assistant 🤖

A sophisticated **multilingual voice-enabled AI assistant** with real-time speech recognition, text-to-speech (English & Punjabi), and intelligent model routing. Runs fully locally or via external APIs, with conversation memory and startup weather announcements.

![AI Assistant](images/screenshot.png)
![Punjabi](images/screenshot2.png)
![Speaking](images/screenshot3.png)
![User](images/screenshot4.png)

---

## 🚀 Features

- **🎤 Voice Interface**: Real-time speech recognition & natural TTS in English & Punjabi
- **🧠 Multi-Model AI**: Intelligent routing between:
  - **WizardMath**: Math problem solving
  - **Phi-3**: General reasoning
  - **Mistral**: Advanced language understanding
- **⚡ Auto-Start**: Python backend and C# VoiceBridge launched from Electron
- **💾 Memory**: Conversation history with automatic summarization
- **🎨 Modern UI**: React/Electron interface with audio visualizations
- **🔧 Self-Contained**: Start everything with a single command

---

## 🛠 Tech Stack

- **Backend**: Python, gRPC, Transformers, PyTorch, CUDA  
- **Frontend**: Electron, React, Framer Motion  
- **Voice**: Python `speech_recognition` (local), gTTS, pyttsx3
- **Audio Processing**: pydub + FFmpeg
- **AI Models**: WizardMath-7B (local), Phi-3-mini (local), Mistral API
- **Location & Weather**: geocoder + OpenWeatherMap API
- **Memory**: SQLite local conversation database with automatic summarization
- **Bridge**: C# VoiceBridge (NAudio)

---

## 🎯 Quick Start

### Prerequisites

- Node.js 16+  
- Python 3.8+  
- .NET 8.0 (VoiceBridge)  
- FFmpeg  
- NVIDIA GPU (recommended)

### Installation & Running

1. **Clone the repository**
git clone https://github.com/yourusername/jarvis-ai-assistant.git
cd jarvis-ai-assistant

2. **Build VoiceBridge**
cd VoiceBridge
dotnet build -c Release

3. **Update VoiceBridge path in ElectronApp/main.js**
const voiceExe = 'D:/your-project-folder/VoiceBridge/bin/Release/net8.0/VoiceBridge.exe';

4. **Add API Keys (optional)**
# PythonServer/ai_server.py
self.mistral_api_key = "YOUR_MISTRAL_KEY"
self.weather_api_key = "YOUR_OPENWEATHER_KEY"

5. **Build UI dist**
cd ../ui
npm build

6. **Install dependencies & start Electron app**
cd ../electron
npm install
npm start

**Electron will automatically:**
- Start Python gRPC server 
- Launch VoiceBridge
- Load AI models 
- Open the UI
**For First-Time Setup**
- Downloads AI models
- Initializes speech systems
- Sets up conversation database

---

### Usage
- **Voice**: Click the microphone, speak naturally, switch languages
- **Text**: Type and press Enter/send
- **Punjabi** → auto-detected

---

### 🔧 Troubleshooting
- **VoiceBridge.exe not found** → ensure built & path correct
- **Python not found** → check installation & PATH
- **Punjabi TTS issues** → FFmpeg installed & internet connected
- **Slow AI models** → first-time download, GPU recommended
- **Voice recognition issues** → check microphone
