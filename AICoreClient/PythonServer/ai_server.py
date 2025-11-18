import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import grpc
from concurrent import futures
import ai_service_pb2
import ai_service_pb2_grpc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from mistralai import Mistral
import os
import torch
import logging
import socket
import re
from accelerate import infer_auto_device_map
import pyttsx3
import win32com.client
import threading
import queue
from datetime import datetime
import time
import requests
import geocoder
import sqlite3
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import tempfile
from langdetect import detect
from pydub.utils import which
AudioSegment.converter = which("C:/ffmpeg/bin/ffmpeg.exe")# Add to existing imports
import speech_recognition as sr
import io
import sys
import codecs
from transformers.cache_utils import DynamicCache
import types

# COMPREHENSIVE UTF-8 SETUP FOR WINDOWS
if sys.platform == "win32":
    # Method 1: Reconfigure streams if available (Python 3.7+)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    
    # Method 2: Wrap streams for older Python
    else:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
    
    # Method 3: Set environment
    os.environ["PYTHONIOENCODING"] = "utf-8"
    
    # Method 4: Force UTF-8 mode
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("=== Python Server Starting with UTF-8 Encoding ===")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

speech_status_subscribers = []

def patch_dynamic_cache():
    """Patch DynamicCache to fix 'seen_tokens' attribute error"""
    original_update = DynamicCache.update

    def patched_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # Call the original update method
        original_update(self, key_states, value_states, layer_idx, cache_kwargs)
        
        # Add the seen_tokens attribute if it doesn't exist (for backward compatibility)
        if not hasattr(self, 'seen_tokens'):
            # Calculate seen_tokens based on the current cache state
            if len(self.key_cache) > 0 and self.key_cache[0] is not None:
                self.seen_tokens = self.key_cache[0].shape[2]
            else:
                self.seen_tokens = 0
    
    # Patch the get_max_length method if it's missing
    if not hasattr(DynamicCache, 'get_max_length'):
        def get_max_length(self):
            """Get the maximum sequence length that can be stored in the cache."""
            if len(self.key_cache) <= 0 or self.key_cache[0] is None:
                return 0
            return self.key_cache[0].shape[-2]  # shape: [batch, heads, seq_len, dim]
        
        DynamicCache.get_max_length = get_max_length
    
    # Apply the patch
    DynamicCache.update = patched_update
    
    # Add the seen_tokens property directly to the class
    if not hasattr(DynamicCache, 'seen_tokens'):
        DynamicCache.seen_tokens = property(lambda self: getattr(self, '_seen_tokens', 0))

# APPLY THE PATCH
patch_dynamic_cache()

class AIService(ai_service_pb2_grpc.AIServiceServicer):
    def __init__(self):
        # Configuration
        self.model_name = "WizardLMTeam/WizardMath-7B-V1.1"
        self.phi3_model_name = "microsoft/Phi-3-mini-4k-instruct"
        self.mistral_api_key = ""  # Replace with your key
        self.mistral_model = "mistral-large-latest"  # Or "mistral-medium", "mistral-small"
        self.weather_api_key = "" # openweather_key
        
        # Initialize with GPU optimization
        self.wizardmath_model = None
        self.wizardmath_tokenizer = None
        self.phi3_model = None
        self.phi3_tokenizer = None
        self._initialize_models()

        # Add speech control variables
        self.current_speech_thread = None
        self.stop_speech_event = threading.Event()
        self.is_speaking = False
        
        # Initialize Mistral client
        self.mistral_client = Mistral(api_key=self.mistral_api_key)

        # Add this voice engine initialization
        self.voice_engine = pyttsx3.init()

        # Initialize voice engine
        self._initialize_voice_engine()

        # Speak startup message
        self._announce_system_status()

        # Initialize database for long-term memory
        self.db_conn = sqlite3.connect('ai_memory.db', check_same_thread=False)
        self._init_db()

        # Short-term memory (in-memory)
        self.current_session = {
            "mistral": [],
            "wizardmath": [],
            "phi3": []
        }
        
        # Memory configuration
        self.max_tokens = 3000
        self.summary_frequency = 3  # Summarize every 3 exchanges
        self.max_summaries_to_keep = 100  # Limit database size

    def _initialize_models(self):
        """Initialize models with maximum GPU optimization"""
        try:
            # Set environment variables to reduce warnings
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Reduce verbosity

            # Verify CUDA
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available - check your PyTorch installation")
            
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")

            # Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

            # Calculate optimal device map
            max_memory = {0: "7GB", "cpu": "10GB"}  # Leave 1GB GPU buffer
            
            # Initialize tokenizer
            self.wizardmath_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.wizardmath_tokenizer.pad_token = self.wizardmath_tokenizer.eos_token

            # Load model with memory optimization
            self.wizardmath_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto",
                max_memory=max_memory,
                dtype=torch.float16,
                offload_folder="./offload",
                low_cpu_mem_usage=True
            ).eval()

            # Initialize Phi-3 Mini
            logger.info("Loading Phi-3 Mini model...")
            try:
                self.phi3_tokenizer = AutoTokenizer.from_pretrained(
                    self.phi3_model_name,
                    trust_remote_code=True
                )
                self.phi3_tokenizer.pad_token = self.phi3_tokenizer.eos_token

                self.phi3_model = AutoModelForCausalLM.from_pretrained(
                    self.phi3_model_name,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map="auto",
                    max_memory=max_memory,
                    dtype=torch.float16,
                    offload_folder="./offload",
                    low_cpu_mem_usage=True
                ).eval()

                logger.info("Phi-3 Mini loaded successfully")

            except Exception as e:
                logger.warning(f"Phi-3 Mini loading failed: {e}")
                logger.warning("Falling back to Mistral for general queries")
                self.phi3_model = None
                self.phi3_tokenizer = None

            logger.info(f"WizardMath loaded on devices: {self.wizardmath_model.hf_device_map}")
            if self.phi3_model:
                logger.info(f"Phi-3 Mini loaded on devices: {self.phi3_model.hf_device_map}")
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise RuntimeError("Failed to initialize model with GPU optimization")

    def _initialize_voice_engine(self):
        """Initialize voice settings with proper speed control"""
        try:
            self.voice_settings = {
                'voice_id': None,
                'base_rate': 165,  # Optimal JARVIS speed (150-170)
                'base_volume': 0.92,  # Slightly below max for clarity
                'base_pitch': 105,   # Slightly deeper than normal
                'engine_type': 'pyttsx3',  # default
                'dynamic_params': {
                    'emphasis_rate': 0.85,  # Slow down 15% for emphasized words
                    'question_pitch': 110,   # Higher pitch for questions
                    'statement_pitch': 100,  # Lower pitch for statements
                    'alert_volume': 1.0,     # Max volume for alerts
                    'normal_volume': 0.9,
                    'pause_factors': {
                        'comma': 0.3,      # Relative pause lengths
                        'period': 0.5,
                        'question': 0.6,
                        'exclamation': 0.4,
                        'colon': 0.4
                    }
                },
                'use_gtts': True  # Enable Google TTS for Punjabi
            }

            # Try Windows SAPI first (more reliable)
            try:
                temp_engine = win32com.client.Dispatch("SAPI.SpVoice")
                voices = temp_engine.GetVoices()
            
                preferred_voices = [
                    "Microsoft David", "Microsoft Hazel", 
                    "Microsoft George", "IVONA 2 Brian"
                ]
            
                for voice_name in preferred_voices:
                    for voice in voices:
                        if voice_name in voice.GetDescription():
                            self.voice_settings['voice'] = voice
                            self.voice_settings['engine_type'] = 'sapi'
                            self.voice_settings['rate'] = 0.5  # Slower than normal (-10 to 10)
                            break
            
                logger.info("Initialized Windows SAPI voice settings")
                return
            except Exception as e:
                logger.info(f"Couldn't initialize SAPI: {e}, falling back to pyttsx3")

            # Fall back to pyttsx3
            try:
                temp_engine = pyttsx3.init()
                voices = temp_engine.getProperty('voices')
            
                for voice in voices:
                    if 'english' in voice.id.lower() and 'male' in voice.id.lower():
                        self.voice_settings['voice_id'] = voice.id
                        break
            
                logger.info("Initialized pyttsx3 voice settings")
            except Exception as e:
                logger.error(f"Failed to initialize any voice engine: {e}")
                self.voice_settings = None
        except Exception as e:
            logger.error(f"Voice initialization error: {e}")
            self.voice_settings = None

    def RecognizeSpeech(self, request, context):
        """New gRPC method for speech recognition"""
        try:
            # SAFE debug logging
            try:
                print(f"[DEBUG] RecognizeSpeech called with language: {request.language_code}")
                print(f"[DEBUG] Audio data size: {len(request.audio_data)} bytes")
            except UnicodeEncodeError:
                pass  # Skip debug printing if encoding fails

            # Convert gRPC audio bytes to WAV
            audio = AudioSegment.from_raw(
                io.BytesIO(request.audio_data),
                sample_width=2,
                frame_rate=request.sample_rate,
                channels=1
            )
        
            # Export to WAV format
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
        
            # Recognize using Google Web API (most accurate free option)
            r = sr.Recognizer()
            with sr.AudioFile(wav_io) as source:
                audio_data = r.record(source)

                if request.language_code == "pa-IN":
                    # SAFE Punjabi processing logging
                    try:
                        print("[DEBUG] Processing Punjabi speech recognition...")
                    except UnicodeEncodeError:
                        pass
                    # Punjabi-specific processing
                    text = r.recognize_google(
                        audio_data,
                        language="pa-IN",
                        show_all=False
                    )
                    # SAFE result logging
                    try:
                        print(f"[DEBUG] Punjabi recognition result: {text}")
                    except UnicodeEncodeError:
                        safe_text = text.encode('unicode_escape').decode('ascii')
                        print(f"[DEBUG] Punjabi recognition result (Unicode): {safe_text}")

                    # Keep Gurmukhi text as-is
                    return ai_service_pb2.QueryResponse(
                        response_text=text,
                        ai_source="PunjabiSpeechRecognition"
                    )
                else:
                    # Default English processing
                    try:
                        print("[DEBUG] Processing English speech recognition...")
                    except UnicodeEncodeError:
                        pass

                    text = r.recognize_google(
                        audio_data,
                        language="en-US",
                        show_all=False
                    )
                    print(f"[DEBUG] English recognition result: {text}")
                    return ai_service_pb2.QueryResponse(
                        response_text=text,
                        ai_source="EnglishSpeechRecognition"
                    )
                
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            import traceback
            error_trace = traceback.format_exc()
            try:
                print(f"[DEBUG] Recognition error: {error_trace}")
            except UnicodeEncodeError:
                pass
            
            return ai_service_pb2.QueryResponse(
                response_text="ਅਸਮਰੱਥ (Unable to process)" if request.language_code == "pa-IN" else "Unable to process",
                ai_source="System"
            )

    def _speak_punjabi(self, text):
        """Robust Punjabi TTS with proper cleanup"""
        temp_path = None
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                temp_path = tmp.name
        
            # Generate speech
            tts = gTTS(text=text, lang='pa', slow=False)
            tts.save(temp_path)
        
            # Verify file
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise RuntimeError("Empty audio file generated")

            # Play audio synchronously
            sound = AudioSegment.from_mp3(temp_path)
            play(sound)  # This will block until playback completes
        
        except Exception as e:
            logger.error(f"Punjabi TTS failed: {e}")
            # Fallback to English
            self._speak_response(f"[Punjabi unavailable] {text}")
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            

    def GetSystemStatus(self, request, context):
        """gRPC endpoint for system status and weather"""
        greeting = self._get_time_based_greeting()
        location, coords = self._get_location()
        weather = self._get_weather(*coords)
    
        # Cross-platform time formatting
        now = datetime.now()
        hour = now.hour % 12
        if hour == 0:
            hour = 12
        ampm = "a.m." if now.hour < 12 else "p.m."
        current_time = f"{hour}:{now.minute:02d} {ampm}"

        if not weather:
            return ai_service_pb2.SystemStatus(
                greeting=greeting,
                weather_report="Weather systems offline",
                location=location,
                local_time=datetime.now().strftime("%H:%M")
            )

        # Convert wind speed to more natural phrasing / Not using currently
        wind_speed = weather['wind']
        if wind_speed < 0.3:
            wind_desc = "calm conditions"
        elif wind_speed < 1.5:
            wind_desc = f"a light breeze at {wind_speed} meters per second"
        else:
            wind_desc = f"winds of {wind_speed} meters per second"

        # Build the detailed weather report
        detailed_report = (
            f"{greeting}, it's {current_time}. The weather in {location} is {weather['temp']}°C "
            f"with {weather['condition']}. Humidity is at {weather['humidity']}% "
            f"and wind speed is {wind_speed} meters per second."
        )

        return ai_service_pb2.SystemStatus(
            greeting=greeting,
            weather_report=detailed_report,
            location=location,
            temperature=weather['temp'],
            conditions=weather['condition'],
            wind_speed=weather['wind'],
            humidity=weather['humidity'],
            local_time=datetime.now().strftime("%H:%M")
        )

    def _announce_system_status(self):
        """J.A.R.V.I.S-style startup announcement"""
        status = self.GetSystemStatus(ai_service_pb2.Empty(), None)
        message = status.weather_report
        print(f"J.A.R.V.I.S: {message}")
        self._speak_response(message)

    def _get_time_based_greeting(self):
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "Good morning"
        elif 12 <= hour < 17:
            return "Good afternoon"
        elif 17 <= hour < 21:
            return "Good evening"
        else:
            return "Good night"

    def _get_location(self):
        try:
            g = geocoder.ip('me')
            return g.city, g.latlng
        except Exception:
            return "Malibu", (34.0259, -118.7798)  # Default to Tony Stark's home

    def _get_weather(self, lat, lng):
            try:
                url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lng}&appid={self.weather_api_key}&units=metric"
                response = requests.get(url, timeout=3)
                response.raise_for_status()
                data = response.json()
                return {
                    'temp': data['main']['temp'],
                    'condition': data['weather'][0]['description'],
                    'humidity': data['main']['humidity'],
                    'wind': data['wind']['speed'],
                    'location': data.get('name', 'Current location')
                }
            except Exception as e:
                logger.error(f"Weather API error: {e}")
                return None

    def _jarvis_style_response(self, text, ai_source):
        """Format responses with J.A.R.V.I.S mannerisms"""

        # First clean the text
        cleaned_text = self._clean_response_text(text)

        # Add appropriate prefixes based on query type
        if ai_source == "WizardMath": #"DeepSeek-Math":
            prefixes = ["The solution appears to be", 
                       "Calculations complete:",
                       "Mathematically speaking,"]
        elif ai_source == "Phi-3":
            prefixes = ["Analysis complete:",
                       "My assessment indicates,",
                       "Based on my reasoning,"]
        else:
            prefixes = ["According to my analysis,",
                       "My research indicates,",
                       "I would suggest,"]
    
        # Randomly select a prefix
        import random
        prefix = random.choice(prefixes)
    
        # Clean up the response text
        cleaned_text = cleaned_text.strip()
        if not cleaned_text.endswith(('.','!','?')):
            cleaned_text += '.'
    
        return f"{prefix} {cleaned_text[0].lower() + cleaned_text[1:]}"  # Lowercase first letter after prefix
    
    def _speak_response(self, text):
        """Proper speech implementation with reliable interruption"""
        if not self.voice_settings:
            print(f"\nJ.A.R.V.I.S: {text}\n")
            return

        # Stop any ongoing speech FIRST
        self.stop_current_speech()

        print(f"\nJ.A.R.V.I.S: {text}\n")

        def speak_job():
            try:
                self._notify_speech_status("start", text)
                self.is_speaking = True
            
                # Split into segments by language
                segments = self._split_text_by_language(text)
            
                for i, (text_part, lang) in enumerate(segments):
                    if self.stop_speech_event.is_set():
                        break
                    
                    if not text_part.strip():
                        continue

                    if lang == 'pa':
                        # Punjabi handling (unchanged)
                        if not self.stop_speech_event.is_set():
                            self._speak_punjabi(text_part)
                    else:
                        # ENGLISH: Use sentence-based approach with proper interruption
                        self._speak_english_sentences(text_part)
                    
                    if self.stop_speech_event.is_set():
                        break
                    
            except Exception as e:
                logger.error(f"Speech failed: {e}")
            finally:
                self._notify_speech_status("end", text)
                self.is_speaking = False
                self.stop_speech_event.clear()

        # Start speech thread
        self.current_speech_thread = threading.Thread(target=speak_job, daemon=True)
        self.current_speech_thread.start()

    def _speak_english_sentences(self, text):
        """Speak English text with reliable sentence-by-sentence control"""

        # Get dynamic parameters based on content
        speech_params = self._get_dynamic_speech_params(text)

        # Process for JARVIS-style speech
        processed_text = self._process_english_speech(text)
        sentences = self._split_into_sentences(processed_text)
    
        # Initialize engine once
        if self.voice_settings['engine_type'] == 'sapi':
            engine = win32com.client.Dispatch("SAPI.SpVoice")
            engine.Rate = self.voice_settings.get('rate', 0)
        else:
            engine = pyttsx3.init()
            if self.voice_settings.get('voice_id'):
                engine.setProperty('voice', self.voice_settings['voice_id'])
            engine.setProperty('rate', 150)
    
        try:
            for i, sentence in enumerate(sentences):
                # Check for interruption before each sentence
                if self.stop_speech_event.is_set():
                    break
                
                # Remove any emphasis markers before speaking
                clean_sentence = re.sub(r'\*(.*?)\*', r'\1', sentence)

                # For SAPI, we need to handle interruption differently
                if self.voice_settings['engine_type'] == 'sapi':
                    # SAPI doesn't have good interruption, so we use a timeout approach
                    self._sapi_speak_with_timeout(engine, clean_sentence, timeout=10)
                else:
                    # pyttsx3 - use non-blocking approach
                    engine.say(clean_sentence)
                    engine.startLoop(False)
                
                    start_time = time.time()
                    while engine.isBusy():
                        if self.stop_speech_event.is_set():
                            engine.endLoop()
                            return
                        if time.time() - start_time > 8:  # 8 second timeout per sentence
                            break
                        time.sleep(0.1)
                
                    engine.endLoop()
                
                # Add natural pause between sentences (except after last one)
                if i < len(sentences) - 1 and not self.stop_speech_event.is_set():
                    time.sleep(0.3)  # 300ms pause between sentences

                # Check for interruption after each sentence
                if self.stop_speech_event.is_set():
                    break
                
        finally:
            # Cleanup
            try:
                if self.voice_settings['engine_type'] == 'pyttsx3':
                    engine.stop()
            except:
                pass

    def _sapi_speak_with_timeout(self, engine, text, timeout=10):
        """Speak with SAPI using a timeout approach since interruption is limited"""
        # SAPI doesn't support interruption well, so we use a separate thread with timeout
        def sapi_speak():
            try:
                engine.Speak(text)
            except:
                pass
            
        speak_thread = threading.Thread(target=sapi_speak, daemon=True)
        speak_thread.start()
    
        # Wait for completion or timeout or interruption
        start_time = time.time()
        while speak_thread.is_alive():
            if self.stop_speech_event.is_set():
                # Try to interrupt by speaking empty string (SAPI workaround)
                try:
                    engine.Speak("")
                except:
                    pass
                break
            if time.time() - start_time > timeout:
                break
            time.sleep(0.1)

    def _split_text_by_language(self, text):
        """Improved language segmentation that keeps punctuation with words"""
        segments = []
        current_lang = 'en'  # Default to English
        current_segment = []
        punjabi_range = range(0x0A00, 0x0A7F)  # Gurmukhi Unicode range

        i = 0
        while i < len(text):
            char = text[i]
        
            # Check if character is Punjabi
            is_punjabi = ord(char) in punjabi_range if char.strip() else False
        
            # Handle language transitions
            if is_punjabi and current_lang != 'pa':
                if current_segment:
                    segments.append((''.join(current_segment), current_lang))
                current_segment = []
                current_lang = 'pa'
            elif not is_punjabi and current_lang == 'pa' and char.strip():
                # Look ahead to see if this is a punctuation after Punjabi
                if i+1 < len(text) and not text[i+1].strip():
                    # If next character is whitespace/punctuation, keep in Punjabi segment
                    pass
                else:
                    if current_segment:
                        segments.append((''.join(current_segment), current_lang))
                    current_segment = []
                    current_lang = 'en'
        
            current_segment.append(char)
            i += 1

        if current_segment:
            segments.append((''.join(current_segment), current_lang))
    
        return segments

    def _process_english_speech(self, text):
        """Add JARVIS speech patterns without using unsupported tags"""
    
        # First clean any remaining artifacts
        text = self._clean_response_text(text)

        # JARVIS emphasis words
        emphasis_words = {
            'alert', 'warning', 'critical', 'sir', 'important', 
            'calculating', 'analysis', 'emergency', 'priority'
        }
    
        # JARVIS-style phrasing replacements
        jarvis_phrases = {
            r'\bi\b': 'I',
            r'\bim\b': "I'm",
            r'\bid\b': "I'd", 
            r'\bive\b': "I've",
            r'\byoure\b': "you're",
            r'\bits\b': "it's",
            r'\bthats\b': "that's",
            r'\bdont\b': "don't",
            r'\bwont\b': "won't",
            r'\bcant\b': "can't",
        }
    
        # Apply phrasing corrections
        processed = text
        for pattern, replacement in jarvis_phrases.items():
            processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
    
        # Add emphasis by capitalizing key words (visual emphasis only)
        for word in emphasis_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            processed = re.sub(pattern, word.upper(), processed, flags=re.IGNORECASE)
    
        # Handle numbers for clearer pronunciation by adding spaces
        processed = re.sub(r'(\d{4,})', r' \1 ', processed)  # Long numbers
        processed = re.sub(r'(\d+)%', r'\1 percent', processed)  # Percentages
    
        # Add JARVIS-style polite phrasing
        if processed.startswith(('I need', 'I want', 'Give me')):
            processed = processed.replace('I need', 'I require', 1)
            processed = processed.replace('I want', 'I would like', 1) 
            processed = processed.replace('Give me', 'Please provide', 1)
    
        return processed

    def _get_dynamic_speech_params(self, text):
        """Calculate speech parameters that actually work with TTS engines"""
        text_lower = text.lower()
    
        # Base parameters
        base_rate = 170  # Slightly faster for JARVIS
        base_volume = 0.9
    
        # Adjust for content type
        if any(word in text_lower for word in ['alert', 'warning', 'critical', 'emergency']):
            # Urgent messages - slower and clearer
            rate = int(base_rate * 0.85)
            volume = min(1.0, base_volume * 1.1)
        elif any(word in text_lower for word in ['calculate', 'equation', 'analysis', 'processing']):
            # Technical content - normal speed
            rate = base_rate
            volume = base_volume
        elif '?' in text:
            # Questions - slightly slower
            rate = int(base_rate * 0.95)
            volume = base_volume
        else:
            # Normal conversation - standard JARVIS speed
            rate = base_rate
            volume = base_volume
    
        return {
            'rate': rate,
            'volume': volume
        }

    def _split_into_sentences(self, text):
        """Split text into natural speaking chunks"""
        # Split by punctuation but keep the delimiters
        parts = re.split('([.!?])', text)
        sentences = []
    
        # Recombine with punctuation
        i = 0
        while i < len(parts):
            if i + 1 < len(parts):
                sentence = (parts[i] + parts[i+1]).strip()
                if sentence:
                    sentences.append(sentence)
                i += 2
            else:
                last_part = parts[i].strip()
                if last_part:
                    sentences.append(last_part)
                i += 1
    
        # Further split long sentences for better pacing
        refined_sentences = []
        for sentence in sentences:
            if len(sentence.split()) > 15:  # Split long sentences
                # Use findall to keep all words including conjunctions
                pattern = r'[^,;.!?]+[,;]?|[^,;.!?]+[.!?]'
                clauses = re.findall(pattern, sentence)

                for clause in clauses:
                    clause = clause.strip()
                    if clause and len(clause.split()) > 2:  # Meaningful clause
                        refined_sentences.append(clause)
                    elif clause:
                        # Very short clause, attach to previous if possible
                        if refined_sentences:
                            refined_sentences[-1] += " " + clause
                        else:
                            refined_sentences.append(clause)
            else:
                refined_sentences.append(sentence)
    
        return refined_sentences

    def _clean_response_text(self, text):
        """Remove formatting markers, emojis, and other unwanted artifacts from AI responses"""
        if not text:
            return text
    
        # Remove markdown formatting
        text = re.sub(r'\*\*|\*|__|_|~~|`', '', text)  # Remove **bold*, *italic*, __underline__, ~~strike~~, `code`
    
        # Remove headers and other markdown
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # Remove # headers
        text = re.sub(r'^-\s*', '', text, flags=re.MULTILINE)   # Remove bullet points
        text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)  # Remove numbered lists
    
        # Remove emojis and special symbols
        text = re.sub(r'[^\w\s.,!?;:()\-@#$%&*+/=<>\[\]{}|\\]', '', text)  # Keep only text, numbers, and basic punctuation
    
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
    
        # Remove any remaining formatting artifacts
        text = re.sub(r'&[a-z]+;', '', text)  # Remove HTML entities
        text = re.sub(r'<[^>]+>', '', text)   # Remove HTML tags
    
        # Clean up common AI response artifacts
        artifacts = [
            r'^```\w*\s*',    # Code block starters
            r'\s*```\s*$',    # Code block enders  
            r'^`[^`]*`$',     # Inline code blocks
            r'^>\s*',         # Quote markers
            r'^#{2,}\s*',     # Remaining header markers
        ]
    
        for pattern in artifacts:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
    
        return text.strip()

    def stop_current_speech(self):
        """Proper speech stopping with multiple fallback methods"""
        try:
            logger.info("Stopping speech...")
        
            # Set stop flag
            self.stop_speech_event.set()
        
            # Method 1: Stop pyttsx3 if active
            if self.voice_settings and self.voice_settings['engine_type'] == 'pyttsx3':
                try:
                    # Force stop any pyttsx3 instance
                    import pyttsx3
                    temp_engine = pyttsx3.init()
                    temp_engine.stop()
                    # Give it a moment to process
                    time.sleep(0.2)
                    temp_engine.stop()
                except Exception as e:
                    logger.debug(f"Pyttsx3 stop: {e}")
        
            # Method 2: For SAPI, try to interrupt by speaking empty string
            if self.voice_settings and self.voice_settings['engine_type'] == 'sapi':
                try:
                    temp_sapi = win32com.client.Dispatch("SAPI.SpVoice")
                    temp_sapi.Speak("")
                except Exception as e:
                    logger.debug(f"SAPI stop: {e}")
        
            # Method 3: Wait for thread to finish
            if self.current_speech_thread and self.current_speech_thread.is_alive():
                self.current_speech_thread.join(timeout=1.0)
            
            # Reset state
            self.is_speaking = False
            self.stop_speech_event.clear()
        
            logger.info("Speech stopped")
        
        except Exception as e:
            logger.error(f"Error stopping speech: {e}")

    def InterruptSpeech(self, request, context):
        """gRPC method to interrupt current speech"""
        try:
            logger.info("Received speech interrupt request")
            self.stop_current_speech()
            return ai_service_pb2.InterruptResponse(success=True, message="Speech interrupted")
        except Exception as e:
            logger.error(f"Failed to interrupt speech: {e}")
            return ai_service_pb2.InterruptResponse(success=False, message=str(e))

    def _init_db(self):
            """Initialize the database tables"""
            cursor = self.db_conn.cursor()
        
            # Create tables if they don't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    original_count INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_model_type ON memory_summaries (model_type)
            ''')
        
            self.db_conn.commit()

    def _add_to_memory(self, model_type, role, content):
            """Add conversation to memory with automatic summarization"""
            try:
                # Add to current session (short-term memory)
                self.current_session[model_type].append({
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                })
            
                # Store in database (long-term raw memory)
                cursor = self.db_conn.cursor()
                cursor.execute('''
                    INSERT INTO conversation_memory (model_type, role, content)
                    VALUES (?, ?, ?)
                ''', (model_type, role, content))
                self.db_conn.commit()
            
                # Check if we need to summarize
                if len(self.current_session[model_type]) >= self.summary_frequency:
                    self._summarize_conversation(model_type)
                
            except Exception as e:
                logger.error(f"Failed to add to memory: {e}")

    def _summarize_conversation(self, model_type):
            """Create and store a summary of recent conversation"""
            try:
                recent_chat = self.current_session[model_type]
                if not recent_chat:
                    return
                
                # Prepare conversation text for summarization
                conversation_text = "\n".join(
                    f"{msg['role']}: {msg['content']}" 
                    for msg in recent_chat
                )
            
                # Get summary from Mistral
                summary_prompt = f"""Please summarize the key points from this conversation in 3-4 bullet points:
            
                {conversation_text}
            
                Summary:
                - """
            
                summary_response = self.mistral_client.chat.complete(
                    model="mistral-small",
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=150
                )
            
                summary = summary_response.choices[0].message.content
            
                # Store summary in database
                cursor = self.db_conn.cursor()
                cursor.execute('''
                    INSERT INTO memory_summaries (model_type, summary, original_count)
                    VALUES (?, ?, ?)
                ''', (model_type, summary, len(recent_chat)))
                self.db_conn.commit()
            
                # Clear current session after summarization
                self.current_session[model_type] = []
            
                # Enforce max summaries limit
                self._cleanup_old_summaries(model_type)
            
            except Exception as e:
                logger.error(f"Summarization failed: {e}")

    def _cleanup_old_summaries(self, model_type):
            """Remove oldest summaries to maintain database size limit"""
            try:
                cursor = self.db_conn.cursor()
            
                # Get count of summaries for this model
                cursor.execute('''
                    SELECT COUNT(*) FROM memory_summaries 
                    WHERE model_type = ?
                ''', (model_type,))
                count = cursor.fetchone()[0]
            
                if count > self.max_summaries_to_keep:
                    # Delete oldest entries exceeding the limit
                    cursor.execute('''
                        DELETE FROM memory_summaries
                        WHERE id IN (
                            SELECT id FROM memory_summaries
                            WHERE model_type = ?
                            ORDER BY timestamp ASC
                            LIMIT ?
                        )
                    ''', (model_type, count - self.max_summaries_to_keep))
                    self.db_conn.commit()
                
            except Exception as e:
                logger.error(f"Failed to clean up old summaries: {e}")

    def _get_recent_summaries(self, model_type, limit=2):
            """Retrieve most recent summaries from database"""
            try:
                cursor = self.db_conn.cursor()
                cursor.execute('''
                    SELECT summary FROM memory_summaries
                    WHERE model_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (model_type, limit))
            
                return [row[0] for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Failed to fetch summaries: {e}")
                return []

    def _get_context(self, model_type, new_query):
            """Build context from both short-term and long-term memory"""
            # Get recent messages from current session
            recent_messages = self.current_session[model_type].copy()
        
            # Get relevant summaries from database
            summaries = self._get_recent_summaries(model_type)
        
            # Prepare context messages
            context_messages = []
        
            # Add summaries as system messages
            for summary in summaries:
                context_messages.append({
                    "role": "system",
                    "content": f"Previous conversation summary: {summary}"
                })
        
            # Add recent messages
            for msg in recent_messages:
                context_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
            # Add new query
            context_messages.append({
                "role": "user",
                "content": new_query
            })
        
            return context_messages

    def ProcessQuery(self, request, context):
        """Process queries with full J.A.R.V.I.S personality"""
        try:
            # Display user input
            # SAFE Unicode printing for user input
            user_input = request.input_text
            try:
                print(f"\nUSER: {user_input}")
            except UnicodeEncodeError:
                # Fallback: print escaped Unicode
                safe_text = user_input.encode('unicode_escape').decode('ascii')
                print(f"\nUSER (Unicode): {safe_text}")

            # Detect if input is Punjabi
            is_punjabi = False
            try:
                # Check for Gurmukhi Unicode range directly
                punjabi_range = range(0x0A00, 0x0A7F)
                has_punjabi_chars = any(ord(char) in punjabi_range for char in user_input if char.strip())
            
                if has_punjabi_chars:
                    is_punjabi = True
                    print("Detected Punjabi text input")
                else:
                    # Fall back to langdetect
                    try:
                        is_punjabi = detect(user_input) == 'pa'
                    except:
                        is_punjabi = False
            except Exception as detect_error:
                print(f"Language detection error: {detect_error}")
                is_punjabi = False
        
            # Get response
            if self._is_math_or_code_query(request.input_text):
                raw_response = self._process_with_wizardmath(request.input_text)
                ai_source = "WizardMath"
            elif self._should_use_phi3(user_input):  # NEW: Phi-3 for general queries
                raw_response = self._process_with_phi3(user_input)
                ai_source = "Phi-3"
            else:
                raw_response = self._process_with_mistral(request.input_text)
                ai_source = "Mistral"
        
            # Apply J.A.R.V.I.S formatting
            response = self._jarvis_style_response(raw_response, ai_source)

            # Special handling for Punjabi responses
            if is_punjabi:
                response = response.replace("sir", "ਜੀ")  # Replace "sir" with Punjabi equivalent
        
            # Display and speak simultaneously
            # SAFE Unicode printing for JARVIS response
            try:
                print(f"J.A.R.V.I.S: {response}")
            except UnicodeEncodeError:
                safe_response = response.encode('unicode_escape').decode('ascii')
                print(f"J.A.R.V.I.S (Unicode): {safe_response}")

            self._speak_response(response)
        
            return ai_service_pb2.QueryResponse(
                response_text=response,
                ai_source=ai_source
                #is_punjabi=is_punjabi  # Optional: Add field to proto if needed
            )
        
        except Exception as e:
            # Print the actual error for debugging
            import traceback
            error_details = traceback.format_exc()
            print(f"EXCEPTION in ProcessQuery: {str(e)}")
            print(f"TRACEBACK: {error_details}")
    
            # Check if it's a Punjabi query and provide appropriate error message
            error_msg = "ਮਾਫ਼ ਕਰਨਾ ਜੀ, ਮੈਂ ਇਸ ਸਵਾਲ ਨੂੰ ਸਮਝ ਨਹੀਂ ਸਕਿਆ।"  # "Sorry, I couldn't understand this question"
        
            # SAFE error printing
            try:
                print(f"J.A.R.V.I.S: {error_msg}")
            except UnicodeEncodeError:
                safe_error = error_msg.encode('unicode_escape').decode('ascii')
                print(f"J.A.R.V.I.S (Unicode): {safe_error}")
            
            self._speak_response(error_msg)
            return ai_service_pb2.QueryResponse(
                response_text=error_msg,
                ai_source="System"
            )

    def _is_math_or_code_query(self, text):
        """More precise routing between DeepSeek and Mistral"""
        text_lower = text.lower()
    
        # Keywords that DEFINITELY require DeepSeek
        math_keywords = ["solve", "calculate", "equation", "integral", "derivative", "x=", "wizardmath"]
        # code_keywords = ["code", "python", "function", "algorithm", "programming", "loop"]
    
        # Questions that should go to Mistral despite keywords
        mistral_exceptions = [
            "what is", "explain", "tell me about", "history of", 
            "who invented", "why is", "how does"
        ]
    
        # Check for exceptions first
        if any(phrase in text_lower for phrase in mistral_exceptions):
            return False
    
        # Default to DeepSeek for math/code
        return any(keyword in text_lower for keyword in math_keywords) # + code_keywords)

    def _should_use_phi3(self, text):
        """Determine if Phi-3 should handle this query"""
        text_lower = text.lower()
    
        # Phi-3 handles general knowledge, reasoning, and conversation
        phi3_keywords = [
            "what is", "how", "explain", "tell me about", "why is",
            "who is", "when did", "where is", "can you", "could you",
            "would you", "should I", "what are", "how does", "help me", "whats"
        ]
    
        # Exceptions that should go to Mistral (complex/long-form)
        mistral_exceptions = [
            "write a", "create a", "generate", "long", "detailed",
            "comprehensive", "essay", "article", "story"
        ]
    
        # Check for exceptions first
        if any(phrase in text_lower for phrase in mistral_exceptions):
            return False
    
        # Use Phi-3 for general knowledge and reasoning
        return any(phrase in text_lower for phrase in phi3_keywords)

    def _process_with_wizardmath(self, prompt):
        """Process math queries with WizardMath with reliable answer extraction"""
        # Use a more structured prompt template
        template = """Please solve the following math problem carefully:
    
        Problem: {prompt}
    
        Requirements:
        1. Show each calculation step clearly
        2. Box the final answer like this: \boxed{{answer}}
        3. Verify your calculations
    
        Let's begin:
        """
    
        inputs = self.wizardmath_tokenizer(template.format(prompt=prompt), 
                    return_tensors="pt").to(self.wizardmath_model.device)
        outputs = self.wizardmath_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.1,  # Completely deterministic
            pad_token_id=self.wizardmath_tokenizer.eos_token_id
        )
    
        full_response = self.wizardmath_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        # Robust answer extraction methods
        answer = None
    
        # Method 1: Check for boxed answer \boxed{...}
        boxed_pattern = r'\\boxed\{([^}]*)\}'
        boxed_match = re.search(boxed_pattern, full_response)
        if boxed_match:
            answer = boxed_match.group(1).strip()
    
        # Method 2: Check for "Final Answer:" pattern
        if not answer and "Final Answer:" in full_response:
            answer_part = full_response.split("Final Answer:")[1].strip()
            answer = answer_part.split('\n')[0].split('.')[0].strip()
    
        # Method 3: Extract last number in response
        if not answer:
            numbers = re.findall(r'-?\d+\.?\d*', full_response)
            if numbers:
                answer = numbers[-1]
    
        # Final fallback
        if not answer:
            answer = "Could not determine the answer"
    
        return answer

    def _process_with_mistral(self, prompt):
        """Enhanced Mistral processing with memory integration"""
        try:
            # Step 1: Get context from memory system
            messages = self._get_context("mistral", prompt)
        
            # Debug logging (optional)
            logger.debug(f"Mistral API messages context:\n{messages}")
        
            # Step 2: Call Mistral API with full context
            chat_response = self.mistral_client.chat.complete(
                model=self.mistral_model,
                messages=messages,
                temperature=0.7,  # Added for more controlled responses
                max_tokens=500    # Added to prevent cutoff responses
            )
        
            response_content = chat_response.choices[0].message.content
            print(f"Received from Mistral: {response_content}")
        
            # Step 3: Store exchange in memory
            self._add_to_memory("mistral", "user", prompt)
            self._add_to_memory("mistral", "assistant", response_content)
        
            # Step 4: Return the generated response
            return response_content
        
        except Exception as e:
            print(f"Mistral API error: {e}")
            import traceback
            print(f"Mistral traceback: {traceback.format_exc()}")
        
            # Store the failed query for debugging (without response)
            self._add_to_memory("mistral", "user", f"[FAILED QUERY] {prompt}")
        
            return "I couldn't process your request. Please try again later."

    def _process_with_phi3(self, prompt):
        """Process general queries with Phi-3 Mini"""
        try:
            # Check if Phi-3 is available
            # if self.phi3_model is None or self.phi3_tokenizer is None:
            #     logger.warning("Phi-3 not available, falling back to Mistral")
            #     return self._process_with_mistral(prompt)

            # Use simpler chat formatting to avoid template issues
            chat_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        
            inputs = self.phi3_tokenizer(
                chat_prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=2048
            ).to(self.phi3_model.device)
        
            # Use simpler generation parameters to avoid cache issues
            generation_config = {
                'max_new_tokens': 256,
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9,
                'pad_token_id': self.phi3_tokenizer.eos_token_id,
                'eos_token_id': self.phi3_tokenizer.eos_token_id,
                'repetition_penalty': 1.1,
            }
        
            # Disable cache to avoid DynamicCache issues
            with torch.no_grad():
                outputs = self.phi3_model.generate(
                    **inputs,
                    **generation_config,
                    use_cache=False  # This avoids the DynamicCache entirely
                )
        
            response = self.phi3_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
            # Extract only the assistant's response
            if "<|assistant|>" in response:
                assistant_response = response.split("<|assistant|>")[-1].strip()
            else:
                assistant_response = response.replace(chat_prompt, "").strip()
        
            # Clean up any end tokens
            assistant_response = assistant_response.replace("<|end|>", "").strip()
        
            if not assistant_response or len(assistant_response) < 2:
                assistant_response = "I've processed your request. How can I assist you further?"
        
            # Store in memory
            self._add_to_memory("phi3", "user", prompt)
            self._add_to_memory("phi3", "assistant", assistant_response)
        
            return assistant_response
        
        except Exception as e:
            logger.error(f"Phi-3 processing failed: {e}")
            return "I apologize, but I'm having trouble processing your request at the moment."

    def _notify_speech_status(self, status, message=""):
        for q in speech_status_subscribers:
            q.put((status, message))

        # Also print to stdout for Electron
        if status == "start":
            print("[SPEECH_EVENT] start", flush=True)
        elif status == "end":
            print("[SPEECH_EVENT] end", flush=True)

    def StreamSpeechStatus(self, request, context):
        q = queue.Queue()
        speech_status_subscribers.append(q)
        try:
            while True:
                status, msg = q.get()
                yield ai_service_pb2.SpeechStatus(status=status, message=msg)
        except Exception as e:
            logger.error(f"Speech stream ended: {e}")
        finally:
            speech_status_subscribers.remove(q)

def find_available_port(start_port=50051, max_tries=100):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
                s.bind(('::', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_tries - 1}")

def serve():
    """Start the gRPC server with robust port handling"""
    # Configure server with port reuse
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=(('grpc.so_reuseport', 1),)  # Allow port reuse
    )
    ai_service_pb2_grpc.add_AIServiceServicer_to_server(AIService(), server)
    
    port = find_available_port()
    server.add_insecure_port(f'[::]:{port}')
    
    try:

        server.start()
        # logger.info(f"Server started successfully on port {port} (PID: {os.getpid()})", flush=True)
        print("Server started successfully on port 50051", flush=True)
        logger.info("Python gRPC server is running...")
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server gracefully...")
        server.stop(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        logger.info("Server stopped")

if __name__ == "__main__":
    # Create offload directory if needed
    if not os.path.exists("./offload"):
        os.makedirs("./offload")
    
    # Start the server
    serve()