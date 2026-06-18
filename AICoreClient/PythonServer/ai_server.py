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
import json
import gc
import hashlib
import subprocess
from pathlib import Path
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
from runtime_config import settings as runtime_settings
from memory_service import MemoryService
from tool_registry import create_default_registry
from orchestrator import RuntimeOrchestrator
from voice_utils import normalize_for_tts, split_into_speech_chunks
from embedding_service import create_embedding_provider
from model_runtime_service import RuntimeModelServiceFacade
from model_registry import create_default_model_registry
from runtime_grpc_services import register_runtime_grpc_services
from math_service import MathService
from punjabi_service import PunjabiConversationService
from speech_style import (
    casual_jarvis_reply,
    clean_model_disclaimers,
    dynamic_speech_params,
    format_jarvis_response,
    prepare_spoken_response_text,
    process_english_speech,
)

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
        self.model_name = runtime_settings.wizardmath_model
        self.phi3_model_name = runtime_settings.phi3_model
        self.mistral_api_key = runtime_settings.mistral_api_key
        self.mistral_model = runtime_settings.mistral_model
        self.weather_api_key = runtime_settings.weather_api_key
        self.embedding_provider = create_embedding_provider(runtime_settings)
        self.model_registry = create_default_model_registry(runtime_settings, self.embedding_provider)
        self.math_service = MathService()
        self.punjabi_service = PunjabiConversationService()
        math_health = self.math_service.health()
        self.model_registry.set_availability(
            "math_service",
            True,
            "available",
            metadata=math_health,
        )
        self.memory = MemoryService(runtime_settings.database_path, embedding_provider=self.embedding_provider)
        self.tool_registry = create_default_registry(runtime_settings)
        self.runtime_orchestrator = RuntimeOrchestrator(self.memory, self.tool_registry, self.model_registry)
        self.runtime_orchestrator.bootstrap_project_memory()
        
        # Initialize with GPU optimization
        self.wizardmath_model = None
        self.wizardmath_tokenizer = None
        self.phi3_model = None
        self.phi3_tokenizer = None
        self.model_lock = threading.RLock()
        self._initialize_models()

        # Add speech control variables
        self.current_speech_thread = None
        self.stop_speech_event = threading.Event()
        self.is_speaking = False
        self.speech_state_lock = threading.RLock()
        self.speech_generation = 0
        self.active_sapi_voice = None
        self.active_tts_engine = None
        self.active_audio_process = None
        self.tts_prime_lock = threading.RLock()
        self.punjabi_tts_jobs = {}
        self.punjabi_tts_cache_dir = runtime_settings.runtime_state_dir / "tts_cache" / "punjabi"
        self.punjabi_tts_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Mistral client when configured. Local/runtime features still work without it.
        self.mistral_client = Mistral(api_key=self.mistral_api_key) if self.mistral_api_key else None

        # Add this voice engine initialization
        self.voice_engine = pyttsx3.init()

        # Initialize voice engine
        self._initialize_voice_engine()

        # Speak startup message
        self._announce_system_status()

        # Compatibility handles while memory migrates out of this monolithic service.
        self.db_conn = self.memory.conn
        self.current_session = self.memory.current_session
        self.max_tokens = 3000
        self.summary_frequency = self.memory.summary_frequency
        self.max_summaries_to_keep = self.memory.max_summaries_to_keep
        self.runtime_model_service = RuntimeModelServiceFacade(
            self.model_registry,
            load_handlers={"phi3": self._load_phi3_model},
            unload_handlers={"phi3": self._unload_phi3_model},
            probe_handlers={"phi3": self._probe_phi3_model},
        )

    def _initialize_models(self):
        """Initialize models with maximum GPU optimization"""
        try:
            # Set environment variables to reduce warnings
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Reduce verbosity

            if not runtime_settings.eager_load_wizardmath and not runtime_settings.eager_load_phi3:
                logger.info("Skipping local transformer startup loads. Enable a model with HIVEMIND_EAGER_LOAD_* when needed.")
                self.model_registry.set_availability("wizardmath", False, "startup_skipped")
                self.model_registry.set_availability("phi3", False, "startup_skipped")
                return

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
            offload_folder = runtime_settings.runtime_state_dir / "model_offload"
            offload_folder.mkdir(parents=True, exist_ok=True)

            if not runtime_settings.eager_load_wizardmath:
                logger.info("Skipping WizardMath startup load. Set HIVEMIND_EAGER_LOAD_WIZARDMATH=true to load it at startup.")
                self.model_registry.set_availability("wizardmath", False, "startup_skipped")
            else:
                wizardmath_ref, wizardmath_is_local = self._resolve_model_reference(
                    self.model_name,
                    "WizardMath",
                    runtime_settings.wizardmath_revision,
                )
                wizardmath_load_options = self._transformers_load_options(
                    runtime_settings.wizardmath_revision,
                    wizardmath_is_local,
                )

                # Initialize tokenizer
                self.wizardmath_tokenizer = AutoTokenizer.from_pretrained(
                    wizardmath_ref,
                    trust_remote_code=True,
                    **wizardmath_load_options,
                )
                self.wizardmath_tokenizer.pad_token = self.wizardmath_tokenizer.eos_token

                # Load model with memory optimization
                self.wizardmath_model = AutoModelForCausalLM.from_pretrained(
                    wizardmath_ref,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map="auto",
                    max_memory=max_memory,
                    dtype=torch.float16,
                    offload_folder=str(offload_folder),
                    low_cpu_mem_usage=True,
                    **wizardmath_load_options,
                ).eval()

                logger.info(f"WizardMath loaded on devices: {self.wizardmath_model.hf_device_map}")
                self.model_registry.set_availability("wizardmath", True, "loaded")

            if runtime_settings.eager_load_phi3:
                # Initialize Phi-3 Mini
                logger.info("Loading Phi-3 Mini model...")
                try:
                    phi3_ref, phi3_is_local = self._resolve_model_reference(
                        self.phi3_model_name,
                        "Phi-3",
                        runtime_settings.phi3_revision,
                    )
                    phi3_load_options = self._transformers_load_options(
                        runtime_settings.phi3_revision,
                        phi3_is_local,
                    )

                    self.phi3_tokenizer = AutoTokenizer.from_pretrained(
                        phi3_ref,
                        trust_remote_code=True,
                        **phi3_load_options,
                    )
                    self.phi3_tokenizer.pad_token = self.phi3_tokenizer.eos_token

                    self.phi3_model = AutoModelForCausalLM.from_pretrained(
                        phi3_ref,
                        trust_remote_code=True,
                        quantization_config=quantization_config,
                        device_map="auto",
                        max_memory=max_memory,
                        dtype=torch.float16,
                        offload_folder=str(offload_folder),
                        low_cpu_mem_usage=True,
                        attn_implementation="eager",
                        **phi3_load_options,
                    ).eval()

                    logger.info("Phi-3 Mini loaded successfully")

                except Exception as e:
                    logger.warning(f"Phi-3 Mini loading failed: {e}")
                    logger.warning("Falling back to Mistral for general queries")
                    self.phi3_model = None
                    self.phi3_tokenizer = None
            else:
                logger.info("Skipping Phi-3 startup load. Set HIVEMIND_EAGER_LOAD_PHI3=true to preload it.")
                self.model_registry.set_availability("phi3", False, "startup_skipped")

            if self.phi3_model:
                logger.info(f"Phi-3 Mini loaded on devices: {self.phi3_model.hf_device_map}")
                self.model_registry.set_availability("phi3", True, "loaded")
            elif runtime_settings.eager_load_phi3:
                self.model_registry.set_availability("phi3", False, "not_loaded")
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise RuntimeError("Failed to initialize model with GPU optimization")

    def _transformers_load_options(self, revision="", local_reference=False):
        options = {}
        if runtime_settings.transformers_local_files_only or local_reference:
            logger.info("Transformers local-files-only mode is enabled.")
            options["local_files_only"] = True
        if revision and not local_reference:
            options["revision"] = revision
        return options

    def _resolve_model_reference(self, model_name, label, revision=""):
        model_path = Path(model_name).expanduser()
        if model_path.exists():
            return str(model_path), True

        if not runtime_settings.local_snapshot_fallback:
            return model_name, False

        cache_dir = self._hf_model_cache_dir(model_name)
        snapshots_dir = cache_dir / "snapshots"
        if not snapshots_dir.exists():
            return model_name, False

        preferred_revision = revision or self._read_hf_ref(cache_dir / "refs" / "main")
        if preferred_revision:
            preferred_snapshot = snapshots_dir / preferred_revision
            if self._snapshot_has_required_weights(preferred_snapshot):
                logger.info(f"{label} will load from local cached snapshot {preferred_revision}.")
                return str(preferred_snapshot), True
            if preferred_snapshot.exists():
                logger.warning(f"{label} cached snapshot {preferred_revision} is incomplete; looking for another complete local snapshot.")

        complete_snapshots = [
            snapshot for snapshot in snapshots_dir.iterdir()
            if snapshot.is_dir() and self._snapshot_has_required_weights(snapshot)
        ]
        if not complete_snapshots:
            return model_name, False

        selected = max(complete_snapshots, key=lambda item: item.stat().st_mtime)
        logger.warning(
            f"{label} current cached revision is incomplete; using complete local snapshot {selected.name}."
        )
        return str(selected), True

    def _hf_model_cache_dir(self, model_name):
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            cache_root = Path(HF_HUB_CACHE)
        except Exception:
            cache_root = Path(os.getenv("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
        return cache_root / f"models--{model_name.replace('/', '--')}"

    def _read_hf_ref(self, ref_path):
        try:
            return ref_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _snapshot_has_required_weights(self, snapshot_path):
        if not snapshot_path.exists():
            return False

        index_path = snapshot_path / "model.safetensors.index.json"
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                required_files = set(index.get("weight_map", {}).values())
            except (OSError, json.JSONDecodeError):
                return False
            return bool(required_files) and all((snapshot_path / file_name).exists() for file_name in required_files)

        return any(snapshot_path.glob("*.safetensors")) or any(snapshot_path.glob("pytorch_model*.bin"))

    def _load_phi3_model(self):
        with self.model_lock:
            if self.phi3_model is not None and self.phi3_tokenizer is not None:
                self.model_registry.set_availability("phi3", True, "loaded")
                return "Phi-3 is already loaded."

            self.model_registry.set_availability("phi3", False, "loading")
            try:
                os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
                os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA not available - check your PyTorch installation")

                logger.info(f"Loading Phi-3 on {torch.cuda.get_device_name(0)}")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                offload_folder = runtime_settings.runtime_state_dir / "model_offload"
                offload_folder.mkdir(parents=True, exist_ok=True)
                phi3_ref, phi3_is_local = self._resolve_model_reference(
                    self.phi3_model_name,
                    "Phi-3",
                    runtime_settings.phi3_revision,
                )
                phi3_load_options = self._transformers_load_options(
                    runtime_settings.phi3_revision,
                    phi3_is_local,
                )

                tokenizer = AutoTokenizer.from_pretrained(
                    phi3_ref,
                    trust_remote_code=True,
                    **phi3_load_options,
                )
                tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLM.from_pretrained(
                    phi3_ref,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map="auto",
                    max_memory={0: "7GB", "cpu": "12GB"},
                    dtype=torch.float16,
                    offload_folder=str(offload_folder),
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",
                    **phi3_load_options,
                ).eval()

                self.phi3_tokenizer = tokenizer
                self.phi3_model = model
                self.model_registry.set_availability(
                    "phi3",
                    True,
                    "loaded",
                    metadata={
                        "device_map": getattr(model, "hf_device_map", {}),
                        "model_reference": str(phi3_ref),
                        "cuda_memory_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
                    },
                )
                return f"Phi-3 loaded successfully from {phi3_ref}."
            except Exception as exc:
                self.phi3_model = None
                self.phi3_tokenizer = None
                self.model_registry.set_availability("phi3", False, "load_failed", metadata={"last_error": str(exc)})
                logger.exception("Phi-3 load failed")
                return f"Phi-3 failed to load: {exc}"

    def _unload_phi3_model(self):
        with self.model_lock:
            was_loaded = self.phi3_model is not None or self.phi3_tokenizer is not None
            self.phi3_model = None
            self.phi3_tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model_registry.set_availability(
                "phi3",
                False,
                "unloaded",
                metadata={
                    "cuda_memory_gb": round(torch.cuda.memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else 0,
                },
            )
            return "Phi-3 unloaded." if was_loaded else "Phi-3 was already unloaded."

    def _probe_phi3_model(self, prompt):
        if self.phi3_model is None or self.phi3_tokenizer is None:
            raise RuntimeError("Phi-3 is not loaded. Load Phi-3 before running the probe.")
        return self._generate_phi3_response(prompt, remember=False, max_new_tokens=160)

    def _handle_model_command(self, user_input):
        lowered = (user_input or "").strip().lower()
        if lowered in {"load phi", "load phi3", "load phi-3", "preload phi", "preload phi3", "preload phi-3"}:
            return self._load_phi3_model(), "ModelRegistry"
        if lowered in {"unload phi", "unload phi3", "unload phi-3", "release phi", "release phi3", "release phi-3"}:
            return self._unload_phi3_model(), "ModelRegistry"
        if lowered in {"model status", "models status", "list models", "model health", "available models"}:
            return self.model_registry.describe(), "ModelRegistry"
        return None, None

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
                            self.voice_settings['rate'] = -1
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

            language_code = (request.language_code or "en-US").strip()
            audio_bytes = bytes(request.audio_data)
            if audio_bytes.startswith(b"RIFF"):
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
            else:
                audio = AudioSegment.from_raw(
                    io.BytesIO(audio_bytes),
                    sample_width=2,
                    frame_rate=request.sample_rate or 16000,
                    channels=1
                )
        
            # Export to WAV format
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)
        
            # Recognize using Google Web API (most accurate free option)
            r = sr.Recognizer()
            with sr.AudioFile(wav_io) as source:
                audio_data = r.record(source)

                last_error = None
                for google_language, source_name in self._speech_language_attempts(language_code):
                    try:
                        print(f"[DEBUG] Processing speech recognition with {google_language}...")
                    except UnicodeEncodeError:
                        pass

                    try:
                        text = r.recognize_google(
                            audio_data,
                            language=google_language,
                            show_all=False
                        )
                        try:
                            print(f"[DEBUG] {google_language} recognition result: {text}")
                        except UnicodeEncodeError:
                            safe_text = text.encode('unicode_escape').decode('ascii')
                            print(f"[DEBUG] {google_language} recognition result (Unicode): {safe_text}")
                        return ai_service_pb2.QueryResponse(
                            response_text=text,
                            ai_source=source_name
                        )
                    except sr.UnknownValueError as exc:
                        last_error = exc
                        continue
                    except sr.RequestError:
                        raise

                if last_error:
                    raise last_error
                raise RuntimeError("Speech recognition produced no language attempts.")
                
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            import traceback
            error_trace = traceback.format_exc()
            try:
                print(f"[DEBUG] Recognition error: {error_trace}")
            except UnicodeEncodeError:
                pass
            
            return ai_service_pb2.QueryResponse(
                response_text="ਅਸਮਰੱਥ (Unable to process)" if (request.language_code or "").startswith("pa") else "Unable to process",
                ai_source="System"
            )

    def _speech_language_attempts(self, language_code):
        normalized = (language_code or "en-US").strip().lower()
        if normalized.startswith("pa"):
            return [("pa-IN", "PunjabiSpeechRecognition")]
        if normalized in {"mixed-in", "auto", "auto-in", "bilingual", "bilingual-in"}:
            return [
                ("en-IN", "MixedSpeechRecognition"),
                ("pa-IN", "PunjabiSpeechRecognition"),
                ("en-US", "EnglishSpeechRecognition"),
            ]
        if normalized == "en-in":
            return [("en-IN", "EnglishSpeechRecognition")]
        return [("en-US", "EnglishSpeechRecognition")]

    def _speak_punjabi(self, text, generation=None):
        """Robust Punjabi TTS with proper cleanup"""
        try:
            text = normalize_for_tts(self._clean_response_text(text)).strip()
            if not text:
                return
            if generation is not None and self._speech_cancelled(generation):
                return

            audio_path = self._schedule_punjabi_tts_cache(text)
            if not audio_path.exists() or audio_path.stat().st_size == 0:
                self._wait_for_punjabi_tts_cache(audio_path, generation)
            if not audio_path.exists() or audio_path.stat().st_size == 0:
                self._generate_punjabi_tts_audio(text, audio_path)

            if not audio_path.exists() or audio_path.stat().st_size == 0:
                raise RuntimeError("Empty audio file generated")

            if generation is not None and self._speech_cancelled(generation):
                return

            self._play_audio_file_interruptible(audio_path, generation)
        
        except Exception as e:
            logger.error(f"Punjabi TTS failed: {e}")
            fallback = "Punjabi voice playback failed. I can still show the Punjabi reply on screen."
            self._speak_english_sentences(fallback, generation or self.speech_generation)

    def _punjabi_tts_cache_path(self, text):
        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:24]
        return self.punjabi_tts_cache_dir / f"{digest}.mp3"

    def _prime_tts_for_segments(self, segments):
        """Warm the expensive speech paths before language switching reaches them."""
        for text_part, lang in segments:
            if lang == "pa" and text_part.strip():
                self._schedule_punjabi_tts_cache(text_part)

    def _schedule_punjabi_tts_cache(self, text):
        text = normalize_for_tts(self._clean_response_text(text)).strip()
        if not text:
            return self._punjabi_tts_cache_path("")
        audio_path = self._punjabi_tts_cache_path(text)
        if audio_path.exists() and audio_path.stat().st_size > 0:
            return audio_path

        key = str(audio_path)
        with self.tts_prime_lock:
            existing = self.punjabi_tts_jobs.get(key)
            if existing and existing.is_alive():
                return audio_path

            def generate():
                try:
                    self._generate_punjabi_tts_audio(text, audio_path)
                except Exception as exc:
                    logger.debug(f"Punjabi TTS cache generation failed: {exc}")
                finally:
                    with self.tts_prime_lock:
                        self.punjabi_tts_jobs.pop(key, None)

            worker = threading.Thread(target=generate, daemon=True)
            self.punjabi_tts_jobs[key] = worker
            worker.start()
        return audio_path

    def _wait_for_punjabi_tts_cache(self, audio_path, generation=None, timeout=12):
        key = str(audio_path)
        start = time.time()
        while True:
            with self.tts_prime_lock:
                worker = self.punjabi_tts_jobs.get(key)
            if not worker or not worker.is_alive():
                return audio_path.exists() and audio_path.stat().st_size > 0
            if generation is not None and self._speech_cancelled(generation):
                return False
            if timeout and time.time() - start > timeout:
                return False
            worker.join(timeout=0.05)

    def _generate_punjabi_tts_audio(self, text, audio_path):
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = audio_path.with_name(f"{audio_path.stem}.{threading.get_ident()}.tmp.mp3")
        tts = gTTS(text=text, lang='pa', slow=False)
        tts.save(str(tmp_path))
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            tmp_path.replace(audio_path)

    def _play_audio_file_interruptible(self, audio_path, generation=None):
        ffplay_path = which("C:/ffmpeg/bin/ffplay.exe") or which("ffplay.exe") or "ffplay"
        process = subprocess.Popen(
            [
                ffplay_path,
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                str(audio_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.active_audio_process = process
        try:
            while process.poll() is None:
                if generation is not None and self._speech_cancelled(generation):
                    process.terminate()
                    try:
                        process.wait(timeout=0.4)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                time.sleep(0.05)
        finally:
            if self.active_audio_process is process:
                self.active_audio_process = None
            if process.poll() is None:
                try:
                    process.terminate()
                except Exception:
                    pass
            

    def GetSystemStatus(self, request, context):
        """gRPC endpoint for system status and weather"""
        greeting = self._get_time_based_greeting()
        location, coords = self._get_location()
        weather = self._get_weather(*coords) if coords else None
    
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
            if g and g.latlng:
                return g.city or "Current location", g.latlng
            return "Malibu", (34.0259, -118.7798)
        except Exception:
            return "Malibu", (34.0259, -118.7798)  # Default to Tony Stark's home

    def _get_weather(self, lat, lng):
            try:
                if not self.weather_api_key:
                    logger.info("OpenWeather API key is not configured; skipping weather lookup")
                    return None
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
        """Format responses with a calm, varied assistant voice."""
        return format_jarvis_response(self._clean_response_text(text), ai_source)

    def _punjabi_voice_prompt(self, user_input):
        return (
            "Reply in Punjabi using proper Gurmukhi script only. "
            "Do not use Latin transliteration unless quoting a technical term. "
            "Keep the answer concise, natural, and easy to speak aloud in one or two sentences. "
            f"User said: {user_input}"
        )

    def _process_with_punjabi(self, user_input):
        mode = (runtime_settings.punjabi_reply_mode or "local").strip().lower()
        if mode in {"mistral", "mistral_if_available", "cloud"} and self.mistral_client:
            return self._process_with_mistral(self._punjabi_voice_prompt(user_input)), "Mistral"
        if mode in {"phi3", "phi"}:
            return self._process_with_phi3(self._punjabi_voice_prompt(user_input)), "Phi-3"
        return self.punjabi_service.handle(user_input), "PunjabiLocal"

    def _punjabi_voice_response(self, text, ai_source):
        cleaned_text = self._clean_response_text(text).strip()
        cleaned_text = re.sub(
            r"^Reply in Punjabi using proper Gurmukhi script only\..*?User said:\s*",
            "",
            cleaned_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        if not cleaned_text:
            cleaned_text = "ਮਾਫ਼ ਕਰਨਾ ਜੀ, ਮੈਨੂੰ ਪੰਜਾਬੀ ਜਵਾਬ ਬਣਾਉਣ ਵਿੱਚ ਦਿੱਕਤ ਆਈ।"
        if not self._contains_gurmukhi(cleaned_text):
            cleaned_text = f"ਜੀ, {cleaned_text}"
        if not cleaned_text.endswith(('.', '!', '?', '।')):
            cleaned_text += "।"
        return cleaned_text

    def _contains_gurmukhi(self, text):
        return any(0x0A00 <= ord(char) <= 0x0A7F for char in text or "")
    
    def _speak_response(self, text):
        """Proper speech implementation with reliable interruption"""
        if not self.voice_settings:
            self._print_jarvis_response(text)
            return

        # Stop ongoing speech only when there is actually something to stop.
        if self._has_active_speech():
            self.stop_current_speech(wait_timeout=0.2)
        with self.speech_state_lock:
            self.speech_generation += 1
            speech_generation = self.speech_generation
            self.stop_speech_event.clear()

        self._print_jarvis_response(text)
        spoken_text = prepare_spoken_response_text(text)
        if not spoken_text:
            spoken_text = "I've put the details on screen."

        def speak_job():
            try:
                self._notify_speech_status("start", text)
                self.is_speaking = True
            
                # Split into segments by language
                segments = self._split_text_by_language(spoken_text)
                self._prime_tts_for_segments(segments)
                english_engine = None
                if any(lang != 'pa' and text_part.strip() for text_part, lang in segments):
                    english_engine = self._create_english_tts_engine()
            
                for i, (text_part, lang) in enumerate(segments):
                    if self._speech_cancelled(speech_generation):
                        break
                    
                    if not text_part.strip():
                        continue

                    if lang == 'pa':
                        # Punjabi handling (unchanged)
                        if not self._speech_cancelled(speech_generation):
                            self._speak_punjabi(text_part, speech_generation)
                    else:
                        # ENGLISH: Use sentence-based approach with proper interruption
                        self._speak_english_sentences(text_part, speech_generation, engine=english_engine, cleanup=False)
                    
                    if self._speech_cancelled(speech_generation):
                        break
                    
            except Exception as e:
                logger.error(f"Speech failed: {e}")
            finally:
                try:
                    if 'english_engine' in locals() and english_engine is not None:
                        self._cleanup_english_tts_engine(english_engine)
                except Exception as exc:
                    logger.debug(f"English TTS cleanup failed: {exc}")
                with self.speech_state_lock:
                    is_current = speech_generation == self.speech_generation
                    if is_current:
                        self.is_speaking = False
                        self.stop_speech_event.clear()
                if is_current:
                    self._notify_speech_status("end", text)

        # Start speech thread
        self.current_speech_thread = threading.Thread(target=speak_job, daemon=True)
        self.current_speech_thread.start()

    def _print_jarvis_response(self, text):
        try:
            print(f"\nJ.A.R.V.I.S: {text}\n")
        except UnicodeEncodeError:
            safe_text = str(text).encode('unicode_escape').decode('ascii')
            print(f"\nJ.A.R.V.I.S (Unicode): {safe_text}\n")

    def _has_active_speech(self):
        with self.speech_state_lock:
            thread_active = self.current_speech_thread is not None and self.current_speech_thread.is_alive()
            audio_active = self.active_audio_process is not None and self.active_audio_process.poll() is None
            return bool(
                self.is_speaking
                or thread_active
                or self.active_sapi_voice
                or self.active_tts_engine
                or audio_active
            )

    def _speech_cancelled(self, generation):
        return self.stop_speech_event.is_set() or generation != self.speech_generation

    def _create_english_tts_engine(self):
        if self.voice_settings['engine_type'] == 'sapi':
            engine = win32com.client.Dispatch("SAPI.SpVoice")
            voice = self.voice_settings.get('voice')
            if voice is not None:
                try:
                    engine.Voice = voice
                except Exception:
                    pass
            self.active_sapi_voice = engine
            return engine

        engine = pyttsx3.init()
        self.active_tts_engine = engine
        if self.voice_settings.get('voice_id'):
            engine.setProperty('voice', self.voice_settings['voice_id'])
        return engine

    def _cleanup_english_tts_engine(self, engine):
        try:
            if self.voice_settings['engine_type'] == 'pyttsx3':
                engine.stop()
        except Exception:
            pass
        if self.active_sapi_voice is engine:
            self.active_sapi_voice = None
        if self.active_tts_engine is engine:
            self.active_tts_engine = None

    def _speak_english_sentences(self, text, generation, engine=None, cleanup=True):
        """Speak English text with reliable sentence-by-sentence control"""

        # Process for JARVIS-style speech
        processed_text = self._process_english_speech(text)
        sentences = self._split_into_sentences(processed_text)
    
        if not sentences:
            return

        if engine is None:
            engine = self._create_english_tts_engine()
            cleanup = True
    
        try:
            for i, sentence in enumerate(sentences):
                # Check for interruption before each sentence
                if self._speech_cancelled(generation):
                    break
                
                # Remove any emphasis markers before speaking
                clean_sentence = re.sub(r'\*(.*?)\*', r'\1', sentence)
                speech_params = self._sentence_speech_params(clean_sentence, i, len(sentences))

                # For SAPI, we need to handle interruption differently
                if self.voice_settings['engine_type'] == 'sapi':
                    engine.Rate = speech_params.get('sapi_rate', self.voice_settings.get('rate', -1))
                    try:
                        engine.Volume = speech_params.get('sapi_volume', 90)
                    except Exception:
                        pass
                    # SAPI doesn't have good interruption, so we use a timeout approach
                    self._sapi_speak_with_timeout(engine, clean_sentence, generation, timeout=10)
                else:
                    engine.setProperty('rate', speech_params.get('rate', self.voice_settings.get('base_rate', 164)))
                    engine.setProperty('volume', speech_params.get('volume', self.voice_settings.get('base_volume', 0.9)))
                    # pyttsx3 - use non-blocking approach
                    engine.say(clean_sentence)
                    engine.startLoop(False)
                
                    start_time = time.time()
                    while engine.isBusy():
                        if self._speech_cancelled(generation):
                            engine.stop()
                            engine.endLoop()
                            return
                        if time.time() - start_time > 8:  # 8 second timeout per sentence
                            break
                        time.sleep(0.03)
                
                    engine.endLoop()
                
                # Add natural pause between sentences (except after last one)
                if i < len(sentences) - 1 and not self._speech_cancelled(generation):
                    self._interruptible_sleep(speech_params.get('pause', 0.24), generation)

                # Check for interruption after each sentence
                if self._speech_cancelled(generation):
                    break
                
        finally:
            # Cleanup
            if cleanup:
                self._cleanup_english_tts_engine(engine)

    def _sapi_speak_with_timeout(self, engine, text, generation, timeout=10):
        """Speak with SAPI asynchronously so interrupt can purge queued speech."""
        speak_async = 1
        purge_before_speak = 2
        try:
            engine.Speak(text, speak_async | purge_before_speak)
        except Exception as exc:
            logger.debug(f"SAPI speak failed: {exc}")
            return

        start_time = time.time()
        while True:
            if self._speech_cancelled(generation):
                try:
                    engine.Speak("", purge_before_speak)
                    engine.Skip("Sentence", 999)
                except:
                    pass
                break
            try:
                if engine.WaitUntilDone(20):
                    break
            except Exception:
                break
            if time.time() - start_time > timeout:
                try:
                    engine.Speak("", purge_before_speak)
                except:
                    pass
                break

    def _interruptible_sleep(self, duration, generation, quantum=0.03):
        end_at = time.time() + max(0.0, float(duration or 0.0))
        while time.time() < end_at:
            if self._speech_cancelled(generation):
                break
            time.sleep(min(quantum, max(0.0, end_at - time.time())))

    def _sentence_speech_params(self, sentence, index, total):
        params = dict(self._get_dynamic_speech_params(sentence))
        lowered = (sentence or "").lower()

        if index == 0 and total > 1:
            params["rate"] = max(140, params["rate"] - 4)
            params["pause"] = max(params.get("pause", 0.24), 0.28)
        if any(term in lowered for term in ["ber", "snr", "errors", "confidence", "probability", "parameters"]):
            params["rate"] = max(138, params["rate"] - 10)
            params["volume"] = min(1.0, params.get("volume", 0.9) + 0.03)
            params["pause"] = max(params.get("pause", 0.24), 0.32)
        if any(term in lowered for term in ["done", "all set", "we have it", "result is in"]):
            params["rate"] = min(178, params["rate"] + 4)
            params["pause"] = max(params.get("pause", 0.24), 0.24)

        params["sapi_rate"] = max(-4, min(3, round((params["rate"] - 160) / 14)))
        params["sapi_volume"] = max(0, min(100, int(params["volume"] * 100)))
        return params

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
        text = normalize_for_tts(self._clean_response_text(text))
        return process_english_speech(text)

    def _get_dynamic_speech_params(self, text):
        """Calculate speech parameters that actually work with TTS engines"""
        return dynamic_speech_params(text)

    def _split_into_sentences(self, text):
        """Split text into natural speaking chunks"""
        return split_into_speech_chunks(text)

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
    
        # Remove emoji/control noise without stripping Unicode letters or combining marks.
        # Punjabi vowel signs are combining marks, so a narrow ASCII-ish allowlist breaks Gurmukhi.
        text = re.sub(r'[\U0001F300-\U0001FAFF\U00002700-\U000027BF]', '', text)
        text = ''.join(char for char in text if char.isprintable() or char in "\n\t")
    
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

    def stop_current_speech(self, wait_timeout=0.25):
        """Proper speech stopping with multiple fallback methods"""
        try:
            if not self._has_active_speech():
                with self.speech_state_lock:
                    self.stop_speech_event.clear()
                    self.is_speaking = False
                return False

            logger.info("Stopping speech...")
            with self.speech_state_lock:
                self.speech_generation += 1
                self.stop_speech_event.set()
        
            # Method 1: Stop pyttsx3 if active
            if self.voice_settings and self.voice_settings['engine_type'] == 'pyttsx3':
                try:
                    if self.active_tts_engine:
                        self.active_tts_engine.stop()
                except Exception as e:
                    logger.debug(f"Pyttsx3 stop: {e}")

            if self.active_audio_process and self.active_audio_process.poll() is None:
                try:
                    self.active_audio_process.terminate()
                    try:
                        self.active_audio_process.wait(timeout=0.3)
                    except subprocess.TimeoutExpired:
                        self.active_audio_process.kill()
                except Exception as e:
                    logger.debug(f"Audio process stop: {e}")
        
            # Method 2: For SAPI, try to interrupt by speaking empty string
            if self.voice_settings and self.voice_settings['engine_type'] == 'sapi':
                try:
                    purge_before_speak = 2
                    sapi = self.active_sapi_voice or win32com.client.Dispatch("SAPI.SpVoice")
                    sapi.Speak("", purge_before_speak)
                    sapi.Skip("Sentence", 999)
                except Exception as e:
                    logger.debug(f"SAPI stop: {e}")
        
            if wait_timeout and self.current_speech_thread and self.current_speech_thread.is_alive():
                self.current_speech_thread.join(timeout=wait_timeout)
            
            with self.speech_state_lock:
                self.is_speaking = False
                if not self.current_speech_thread or not self.current_speech_thread.is_alive():
                    self.stop_speech_event.clear()
                    self.active_sapi_voice = None
                    self.active_tts_engine = None
                    self.active_audio_process = None
            self._notify_speech_status("end", "interrupted")
        
            logger.info("Speech stopped")
            return True
        
        except Exception as e:
            logger.error(f"Error stopping speech: {e}")
            return False

    def InterruptSpeech(self, request, context):
        """gRPC method to interrupt current speech"""
        try:
            stopped = self.stop_current_speech()
            message = "Speech interrupted" if stopped else "No active speech to interrupt"
            if stopped:
                logger.info("Received speech interrupt request")
            return ai_service_pb2.InterruptResponse(success=True, message=message)
        except Exception as e:
            logger.error(f"Failed to interrupt speech: {e}")
            return ai_service_pb2.InterruptResponse(success=False, message=str(e))

    def _init_db(self):
            """Compatibility wrapper; schema ownership moved to MemoryService."""
            self.memory.init_db()

    def _add_to_memory(self, model_type, role, content):
            """Add conversation to memory with automatic summarization"""
            try:
                self.memory.add_conversation(model_type, role, content)

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
            
                if self.mistral_client:
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
                else:
                    summary = conversation_text[:500]

                self.memory.add_summary(model_type, summary, len(recent_chat))
            
            except Exception as e:
                logger.error(f"Summarization failed: {e}")

    def _cleanup_old_summaries(self, model_type):
            """Remove oldest summaries to maintain database size limit"""
            try:
                self.memory.cleanup_old_summaries(model_type)
                
            except Exception as e:
                logger.error(f"Failed to clean up old summaries: {e}")

    def _get_recent_summaries(self, model_type, limit=2):
            """Retrieve most recent summaries from database"""
            try:
                return self.memory.recent_summaries(model_type, limit)
            except Exception as e:
                logger.error(f"Failed to fetch summaries: {e}")
                return []

    def _get_context(self, model_type, new_query):
            """Build context from both short-term and long-term memory"""
            return self.memory.context_messages(model_type, new_query)

    def ProcessQuery(self, request, context):
        """Process queries with full J.A.R.V.I.S personality"""
        try:
            # Display user input
            # SAFE Unicode printing for user input
            raw_input = request.input_text or ""
            runtime_prefix = "__hivemind_runtime__:"
            silent_runtime_request = raw_input.startswith(runtime_prefix)
            user_input = raw_input[len(runtime_prefix):].strip() if silent_runtime_request else raw_input
            try:
                label = "RUNTIME" if silent_runtime_request else "USER"
                print(f"\n{label}: {user_input}")
            except UnicodeEncodeError:
                # Fallback: print escaped Unicode
                safe_text = user_input.encode('unicode_escape').decode('ascii')
                label = "RUNTIME" if silent_runtime_request else "USER"
                print(f"\n{label} (Unicode): {safe_text}")

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

            model_command_response, model_command_source = self._handle_model_command(user_input)
            if model_command_response:
                if silent_runtime_request:
                    return ai_service_pb2.QueryResponse(
                        response_text=model_command_response,
                        ai_source=model_command_source
                )
                self._add_to_memory("runtime", "user", user_input)
                self._add_to_memory("runtime", "assistant", model_command_response)
                response = (
                    self._punjabi_voice_response(model_command_response, model_command_source)
                    if is_punjabi else
                    self._jarvis_style_response(model_command_response, model_command_source)
                )
                self._speak_response(response)
                return ai_service_pb2.QueryResponse(
                    response_text=response,
                    ai_source=model_command_source
                )

            orchestrated_response, orchestrated_source = self.runtime_orchestrator.try_handle(user_input)
            if orchestrated_response:
                if silent_runtime_request:
                    return ai_service_pb2.QueryResponse(
                        response_text=orchestrated_response,
                        ai_source=orchestrated_source
                )
                self._add_to_memory("runtime", "user", user_input)
                self._add_to_memory("runtime", "assistant", orchestrated_response)
                response = (
                    self._punjabi_voice_response(orchestrated_response, orchestrated_source)
                    if is_punjabi else
                    self._jarvis_style_response(orchestrated_response, orchestrated_source)
                )
                self._speak_response(response)
                return ai_service_pb2.QueryResponse(
                    response_text=response,
                    ai_source=orchestrated_source
                )

            if silent_runtime_request:
                return ai_service_pb2.QueryResponse(
                    response_text="Unsupported runtime command.",
                    ai_source="RuntimeOrchestrator"
                )

            casual_response = None if is_punjabi else casual_jarvis_reply(user_input)
            if casual_response:
                self._add_to_memory("phi3", "user", user_input)
                self._add_to_memory("phi3", "assistant", casual_response)
                response = self._jarvis_style_response(casual_response, "JarvisLocal")
                self._speak_response(response)
                return ai_service_pb2.QueryResponse(
                    response_text=response,
                    ai_source="JarvisLocal"
                )
        
            route_decision = self.model_registry.route_decision(user_input)
            print(
                f"[ROUTER] intent={route_decision.intent} target={route_decision.target} "
                f"confidence={route_decision.confidence:.2f} reason={route_decision.reason}",
                flush=True,
            )

            if route_decision.target == "math_service":
                raw_response = self.math_service.handle(user_input)
                ai_source = "MathService"
            elif route_decision.target == "runtime_orchestrator":
                raw_response = (
                    "I recognized this as a runtime/tool request, but no specific adapter route matched it. "
                    "Try `list tools`, `tool health`, or `run capability tool.capability {\"param\": \"value\"}`."
                )
                ai_source = "RuntimeOrchestrator"
            elif is_punjabi:
                raw_response, ai_source = self._process_with_punjabi(user_input)
            elif route_decision.target == "mistral" and self.mistral_client:
                raw_response = self._process_with_mistral(request.input_text)
                ai_source = "Mistral"
            else:
                if route_decision.target == "mistral":
                    logger.info("Mistral route selected but API key is missing; using Phi-3 fallback.")
                raw_response = self._process_with_phi3(user_input)
                ai_source = "Phi-3"
        
            if is_punjabi:
                response = self._punjabi_voice_response(raw_response, ai_source)
            else:
                response = self._jarvis_style_response(raw_response, ai_source)
        
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
        
            self._speak_response(error_msg)
            return ai_service_pb2.QueryResponse(
                response_text=error_msg,
                ai_source="System"
            )

    def _process_with_wizardmath(self, prompt):
        """Process math queries with WizardMath with reliable answer extraction"""
        if self.wizardmath_model is None or self.wizardmath_tokenizer is None:
            logger.warning("WizardMath is not loaded; falling back to Mistral/runtime response")
            return self._process_with_mistral(prompt)

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
            if not self.mistral_client:
                self._add_to_memory("mistral", "user", prompt)
                return "Mistral is not configured. Set HIVEMIND_MISTRAL_API_KEY to enable cloud model routing, or ask for available local runtime tools."

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
        return self._generate_phi3_response(prompt, remember=True, max_new_tokens=160)

    def _generate_phi3_response(self, prompt, remember=True, max_new_tokens=256):
        try:
            if self.phi3_model is None or self.phi3_tokenizer is None:
                logger.warning("Phi-3 is not loaded; falling back when possible")
                if remember and self.mistral_client:
                    return self._process_with_mistral(prompt)
                return (
                    "Phi-3 is not loaded yet, and Mistral is not configured. "
                    "Ask for runtime tools/model status, load Phi-3 from the Runtime panel, "
                    "or enable Phi-3 preload in .env."
                )

            model_prompt = self._phi3_jarvis_prompt(prompt)
            chat_prompt = f"<|user|>\n{model_prompt}<|end|>\n<|assistant|>\n"
        
            inputs = self.phi3_tokenizer(
                chat_prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=2048
            ).to(self.phi3_model.device)
        
            # Use simpler generation parameters to avoid cache issues
            generation_config = {
                'max_new_tokens': max_new_tokens,
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

            input_token_count = inputs["input_ids"].shape[-1]
            generated_tokens = outputs[0][input_token_count:]
            response = self.phi3_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
            # Extract only the assistant's response
            if "<|assistant|>" in response:
                assistant_response = response.split("<|assistant|>")[-1].strip()
            else:
                assistant_response = response.replace(chat_prompt, "").strip()
            assistant_response = self._remove_prompt_echo(assistant_response, model_prompt)
            assistant_response = self._remove_prompt_echo(assistant_response, prompt)
            assistant_response = clean_model_disclaimers(assistant_response)
        
            # Clean up any end tokens
            assistant_response = assistant_response.replace("<|end|>", "").strip()
        
            if not assistant_response or len(assistant_response) < 2:
                assistant_response = "I've processed your request. How can I assist you further?"
        
            # Store in memory
            if remember:
                self._add_to_memory("phi3", "user", prompt)
                self._add_to_memory("phi3", "assistant", assistant_response)
        
            return assistant_response
        
        except Exception as e:
            logger.error(f"Phi-3 processing failed: {e}")
            return "I apologize, but I'm having trouble processing your request at the moment."

    def _phi3_jarvis_prompt(self, prompt):
        return (
            "You are J.A.R.V.I.S inside a local desktop engineering assistant with working voice output. "
            "Sound calm, natural, concise, and technically capable. "
            "Do not say you are text-based, fictional, unable to make sound, or 'as an AI language model'. "
            "Do not prepend canned phrases. For casual greetings, reply in one short sentence. "
            "For technical requests, answer directly in no more than three concise sentences unless detail is required.\n\n"
            f"User: {prompt}"
        )

    def _remove_prompt_echo(self, response, prompt):
        cleaned = (response or "").strip()
        prompt_text = (prompt or "").strip()
        if not cleaned or not prompt_text:
            return cleaned

        prompt_variants = [
            prompt_text,
            f"User: {prompt_text}",
            f"<|user|>\n{prompt_text}",
        ]
        for variant in prompt_variants:
            if cleaned.lower().startswith(variant.lower()):
                cleaned = cleaned[len(variant):].lstrip(" \n\r\t:-")
                break
        return cleaned.strip()

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
    ai_service = AIService()
    ai_service_pb2_grpc.add_AIServiceServicer_to_server(ai_service, server)
    register_runtime_grpc_services(
        server,
        ai_service.runtime_orchestrator.runtime_service,
        ai_service.runtime_model_service,
        ai_service,
    )
    
    port = find_available_port(runtime_settings.port, runtime_settings.port_scan_limit)
    bind_host = f'[{runtime_settings.host}]' if ":" in runtime_settings.host else runtime_settings.host
    server.add_insecure_port(f'{bind_host}:{port}')
    
    try:

        server.start()
        print(f"Server started successfully on port {port}", flush=True)
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
