using System;
using System.IO;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using NAudio.Wave;
using Grpc.Net.Client;
using Grpc.Core;
using Google.Protobuf;
using AICoreClient;
using Hivemind.Runtime;
using System.Text;

namespace AICoreClient.VoiceBridge
{
    class Program
    {
        private static readonly object CaptureLock = new();
        private static CancellationTokenSource? CurrentCaptureCancellation;

        static async Task Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.InputEncoding = System.Text.Encoding.UTF8;

            Console.WriteLine("[VoiceBridge] Starting bridge...");

            var grpcUrl = args.Length > 0 && !string.IsNullOrWhiteSpace(args[0])
                ? args[0]
                : Environment.GetEnvironmentVariable("HIVEMIND_GRPC_URL") ?? "http://localhost:50051";

            Console.WriteLine($"[VoiceBridge] Connecting to {grpcUrl}");

            using var channel = GrpcChannel.ForAddress(grpcUrl);
            var client = new AIService.AIServiceClient(channel);
            var voiceClient = new VoiceService.VoiceServiceClient(channel);
            var commandQueue = Channel.CreateUnbounded<string>();

            _ = Task.Run(async () =>
            {
                while (true)
                {
                    string? input = Console.ReadLine();
                    if (input is null)
                    {
                        commandQueue.Writer.TryComplete();
                        break;
                    }

                    if (IsStopCommand(input))
                    {
                        CancelActiveCapture();
                        continue;
                    }

                    await commandQueue.Writer.WriteAsync(input);
                }
            });

            await foreach (string command in commandQueue.Reader.ReadAllAsync())
            {
                Console.WriteLine("[VoiceBridge] Waiting for capture trigger...");

                if (string.IsNullOrWhiteSpace(command)) continue;
                if (command.Equals("exit", StringComparison.OrdinalIgnoreCase)) break;

                string languageCode = ResolveLanguageCode(command);

                Console.WriteLine("[VoiceBridge] [RECORDING_EVENT] start");
                using var captureCancellation = new CancellationTokenSource();
                SetActiveCapture(captureCancellation);

                try
                {
                    string recognizedText;
                    if (UseStreamingRecognition())
                    {
                        recognizedText = await CaptureAndRecognizeStreamingAsync(voiceClient, languageCode, captureCancellation.Token);
                    }
                    else
                    {
                        recognizedText = await CaptureAndRecognizeUnaryAsync(client, languageCode, captureCancellation.Token);
                    }

                    // Use Base64 to avoid encoding issues
                    string base64Text = Convert.ToBase64String(Encoding.UTF8.GetBytes(recognizedText));
                    Console.WriteLine($"[VoiceBridge] Recognized: {base64Text}");
                }
                catch (OperationCanceledException)
                {
                    Console.WriteLine("[VoiceBridge] Capture cancelled.");
                    Console.WriteLine("[VoiceBridge] [RECORDING_EVENT] cancelled");
                }
                catch (Exception ex)
                {
                    if (UseStreamingRecognition())
                    {
                        try
                        {
                            Console.WriteLine($"[VoiceBridge] Streaming error, falling back: {ex.Message}");
                            string recognizedText = await CaptureAndRecognizeUnaryAsync(client, languageCode, captureCancellation.Token);
                            string base64Text = Convert.ToBase64String(Encoding.UTF8.GetBytes(recognizedText));
                            Console.WriteLine($"[VoiceBridge] Recognized: {base64Text}");
                        }
                        catch (Exception fallbackEx)
                        {
                            Console.WriteLine($"[VoiceBridge] Error: {fallbackEx.Message}");
                        }
                    }
                    else
                    {
                        Console.WriteLine($"[VoiceBridge] Error: {ex.Message}");
                    }
                }
                finally
                {
                    ClearActiveCapture(captureCancellation);
                }

                Console.WriteLine("[VoiceBridge] [RECORDING_EVENT] end");
            }
        }

        static bool IsStopCommand(string command)
        {
            string normalized = (command ?? string.Empty).Trim().ToLowerInvariant();
            return normalized is "stop" or "cancel" or "stop-voice" or "cancel-voice";
        }

        static void SetActiveCapture(CancellationTokenSource cancellation)
        {
            lock (CaptureLock)
            {
                CurrentCaptureCancellation = cancellation;
            }
        }

        static void ClearActiveCapture(CancellationTokenSource cancellation)
        {
            lock (CaptureLock)
            {
                if (ReferenceEquals(CurrentCaptureCancellation, cancellation))
                {
                    CurrentCaptureCancellation = null;
                }
            }
        }

        static void CancelActiveCapture()
        {
            lock (CaptureLock)
            {
                if (CurrentCaptureCancellation is { IsCancellationRequested: false })
                {
                    Console.WriteLine("[VoiceBridge] Stop requested.");
                    CurrentCaptureCancellation.Cancel();
                }
            }
        }

        static string ResolveLanguageCode(string command)
        {
            string normalized = (command ?? string.Empty).Trim().ToLowerInvariant();
            if (normalized.Contains("punjabi") || normalized.Contains("pa-in") || normalized == "pa")
            {
                return "pa-IN";
            }

            if (normalized.Contains("mixed") || normalized.Contains("bilingual") || normalized.Contains("auto"))
            {
                return "mixed-IN";
            }

            if (normalized.Contains("en-in"))
            {
                return "en-IN";
            }

            return "en-US";
        }

        static bool UseStreamingRecognition()
        {
            string? value = Environment.GetEnvironmentVariable("HIVEMIND_VOICE_STREAMING");
            return string.IsNullOrWhiteSpace(value) || value.Trim().Equals("1") || value.Trim().Equals("true", StringComparison.OrdinalIgnoreCase) || value.Trim().Equals("yes", StringComparison.OrdinalIgnoreCase);
        }

        static async Task<string> CaptureAndRecognizeUnaryAsync(AIService.AIServiceClient client, string languageCode, CancellationToken cancellationToken)
        {
            var audioData = CaptureUtteranceAudio(cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();

            var audioRequest = new AudioRequest
            {
                AudioData = ByteString.CopyFrom(audioData),
                LanguageCode = languageCode,
                SampleRate = 16000
            };

            var response = await client.RecognizeSpeechAsync(audioRequest, cancellationToken: cancellationToken);
            return response.ResponseText;
        }

        static async Task<string> CaptureAndRecognizeStreamingAsync(VoiceService.VoiceServiceClient voiceClient, string languageCode, CancellationToken cancellationToken)
        {
            using var call = voiceClient.StreamRecognizeSpeech();
            var recognizedText = string.Empty;
            var responseTask = Task.Run(async () =>
            {
                await foreach (var speechEvent in call.ResponseStream.ReadAllAsync(cancellationToken))
                {
                    if (speechEvent.Type == "partial")
                    {
                        Console.WriteLine($"[VoiceBridge] Partial: {speechEvent.Text}");
                    }
                    else if (speechEvent.Type == "final")
                    {
                        recognizedText = speechEvent.Text;
                    }
                    else if (speechEvent.Type == "error")
                    {
                        throw new InvalidOperationException(speechEvent.Text);
                    }
                }
            }, cancellationToken);

            await StreamUtteranceAudioAsync(call.RequestStream, languageCode, cancellationToken);
            if (cancellationToken.IsCancellationRequested)
            {
                call.Dispose();
                cancellationToken.ThrowIfCancellationRequested();
            }
            await responseTask;
            if (string.IsNullOrWhiteSpace(recognizedText))
            {
                throw new InvalidOperationException("Streaming recognition completed without final text.");
            }
            return recognizedText;
        }

        static async Task StreamUtteranceAudioAsync(IClientStreamWriter<AudioChunk> requestStream, string languageCode, CancellationToken cancellationToken)
        {
            using var waveIn = new WaveInEvent
            {
                WaveFormat = new WaveFormat(16000, 16, 1),
                BufferMilliseconds = GetIntEnv("HIVEMIND_VOICE_BUFFER_MS", 100)
            };

            using var stopEvent = new ManualResetEventSlim(false);
            var audioQueue = Channel.CreateUnbounded<AudioChunk>();
            int silenceMs = GetIntEnv("HIVEMIND_VOICE_SILENCE_MS", 900);
            int maxSeconds = GetIntEnv("HIVEMIND_VOICE_MAX_SECONDS", 12);
            int minMs = GetIntEnv("HIVEMIND_VOICE_MIN_MS", 700);
            int noSpeechMs = GetIntEnv("HIVEMIND_VOICE_NO_SPEECH_MS", 2200);
            int threshold = GetIntEnv("HIVEMIND_VOICE_SILENCE_THRESHOLD", 650);
            DateTime startedAt = DateTime.UtcNow;
            DateTime lastVoiceAt = startedAt;
            bool speechStarted = false;

            var writerTask = Task.Run(async () =>
            {
                await foreach (var chunk in audioQueue.Reader.ReadAllAsync(cancellationToken))
                {
                    await requestStream.WriteAsync(chunk, cancellationToken);
                }
                await requestStream.WriteAsync(new AudioChunk
                {
                    LanguageCode = languageCode,
                    SampleRate = 16000,
                    EndOfUtterance = true
                }, cancellationToken);
                await requestStream.CompleteAsync();
            }, cancellationToken);

            waveIn.DataAvailable += (s, e) =>
            {
                var chunkBytes = new byte[e.BytesRecorded];
                Buffer.BlockCopy(e.Buffer, 0, chunkBytes, 0, e.BytesRecorded);
                audioQueue.Writer.TryWrite(new AudioChunk
                {
                    AudioData = ByteString.CopyFrom(chunkBytes),
                    LanguageCode = languageCode,
                    SampleRate = 16000,
                    EndOfUtterance = false
                });

                int peak = PeakAmplitude(e.Buffer, e.BytesRecorded);
                DateTime now = DateTime.UtcNow;
                int elapsedMs = (int)(now - startedAt).TotalMilliseconds;
                if (peak >= threshold)
                {
                    speechStarted = true;
                    lastVoiceAt = now;
                }

                int silentForMs = (int)(now - lastVoiceAt).TotalMilliseconds;
                bool hitMax = elapsedMs >= maxSeconds * 1000;
                bool speechEnded = speechStarted && elapsedMs >= minMs && silentForMs >= silenceMs;
                bool noSpeechTimeout = !speechStarted && elapsedMs >= noSpeechMs;
                if (hitMax || speechEnded || noSpeechTimeout || cancellationToken.IsCancellationRequested)
                {
                    stopEvent.Set();
                }
            };

            Console.WriteLine(
                $"[VoiceBridge] gRPC streaming capture: max={maxSeconds}s silence={silenceMs}ms threshold={threshold}"
            );
            waveIn.StartRecording();
            while (!stopEvent.Wait(TimeSpan.FromMilliseconds(50)))
            {
                cancellationToken.ThrowIfCancellationRequested();
            }
            waveIn.StopRecording();
            audioQueue.Writer.TryComplete();
            await writerTask;
        }

        static byte[] CaptureUtteranceAudio(CancellationToken cancellationToken)
        {
            using var waveIn = new WaveInEvent
            {
                WaveFormat = new WaveFormat(16000, 16, 1),
                BufferMilliseconds = GetIntEnv("HIVEMIND_VOICE_BUFFER_MS", 100)
            };

            var ms = new MemoryStream();
            using var stopEvent = new ManualResetEventSlim(false);
            object streamLock = new();
            int silenceMs = GetIntEnv("HIVEMIND_VOICE_SILENCE_MS", 900);
            int maxSeconds = GetIntEnv("HIVEMIND_VOICE_MAX_SECONDS", 12);
            int minMs = GetIntEnv("HIVEMIND_VOICE_MIN_MS", 700);
            int noSpeechMs = GetIntEnv("HIVEMIND_VOICE_NO_SPEECH_MS", 2200);
            int threshold = GetIntEnv("HIVEMIND_VOICE_SILENCE_THRESHOLD", 650);
            DateTime startedAt = DateTime.UtcNow;
            DateTime lastVoiceAt = startedAt;
            bool speechStarted = false;

            waveIn.DataAvailable += (s, e) =>
            {
                lock (streamLock)
                {
                    ms.Write(e.Buffer, 0, e.BytesRecorded);
                }

                int peak = PeakAmplitude(e.Buffer, e.BytesRecorded);
                DateTime now = DateTime.UtcNow;
                int elapsedMs = (int)(now - startedAt).TotalMilliseconds;
                if (peak >= threshold)
                {
                    speechStarted = true;
                    lastVoiceAt = now;
                }

                int silentForMs = (int)(now - lastVoiceAt).TotalMilliseconds;
                bool hitMax = elapsedMs >= maxSeconds * 1000;
                bool speechEnded = speechStarted && elapsedMs >= minMs && silentForMs >= silenceMs;
                bool noSpeechTimeout = !speechStarted && elapsedMs >= noSpeechMs;
                if (hitMax || speechEnded || noSpeechTimeout || cancellationToken.IsCancellationRequested)
                {
                    stopEvent.Set();
                }
            };

            Console.WriteLine(
                $"[VoiceBridge] Streaming capture: max={maxSeconds}s silence={silenceMs}ms threshold={threshold}"
            );
            waveIn.StartRecording();
            while (!stopEvent.Wait(TimeSpan.FromMilliseconds(50)))
            {
                cancellationToken.ThrowIfCancellationRequested();
            }
            waveIn.StopRecording();
            cancellationToken.ThrowIfCancellationRequested();

            lock (streamLock)
            {
                ms.Flush();
            }
            return ms.ToArray();
        }

        static int PeakAmplitude(byte[] buffer, int bytesRecorded)
        {
            int peak = 0;
            for (int index = 0; index + 1 < bytesRecorded; index += 2)
            {
                int sample = BitConverter.ToInt16(buffer, index);
                int amplitude = Math.Abs(sample);
                if (amplitude > peak)
                {
                    peak = amplitude;
                }
            }

            return peak;
        }

        static int GetIntEnv(string name, int defaultValue)
        {
            string? value = Environment.GetEnvironmentVariable(name);
            return int.TryParse(value, out int parsed) && parsed > 0 ? parsed : defaultValue;
        }
    }
}
