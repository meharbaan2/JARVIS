using System;
using System.IO;
using System.Threading.Tasks;
using NAudio.Wave;
using Grpc.Net.Client;
using Google.Protobuf;
using AICoreClient;
using System.Text;

namespace AICoreClient.VoiceBridge
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.InputEncoding = System.Text.Encoding.UTF8;

            Console.WriteLine("[VoiceBridge] Starting bridge...");

            using var channel = GrpcChannel.ForAddress("http://localhost:50051");
            var client = new AIService.AIServiceClient(channel);

            while (true)
            {
                // Electron triggers audio capture via stdin or args
                Console.WriteLine("[VoiceBridge] Waiting for capture trigger...");
                string? command = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(command)) continue;
                if (command.Equals("exit", StringComparison.OrdinalIgnoreCase)) break;

                string languageCode = command.Equals("punjabi", StringComparison.OrdinalIgnoreCase) ? "pa-IN" : "en-US";

                Console.WriteLine("[VoiceBridge] [RECORDING_EVENT] start");

                var audioData = CaptureAudio(5); // Capture 5 sec audio

                var audioRequest = new AudioRequest
                {
                    AudioData = ByteString.CopyFrom(audioData),
                    LanguageCode = languageCode,
                    SampleRate = 16000
                };

                try
                {
                    var response = await client.RecognizeSpeechAsync(audioRequest);

                    string recognizedText = response.ResponseText;

                    // Use Base64 to avoid encoding issues
                    string base64Text = Convert.ToBase64String(Encoding.UTF8.GetBytes(recognizedText));
                    Console.WriteLine($"[VoiceBridge] Recognized: {base64Text}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[VoiceBridge] Error: {ex.Message}");
                }

                Console.WriteLine("[VoiceBridge] [RECORDING_EVENT] end");
            }
        }

        // Simple NAudio capture method
        static byte[] CaptureAudio(int seconds = 5)
        {
            using var waveIn = new WaveInEvent
            {
                WaveFormat = new WaveFormat(16000, 16, 1)
            };

            var ms = new MemoryStream();
            using var writer = new WaveFileWriter(ms, waveIn.WaveFormat);

            var stopEvent = new System.Threading.ManualResetEvent(false);
            waveIn.DataAvailable += (s, e) =>
            {
                writer.Write(e.Buffer, 0, e.BytesRecorded);
                if (ms.Length > waveIn.WaveFormat.AverageBytesPerSecond * seconds)
                    stopEvent.Set();
            };

            waveIn.StartRecording();
            stopEvent.WaitOne(TimeSpan.FromSeconds(seconds + 1));
            waveIn.StopRecording();

            writer.Flush();
            return ms.ToArray();
        }
    }
}