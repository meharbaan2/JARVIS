# from huggingface_hub import InferenceClient

# client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.1")

# try:
#     response = client.text_generation(
#         prompt="<s>[INST] Explain quantum computing in one sentence [/INST]",
#         max_new_tokens=100
#     )
#     print("API Response:", response)
# except Exception as e:
#     print("API Failed:", str(e))

#     def _process_with_mistral(self, prompt):
#         try:
#             # Test with simple prompt
#             test_prompt = "<s>[INST] Hi! [/INST]"
#             response = self.hf_client.text_generation(test_prompt, max_new_tokens=10)
#             print("DEBUG - API Test Response:", response)  # Check console
        
#             # Now process real request
#             real_response = self.hf_client.text_generation(
#                 f"<s>[INST] {prompt} [/INST]",
#                 max_new_tokens=150
#             )
#             return real_response.split('[/INST]')[-1].strip()
        
#         except Exception as e:
#             print("DEBUG - Full Error:", str(e))  # Detailed error
#             return "API temporarily unavailable"

# import win32com.client
# speaker = win32com.client.Dispatch("SAPI.SpVoice")
# speaker.Speak("Testing Windows voice output")

# import pyttsx3

# engine = pyttsx3.init()
# voices = engine.getProperty('voices')

# print("Available Voices:")
# for i, voice in enumerate(voices):
#     print(f"\nVoice {i + 1}:")
#     print(f"  - ID: {voice.id}")
#     print(f"  - Name: {voice.name}")
#     print(f"  - Languages: {getattr(voice, 'languages', ['Unknown'])}")
#     print(f"  - Gender: {getattr(voice, 'gender', 'Unknown')}")

# # Test speaking with each voice
# for i, voice in enumerate(voices):
#     engine.setProperty('voice', voice.id)
#     print(f"\nTesting Voice {i + 1}: {voice.name}...")
#     engine.say(f"Hello, I am {voice.name}. This is a test.")
#     engine.runAndWait()

# def get_recent_conversations(self, model_type="mistral", limit=10):
#     """Debug method to fetch recent raw conversations"""
#     cursor = self.db_conn.cursor()
#     cursor.execute('''
#         SELECT role, content, timestamp 
#         FROM conversation_memory
#         WHERE model_type = ?
#         ORDER BY timestamp DESC
#         LIMIT ?
#     ''', (model_type, limit))
#     return cursor.fetchall()

# def get_recent_summaries(self, model_type="mistral", limit=5):
#     """Debug method to fetch recent summaries"""
#     cursor = self.db_conn.cursor()
#     cursor.execute('''
#         SELECT summary, timestamp 
#         FROM memory_summaries
#         WHERE model_type = ?
#         ORDER BY timestamp DESC
#         LIMIT ?
#     ''', (model_type, limit))
#     return cursor.fetchall()

# from gtts import gTTS
# import os

# text = "ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?"
# tts = gTTS(text=text, lang='pa', slow=False)
# tts.save("test.mp3")

# # Verify file was created
# print(f"File exists: {os.path.exists('test.mp3')}")
# print(f"File size: {os.path.getsize('test.mp3')} bytes")


# os.system("start test.mp3")  # Windows
