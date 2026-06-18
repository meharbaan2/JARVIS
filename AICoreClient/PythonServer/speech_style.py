import hashlib
import re


_OLD_MECHANICAL_PREFIXES = [
    "according to my analysis,",
    "my research indicates,",
    "i would suggest,",
    "analysis complete:",
    "my assessment indicates,",
    "based on my reasoning,",
    "mathematically speaking,",
    "calculations complete:",
    "the solution appears to be",
]

_CONTRACTIONS = {
    r"\bi\b": "I",
    r"\bim\b": "I'm",
    r"\bid\b": "I'd",
    r"\bive\b": "I've",
    r"\byoure\b": "you're",
    r"\bits\b": "it's",
    r"\bthats\b": "that's",
    r"\bdont\b": "don't",
    r"\bwont\b": "won't",
    r"\bcant\b": "can't",
    r"\bcouldnt\b": "couldn't",
    r"\bwouldnt\b": "wouldn't",
    r"\bshouldnt\b": "shouldn't",
}

_STYLE_VARIANTS = {
    "complete": [
        "All set.",
        "Done.",
        "That's ready.",
        "Completed.",
        "We have it.",
    ],
    "operational": [
        "Telemetry is back.",
        "I've got the result.",
        "The run is complete.",
        "Here's the readout.",
        "Result is in.",
    ],
    "recovering": [
        "There's a snag.",
        "I hit a constraint.",
        "That did not complete cleanly.",
        "Something needs attention.",
        "I found the problem.",
    ],
    "precise": [
        "Here's the clean result.",
        "The calculation resolves cleanly.",
        "Short version.",
        "The answer is straightforward.",
        "I've checked the numbers.",
    ],
    "considered": [
        "My read is this.",
        "I would frame it this way.",
        "The useful answer is this.",
        "Here is the practical version.",
        "I think this is the important bit.",
    ],
    "neutral": [
        "Here's my read.",
        "Short version.",
        "I would put it this way.",
        "The useful bit is this.",
        "Let's keep it clean.",
    ],
}

_LEADING_ACKS = (
    "all set.",
    "done.",
    "completed.",
    "short version.",
    "here's",
    "i've",
    "i found",
    "there's",
    "the answer",
)

_CONVERSATIONAL_SOURCES = {"phi-3", "phi3", "mistral", "jarvislocal", "punjabilocal"}

_CASUAL_REPLIES = [
    "Online and ready. What are we working on?",
    "Running smoothly. What do you need?",
    "Here and listening. Where shall we point the engines?",
    "Systems are steady. What's next?",
    "Present and operational. Talk to me.",
]


def prepare_spoken_response_text(text):
    spoken = str(text or "")
    spoken = _remove_visual_only_lines(spoken)
    spoken = _remove_inline_artifacts(spoken)
    spoken = _replace_urls_for_speech(spoken)
    spoken = _remove_windows_paths(spoken)
    spoken = re.sub(r"\s+", " ", spoken).strip()
    return spoken


def format_jarvis_response(text, ai_source=None):
    cleaned = _strip_honorifics(str(text or "").strip())
    cleaned = _remove_mechanical_prefix(cleaned)
    cleaned = clean_model_disclaimers(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "I'm here, but I did not receive a usable response."

    cleaned = _sentence_case_lightly(cleaned)
    cleaned = _restore_runtime_names(cleaned)
    if not cleaned.endswith((".", "!", "?")):
        cleaned += "."

    lowered = cleaned.lower()
    if lowered.startswith(_LEADING_ACKS):
        return cleaned
    if _is_conversational_source(ai_source):
        return cleaned

    mood = _classify_response(cleaned, ai_source)
    if mood == "neutral":
        return cleaned
    opener = _variant_for(mood, f"{ai_source}:{cleaned}")
    return f"{opener} {cleaned}"

def casual_jarvis_reply(text):
    lowered = re.sub(r"[^\w\s']", " ", str(text or "").lower())
    lowered = re.sub(r"\s+", " ", lowered).strip()
    if not lowered:
        return None

    greetings = {
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "whats up",
        "what's up",
        "wassup",
        "how are you",
        "how are you doing",
        "you there",
        "jarvis",
        "are you there",
    }
    if lowered in greetings:
        return _variant_for("casual", lowered).replace("Where shall we point the engines?", "What are we working on?")
    if lowered in {"test", "testing", "audio test", "voice test", "test audio", "test 1 2 3", "testing 1 2 3"}:
        return "Audio path is live. I can hear the request and respond cleanly."
    if lowered.startswith(("say something", "speak something")):
        return "Voice systems are online. Clean enough for testing, dramatic enough for morale."
    return None


def clean_model_disclaimers(text):
    cleaned = str(text or "")
    cleaned = re.sub(r"\b(as an ai language model|as an ai|as a language model)\b[:,]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi'?m\s+(?:just\s+)?(?:a\s+)?(?:text-based|digital)\s+(?:ai|assistant)[^.!?]*[.!?]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bi\s+can(?:not|'t)\s+(?:produce|make|output)\s+(?:actual\s+)?sound[^.!?]*[.!?]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:in|through)\s+this\s+chat\s+format[^.!?]*[.!?]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhow can i assist you today\??", "What do you need?", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bfeel free to ask(?: me)?(?: here)?\b", "ask", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def process_english_speech(text):
    processed = _strip_honorifics(str(text or ""))
    for pattern, replacement in _CONTRACTIONS.items():
        processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)

    processed = re.sub(r"\bJ\.?A\.?R\.?V\.?I\.?S\.?\b", "Jarvis", processed, flags=re.IGNORECASE)
    processed = re.sub(r"(\d{4,})", r" \1 ", processed)
    processed = re.sub(r"\s+([,.!?;:])", r"\1", processed)
    processed = re.sub(r"\s+", " ", processed)
    return processed.strip()


def dynamic_speech_params(text):
    text_lower = (text or "").lower()
    mood = _classify_response(text or "", None)
    rate = 164
    volume = 0.9
    pause = 0.24

    if mood == "recovering" or any(word in text_lower for word in ["alert", "warning", "critical", "emergency"]):
        rate = 148
        volume = 0.96
        pause = 0.34
    elif mood == "operational":
        rate = 160
        volume = 0.91
        pause = 0.22
    elif mood == "precise":
        rate = 156
        volume = 0.9
        pause = 0.28
    elif "?" in (text or ""):
        rate = 154
        pause = 0.3

    return {
        "rate": rate,
        "volume": volume,
        "pause": pause,
        "sapi_rate": max(-4, min(3, round((rate - 160) / 14))),
        "sapi_volume": max(0, min(100, int(volume * 100))),
    }


def _strip_honorifics(text):
    text = re.sub(r"^\s*(yes|certainly|of course),?\s+sir[,.\s]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*sir[,.\s]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[,;:\s]+sir([.!?])", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"[,;:\s]+sir\b", "", text, flags=re.IGNORECASE)
    return text.strip()


def _remove_mechanical_prefix(text):
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in _OLD_MECHANICAL_PREFIXES:
        if lowered.startswith(prefix):
            return stripped[len(prefix):].lstrip(" ,:-")
    return stripped


def _sentence_case_lightly(text):
    if not text:
        return text
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.upper() + text[index + 1:]
    return text


def _restore_runtime_names(text):
    replacements = {
        r"\bOfdmsim\b": "OFDMSim",
        r"\bOfdm\b": "OFDM",
        r"\bBer\b": "BER",
        r"\bSnr\b": "SNR",
        r"\bCadconverter\b": "CADConverter",
        r"\bFolderguardian\b": "FolderGuardian",
        r"\bJarvis\b": "Jarvis",
    }
    restored = text
    for pattern, replacement in replacements.items():
        restored = re.sub(pattern, replacement, restored, flags=re.IGNORECASE)
    return restored


def _classify_response(text, ai_source):
    lowered = (text or "").lower()
    source = (ai_source or "").lower()
    if _has_failure_marker(lowered):
        return "recovering"
    if any(marker in lowered for marker in ["success", "complete", "completed", "loaded", "initialized", "all set"]):
        return "complete"
    if any(marker in source for marker in ["runtime", "toolregistry", "bayesian", "memory", "modelregistry"]):
        return "operational"
    if any(marker in source for marker in ["math", "wizard"]) or any(marker in lowered for marker in ["result:", "calculation", "equation"]):
        return "precise"
    if "?" in text:
        return "considered"
    return "neutral"


def _variant_for(mood, seed_text):
    if mood == "casual":
        variants = _CASUAL_REPLIES
    else:
        variants = _STYLE_VARIANTS.get(mood) or _STYLE_VARIANTS["neutral"]
    digest = hashlib.sha256(seed_text.encode("utf-8", errors="ignore")).digest()
    return variants[digest[0] % len(variants)]


def _is_conversational_source(ai_source):
    return (ai_source or "").strip().lower() in _CONVERSATIONAL_SOURCES


def _has_failure_marker(lowered_text):
    patterns = [
        r"\bfailed\b",
        r"\bfailure\b",
        r"\berror\b",
        r"\bunable\b",
        r"\binvalid\b",
        r"\bmissing\b",
        r"\bnot configured\b",
        r"\bskipped\b",
    ]
    return any(re.search(pattern, lowered_text) for pattern in patterns)


def _remove_visual_only_lines(text):
    kept = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^(artifact|file|path|source|log|details?)\s*:", stripped, flags=re.IGNORECASE):
            continue
        kept.append(stripped)
    return " ".join(kept)


def _remove_inline_artifacts(text):
    cleaned = re.sub(r"\bArtifact\s*:\s*[A-Za-z]:\\\S+", "", str(text or ""), flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:file|path|source|log)\s*:\s*[A-Za-z]:\\\S+", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _replace_urls_for_speech(text):
    replaced = re.sub(r"https?://\S+|www\.\S+", "I left the link in the text.", str(text or ""), flags=re.IGNORECASE)
    replaced = re.sub(r"(I left the link in the text\.\s*)+", "I left the link in the text. ", replaced)
    return replaced


def _remove_windows_paths(text):
    without_paths = re.sub(r"[A-Za-z]:\\[^\s]+(?:\\[^\s]+)*", "", str(text or ""))
    without_paths = re.sub(r"\b[A-Za-z0-9_.-]+\\[A-Za-z0-9_.\\-]+", "", without_paths)
    return without_paths
