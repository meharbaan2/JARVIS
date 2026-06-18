import re


DECIMAL_PLACEHOLDER = "__HIVEMIND_DECIMAL__"


def normalize_for_tts(text):
    if not text:
        return text
    normalized = str(text)
    normalized = _shorten_long_decimals(normalized)
    normalized = re.sub(r"(\d+)\.(\d+)", rf"\1{DECIMAL_PLACEHOLDER}\2", normalized)
    normalized = normalized.replace("°C", " degrees Celsius")
    normalized = normalized.replace("°F", " degrees Fahrenheit")
    normalized = re.sub(r"(\d+)%", r"\1 percent", normalized)
    normalized = re.sub(r"\bm/s\b", "meters per second", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace(DECIMAL_PLACEHOLDER, " point ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _shorten_long_decimals(text):
    def replace(match):
        whole = match.group(1)
        fraction = match.group(2)
        if len(fraction) <= 3:
            return match.group(0)
        if set(fraction) == {"0"}:
            return whole
        value = float(f"{whole}.{fraction}")
        if abs(value) < 0.0005:
            return "0"
        shortened = f"{value:.3g}" if abs(value) < 1 else f"{value:.2f}"
        if "." in shortened:
            shortened = shortened.rstrip("0").rstrip(".")
        return shortened

    shortened = re.sub(r"\b(\d+)\.0+\b", r"\1", str(text or ""))
    return re.sub(r"\b(\d+)\.(\d{4,})\b", replace, shortened)


def split_into_speech_chunks(text, max_words=18):
    text = normalize_for_tts(text)
    if not text:
        return []

    parts = re.split(r"(?<!\d)([.!?])(?!\d)", text)
    sentences = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts):
            sentence = (parts[i] + parts[i + 1]).strip()
            i += 2
        else:
            sentence = parts[i].strip()
            i += 1
        if sentence:
            sentences.append(sentence)

    chunks = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_words:
            chunks.append(sentence)
            continue

        current = []
        for word in words:
            current.append(word)
            if len(current) >= max_words or word.endswith((",", ";", ":")):
                chunks.append(" ".join(current).strip())
                current = []
        if current:
            chunks.append(" ".join(current).strip())

    return chunks
