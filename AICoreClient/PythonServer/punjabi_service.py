import re


GURMUKHI_RE = re.compile(r"[\u0A00-\u0A7F]")


class PunjabiConversationService:
    """Fast local Punjabi fallback for simple conversational turns.

    This is intentionally small and deterministic. Phi-3 is not reliable for
    Punjabi chat, so this keeps basic voice conversation coherent while a real
    Punjabi-capable model remains a configurable future upgrade.
    """

    def handle(self, text):
        normalized = _normalize(text)
        if not normalized:
            return "ਜੀ, ਮੈਂ ਸੁਣ ਰਿਹਾ ਹਾਂ। ਤੁਸੀਂ ਦੱਸੋ, ਮੈਂ ਕੀ ਮਦਦ ਕਰਾਂ?"

        if _has_any(normalized, "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "ਸੱਤ ਸ੍ਰੀ ਅਕਾਲ", "ਹੈਲੋ", "ਹੈਲੋ ਜੀ"):
            return "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ ਜੀ। ਮੈਂ ਤਿਆਰ ਹਾਂ, ਦੱਸੋ ਮੈਂ ਤੁਹਾਡੀ ਕੀ ਮਦਦ ਕਰਾਂ?"

        if _has_any(normalized, "ਕੀ ਹਾਲ", "ਕਿਵੇਂ ਹੋ", "ਠੀਕ ਹੋ", "ਤੂੰ ਠੀਕ", "ਤੁਸੀਂ ਠੀਕ"):
            return "ਮੈਂ ਠੀਕ ਹਾਂ ਜੀ। ਤੁਸੀਂ ਦੱਸੋ, ਅੱਜ ਮੈਂ ਤੁਹਾਡੀ ਕੀ ਮਦਦ ਕਰਾਂ?"

        if _has_any(normalized, "ਧੰਨਵਾਦ", "ਸ਼ੁਕਰੀਆ", "ਸ਼ੁਕਰੀਆ", "ਥੈਂਕ"):
            return "ਕੋਈ ਗੱਲ ਨਹੀਂ ਜੀ। ਹੋਰ ਕੁਝ ਚਾਹੀਦਾ ਹੋਵੇ ਤਾਂ ਦੱਸੋ।"

        if _has_any(normalized, "ਤੂੰ ਕੌਣ", "ਤੁਸੀਂ ਕੌਣ", "ਤੇਰਾ ਨਾਮ", "ਤੁਹਾਡਾ ਨਾਮ"):
            return "ਮੈਂ ਜਾਰਵਿਸ ਹਾਂ, ਤੁਹਾਡਾ ਲੋਕਲ ਇੰਜੀਨੀਅਰਿੰਗ ਅਸਿਸਟੈਂਟ। ਮੈਂ ਗੱਲਬਾਤ, ਮੈਮਰੀ, ਅਤੇ ਟੂਲ ਚਲਾਉਣ ਵਿੱਚ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ।"

        if _has_any(normalized, "ਕੀ ਕਰ ਸਕਦਾ", "ਕੀ ਕਰ ਸਕਦੇ", "ਮਦਦ", "ਕੰਮ"):
            return "ਮੈਂ ਤੁਹਾਡੇ ਪ੍ਰੋਜੈਕਟਾਂ ਦੀ ਮੈਮਰੀ ਖੋਜ ਸਕਦਾ ਹਾਂ, OFDMSim ਜਾਂ CADConverter ਵਰਗੇ ਟੂਲ ਚਲਾ ਸਕਦਾ ਹਾਂ, ਅਤੇ ਨਤੀਜੇ ਸਮਝਾ ਸਕਦਾ ਹਾਂ।"

        if _has_any(normalized, "ਬੰਦ", "ਚੁੱਪ", "ਰੁਕ", "ਰੋਕ"):
            return "ਠੀਕ ਹੈ ਜੀ, ਮੈਂ ਰੁਕ ਜਾਂਦਾ ਹਾਂ।"

        if _looks_like_question(normalized):
            return "ਜੀ, ਮੈਂ ਸਮਝ ਗਿਆ। ਇਸ ਵੇਲੇ ਪੰਜਾਬੀ ਲਈ ਤੇਜ਼ ਲੋਕਲ ਜਵਾਬ ਚਾਲੂ ਹੈ। ਜੇ ਇਹ ਕਿਸੇ ਟੂਲ ਜਾਂ ਪ੍ਰੋਜੈਕਟ ਬਾਰੇ ਹੈ, ਤਾਂ ਨਾਮ ਨਾਲ ਦੱਸੋ।"

        return "ਜੀ, ਮੈਂ ਸਮਝ ਗਿਆ। ਤੁਸੀਂ ਹੋਰ ਥੋੜ੍ਹਾ ਸਾਫ਼ ਦੱਸੋ, ਮੈਂ ਉਸ ਮੁਤਾਬਕ ਮਦਦ ਕਰਾਂਗਾ।"


def _normalize(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _has_any(text, *phrases):
    return any(phrase.lower() in text for phrase in phrases)


def _looks_like_question(text):
    return "?" in text or _has_any(text, "ਕੀ", "ਕਿਉਂ", "ਕਿਵੇਂ", "ਕਦੋਂ", "ਕਿੱਥੇ", "ਕੌਣ", "ਕਿਹੜਾ")


def contains_gurmukhi(text):
    return bool(GURMUKHI_RE.search(text or ""))
