"""
Configuration settings for the Voice Translator application.
"""
import os
from typing import Dict, List

# Audio settings
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "channels": 1,
    "chunk_duration": 30,  # seconds
    "format": "wav",
    "max_duration": 300,  # 5 minutes max
}

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish", 
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese (Simplified)",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "hu": "Hungarian",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mt": "Maltese",
    "ga": "Irish",
    "cy": "Welsh",
    "eu": "Basque",
    "ca": "Catalan",
    "gl": "Galician",
    "is": "Icelandic",
    "mk": "Macedonian",
    "sq": "Albanian",
    "sr": "Serbian",
    "bs": "Bosnian",
    "me": "Montenegrin",
    "uk": "Ukrainian",
    "be": "Belarusian",
    "kk": "Kazakh",
    "ky": "Kyrgyz",
    "uz": "Uzbek",
    "tg": "Tajik",
    "mn": "Mongolian",
    "ka": "Georgian",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "he": "Hebrew",
    "fa": "Persian",
    "ur": "Urdu",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ne": "Nepali",
    "si": "Sinhala",
    "my": "Myanmar",
    "km": "Khmer",
    "lo": "Lao",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "tl": "Filipino",
    "haw": "Hawaiian",
    "mg": "Malagasy",
    "sw": "Swahili",
    "yo": "Yoruba",
    "ig": "Igbo",
    "zu": "Zulu",
    "af": "Afrikaans",
    "am": "Amharic",
    "ha": "Hausa",
    "so": "Somali",
    "rw": "Kinyarwanda",
    "ny": "Chichewa",
    "sn": "Shona",
    "st": "Sesotho",
    "tn": "Setswana",
    "ts": "Tsonga",
    "ve": "Venda",
    "xh": "Xhosa",
    "ss": "Swati"
}

# STT Engine options
STT_ENGINES = {
    "whisper": "OpenAI Whisper (Local)",
    "google": "Google Speech Recognition",
    "huggingface": "HuggingFace Models"
}

# Translation services
TRANSLATION_SERVICES = {
    "google": "Google Translate",
    "huggingface": "Hugging Face Models",
    "azure": "Azure Translator",
    "aws": "AWS Translate"
}

# TTS engines
TTS_ENGINES = {
    "gtts": "Google Text-to-Speech",
    "pyttsx3": "System TTS",
    "azure": "Azure Speech Services",
    "aws": "AWS Polly"
}

# Whisper model options
WHISPER_MODELS = {
    "tiny": "Tiny (39 MB)",
    "base": "Base (74 MB)", 
    "small": "Small (244 MB)",
    "medium": "Medium (769 MB)",
    "large": "Large (1550 MB)"
}

# Hugging Face translation models
HF_TRANSLATION_MODELS = {
    "helsinki-nlp/opus-mt-en-es": "English to Spanish",
    "helsinki-nlp/opus-mt-en-fr": "English to French",
    "helsinki-nlp/opus-mt-en-de": "English to German",
    "helsinki-nlp/opus-mt-en-it": "English to Italian",
    "helsinki-nlp/opus-mt-en-pt": "English to Portuguese",
    "helsinki-nlp/opus-mt-en-ru": "English to Russian",
    "helsinki-nlp/opus-mt-en-ja": "English to Japanese",
    "helsinki-nlp/opus-mt-en-ko": "English to Korean",
    "helsinki-nlp/opus-mt-en-zh": "English to Chinese",
    "helsinki-nlp/opus-mt-en-ar": "English to Arabic",
    "facebook/mbart-large-50-many-to-many-mmt": "Multilingual (50 languages)"
}

# Default settings
DEFAULT_SETTINGS = {
    "stt_engine": "whisper",
    "whisper_model": "base",
    "translation_service": "google", 
    "tts_engine": "gtts",
    "source_language": "auto",
    "target_language": "en",
    "chunk_audio": True,
    "show_confidence": True,
    "auto_translate": True,
    "auto_speak": False
}

# API Keys (to be set via environment variables)
API_KEYS = {
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "AZURE_SPEECH_KEY": os.getenv("AZURE_SPEECH_KEY"),
    "AZURE_SPEECH_REGION": os.getenv("AZURE_SPEECH_REGION"),
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "AWS_REGION": os.getenv("AWS_REGION", "us-east-1"),
    "HUGGINGFACE_API_KEY": os.getenv("HUGGINGFACE_API_KEY")
}

# File paths
TEMP_DIR = "temp"
MODELS_DIR = "models"
DOWNLOADS_DIR = "downloads"

# Create directories if they don't exist
for directory in [TEMP_DIR, MODELS_DIR, DOWNLOADS_DIR]:
    os.makedirs(directory, exist_ok=True)

def get_available_engines():
    """Get dynamically available engines based on what's actually loaded."""
    try:
        # Import here to avoid circular imports
        from src.speech_to_text import stt_manager
        from src.translator import translation_manager
        from src.text_to_speech import tts_manager

        available_stt = {k: STT_ENGINES.get(k, k.title()) for k in stt_manager.get_available_engines()}
        available_translation = {k: TRANSLATION_SERVICES.get(k, k.title()) for k in translation_manager.get_available_services()}
        available_tts = {k: TTS_ENGINES.get(k, k.title()) for k in tts_manager.get_available_engines()}

        return available_stt, available_translation, available_tts
    except ImportError:
        # Fallback to static definitions if managers not available
        return STT_ENGINES, TRANSLATION_SERVICES, TTS_ENGINES
