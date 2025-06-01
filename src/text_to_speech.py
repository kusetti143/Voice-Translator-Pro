"""
Text-to-Speech module supporting multiple TTS engines.
"""
import streamlit as st
from typing import Dict, Optional, List
import tempfile
import os
import io
from gtts import gTTS
import pyttsx3
import requests
import json
import base64
from config.settings import TTS_ENGINES, API_KEYS, SUPPORTED_LANGUAGES
from src.utils import Timer, logger, show_progress_bar

class GoogleTTS:
    """Google Text-to-Speech service wrapper."""
    
    def __init__(self):
        self.supported_languages = {
            'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it',
            'pt': 'pt', 'ru': 'ru', 'ja': 'ja', 'ko': 'ko', 'zh': 'zh',
            'ar': 'ar', 'hi': 'hi', 'tr': 'tr', 'nl': 'nl', 'sv': 'sv',
            'da': 'da', 'no': 'no', 'fi': 'fi', 'pl': 'pl', 'cs': 'cs',
            'hu': 'hu', 'ro': 'ro', 'bg': 'bg', 'hr': 'hr', 'sk': 'sk',
            'sl': 'sl', 'et': 'et', 'lv': 'lv', 'lt': 'lt', 'mt': 'mt',
            'ga': 'ga', 'cy': 'cy', 'eu': 'eu', 'ca': 'ca', 'gl': 'gl',
            'is': 'is', 'mk': 'mk', 'sq': 'sq', 'sr': 'sr', 'bs': 'bs',
            'uk': 'uk', 'be': 'be', 'ka': 'ka', 'hy': 'hy', 'az': 'az',
            'he': 'he', 'fa': 'fa', 'ur': 'ur', 'bn': 'bn', 'ta': 'ta',
            'te': 'te', 'ml': 'ml', 'kn': 'kn', 'gu': 'gu', 'pa': 'pa',
            'ne': 'ne', 'si': 'si', 'my': 'my', 'km': 'km', 'lo': 'lo',
            'th': 'th', 'vi': 'vi', 'id': 'id', 'ms': 'ms', 'tl': 'tl',
            'sw': 'sw', 'yo': 'yo', 'zu': 'zu', 'af': 'af'
        }
    
    def synthesize(self, text: str, language: str = "en", slow: bool = False) -> Optional[str]:
        """
        Synthesize speech using Google TTS.
        
        Args:
            text (str): Text to synthesize
            language (str): Language code
            slow (bool): Whether to speak slowly
            
        Returns:
            Optional[str]: Path to generated audio file
        """
        try:
            if not text.strip():
                return None
            
            # Map language code if needed
            lang_code = self.supported_languages.get(language, 'en')
            
            with Timer("Google TTS synthesis"):
                show_progress_bar(0.3, "Generating speech with Google TTS...")
                
                tts = gTTS(text=text, lang=lang_code, slow=slow)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tts.save(tmp_file.name)
                    audio_file = tmp_file.name
                
                show_progress_bar(1.0, "Google TTS completed!")
                
                logger.info(f"Google TTS generated audio: {audio_file}")
                return audio_file
                
        except Exception as e:
            logger.error(f"Google TTS error: {e}")
            st.error(f"Google TTS error: {e}")
            return None

class SystemTTS:
    """System Text-to-Speech using pyttsx3."""
    
    def __init__(self):
        self.engine = None
        self.voices = []
        self._initialize_engine()
    
    def _initialize_engine(self) -> bool:
        """Initialize the TTS engine."""
        try:
            self.engine = pyttsx3.init()
            self.voices = self.engine.getProperty('voices')
            logger.info(f"System TTS initialized with {len(self.voices)} voices")
            return True
        except Exception as e:
            logger.error(f"Error initializing system TTS: {e}")
            return False
    
    def get_available_voices(self) -> List[Dict]:
        """Get list of available system voices."""
        voice_list = []
        for i, voice in enumerate(self.voices):
            voice_list.append({
                'id': i,
                'name': voice.name,
                'language': getattr(voice, 'languages', ['en'])[0] if hasattr(voice, 'languages') else 'en'
            })
        return voice_list
    
    def synthesize(self, text: str, voice_id: int = 0, rate: int = 200, volume: float = 0.9) -> Optional[str]:
        """
        Synthesize speech using system TTS.
        
        Args:
            text (str): Text to synthesize
            voice_id (int): Voice ID to use
            rate (int): Speech rate
            volume (float): Volume level (0.0 to 1.0)
            
        Returns:
            Optional[str]: Path to generated audio file
        """
        try:
            if not text.strip() or not self.engine:
                return None
            
            with Timer("System TTS synthesis"):
                show_progress_bar(0.3, "Generating speech with System TTS...")
                
                # Set voice properties
                if voice_id < len(self.voices):
                    self.engine.setProperty('voice', self.voices[voice_id].id)
                
                self.engine.setProperty('rate', rate)
                self.engine.setProperty('volume', volume)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    audio_file = tmp_file.name
                
                self.engine.save_to_file(text, audio_file)
                self.engine.runAndWait()
                
                show_progress_bar(1.0, "System TTS completed!")
                
                logger.info(f"System TTS generated audio: {audio_file}")
                return audio_file
                
        except Exception as e:
            logger.error(f"System TTS error: {e}")
            st.error(f"System TTS error: {e}")
            return None

class AzureTTS:
    """Azure Text-to-Speech service wrapper."""
    
    def __init__(self):
        self.api_key = API_KEYS.get("AZURE_SPEECH_KEY")
        self.region = API_KEYS.get("AZURE_SPEECH_REGION", "eastus")
        self.endpoint = f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"
        
    def get_voices(self) -> List[Dict]:
        """Get available Azure voices."""
        try:
            if not self.api_key:
                return []
            
            headers = {
                'Ocp-Apim-Subscription-Key': self.api_key
            }
            
            response = requests.get(
                f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/voices/list",
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error getting Azure voices: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting Azure voices: {e}")
            return []
    
    def synthesize(self, text: str, voice_name: str = "en-US-AriaNeural", language: str = "en-US") -> Optional[str]:
        """
        Synthesize speech using Azure TTS.
        
        Args:
            text (str): Text to synthesize
            voice_name (str): Azure voice name
            language (str): Language code
            
        Returns:
            Optional[str]: Path to generated audio file
        """
        try:
            if not self.api_key:
                st.error("Azure Speech API key not configured")
                return None
            
            if not text.strip():
                return None
            
            with Timer("Azure TTS synthesis"):
                show_progress_bar(0.3, "Generating speech with Azure TTS...")
                
                headers = {
                    'Ocp-Apim-Subscription-Key': self.api_key,
                    'Content-Type': 'application/ssml+xml',
                    'X-Microsoft-OutputFormat': 'audio-16khz-128kbitrate-mono-mp3'
                }
                
                # Create SSML
                ssml = f"""
                <speak version='1.0' xml:lang='{language}'>
                    <voice xml:lang='{language}' name='{voice_name}'>
                        {text}
                    </voice>
                </speak>
                """
                
                response = requests.post(self.endpoint, headers=headers, data=ssml)
                
                if response.status_code == 200:
                    # Save audio to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                        tmp_file.write(response.content)
                        audio_file = tmp_file.name
                    
                    show_progress_bar(1.0, "Azure TTS completed!")
                    logger.info(f"Azure TTS generated audio: {audio_file}")
                    return audio_file
                else:
                    logger.error(f"Azure TTS error: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Azure TTS error: {e}")
            st.error(f"Azure TTS error: {e}")
            return None

class TTSManager:
    """Manager for different TTS engines."""
    
    def __init__(self):
        self.engines = {
            "gtts": GoogleTTS(),
            "pyttsx3": SystemTTS(),
            "azure": AzureTTS()
        }
        self.current_engine = "gtts"
        
    def set_engine(self, engine_name: str) -> bool:
        """Set the current TTS engine."""
        if engine_name in self.engines:
            self.current_engine = engine_name
            logger.info(f"TTS engine set to: {engine_name}")
            return True
        else:
            logger.error(f"Unknown TTS engine: {engine_name}")
            return False
    
    def synthesize(self, text: str, language: str = "en", **kwargs) -> Optional[str]:
        """
        Synthesize speech using the current engine.
        
        Args:
            text (str): Text to synthesize
            language (str): Language code
            **kwargs: Engine-specific parameters
            
        Returns:
            Optional[str]: Path to generated audio file
        """
        try:
            if not text.strip():
                return None
            
            engine = self.engines[self.current_engine]
            
            if self.current_engine == "gtts":
                return engine.synthesize(text, language, kwargs.get('slow', False))
            elif self.current_engine == "pyttsx3":
                return engine.synthesize(
                    text,
                    kwargs.get('voice_id', 0),
                    kwargs.get('rate', 200),
                    kwargs.get('volume', 0.9)
                )
            elif self.current_engine == "azure":
                return engine.synthesize(
                    text,
                    kwargs.get('voice_name', 'en-US-AriaNeural'),
                    language
                )
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return None
    
    def synthesize_batch(self, texts: List[str], language: str = "en", **kwargs) -> List[Optional[str]]:
        """Synthesize multiple texts."""
        results = []
        total_texts = len(texts)
        
        for i, text in enumerate(texts):
            show_progress_bar((i + 1) / total_texts, f"Synthesizing audio {i + 1}/{total_texts}")
            result = self.synthesize(text, language, **kwargs)
            results.append(result)
        
        return results
    
    def get_available_engines(self) -> List[str]:
        """Get list of available TTS engines."""
        return list(self.engines.keys())
    
    def get_engine_info(self, engine_name: str) -> Dict:
        """Get information about a specific engine."""
        if engine_name not in self.engines:
            return {}
        
        engine = self.engines[engine_name]
        
        if engine_name == "pyttsx3":
            return {
                "voices": engine.get_available_voices(),
                "supports_rate": True,
                "supports_volume": True
            }
        elif engine_name == "azure":
            return {
                "voices": engine.get_voices(),
                "supports_ssml": True
            }
        elif engine_name == "gtts":
            return {
                "supported_languages": list(engine.supported_languages.keys()),
                "supports_slow": True
            }
        
        return {}

# Global TTS manager instance
tts_manager = TTSManager()
