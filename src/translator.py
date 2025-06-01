"""
Translation module supporting multiple translation services.
"""
import streamlit as st
from typing import Dict, List, Optional
import requests
import json
import uuid
from googletrans import Translator as GoogleTranslator
import torch

# Optional imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    pipeline = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
from config.settings import TRANSLATION_SERVICES, HF_TRANSLATION_MODELS, SUPPORTED_LANGUAGES, API_KEYS
from src.utils import Timer, logger, show_progress_bar, detect_language

class GoogleTranslate:
    """Google Translate service wrapper."""
    
    def __init__(self):
        self.translator = GoogleTranslator()
        
    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> Dict:
        """
        Translate text using Google Translate.
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code
            
        Returns:
            Dict: Translation result
        """
        try:
            if not text.strip():
                return {"text": "", "source_lang": source_lang, "target_lang": target_lang}
            
            with Timer("Google Translate"):
                show_progress_bar(0.3, "Translating with Google Translate...")
                
                result = self.translator.translate(
                    text, 
                    src=source_lang if source_lang != "auto" else None,
                    dest=target_lang
                )
                
                show_progress_bar(1.0, "Translation completed!")
                
                return {
                    "text": result.text,
                    "source_lang": result.src,
                    "target_lang": target_lang,
                    "confidence": 0.9,  # Google doesn't provide confidence
                    "service": "google"
                }
                
        except Exception as e:
            logger.error(f"Google Translate error: {e}")
            return {
                "text": text,  # Return original text on error
                "source_lang": source_lang,
                "target_lang": target_lang,
                "error": str(e),
                "service": "google"
            }

class HuggingFaceTranslate:
    """Hugging Face translation models wrapper."""

    def __init__(self):
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers not available")
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
    def load_model(self, model_name: str) -> bool:
        """
        Load a Hugging Face translation model.

        Args:
            model_name (str): Model name/path

        Returns:
            bool: True if model loaded successfully
        """
        try:
            if not HF_AVAILABLE:
                return False
            if model_name not in self.pipelines:
                with Timer(f"Loading HF model '{model_name}'"):
                    show_progress_bar(0.3, f"Loading {model_name}...")
                    
                    device = 0 if torch.cuda.is_available() else -1
                    
                    if "mbart" in model_name.lower():
                        # For mBART models
                        self.pipelines[model_name] = pipeline(
                            "translation",
                            model=model_name,
                            device=device
                        )
                    else:
                        # For OPUS-MT models
                        self.pipelines[model_name] = pipeline(
                            "translation",
                            model=model_name,
                            device=device
                        )
                    
                    show_progress_bar(1.0, "Model loaded!")
                    
            logger.info(f"HF translation model '{model_name}' loaded")
            return True
            
        except Exception as e:
            logger.error(f"Error loading HF model '{model_name}': {e}")
            return False
    
    def translate(self, text: str, model_name: str, source_lang: str = None, target_lang: str = None) -> Dict:
        """
        Translate text using Hugging Face model.
        
        Args:
            text (str): Text to translate
            model_name (str): Model to use
            source_lang (str): Source language
            target_lang (str): Target language
            
        Returns:
            Dict: Translation result
        """
        try:
            if not text.strip():
                return {"text": "", "source_lang": source_lang, "target_lang": target_lang}
            
            if not self.load_model(model_name):
                return {"text": text, "error": "Model loading failed", "service": "huggingface"}
            
            with Timer("HuggingFace translation"):
                show_progress_bar(0.5, "Translating with Hugging Face...")
                
                pipe = self.pipelines[model_name]
                
                # Handle different model types
                if "mbart" in model_name.lower():
                    # mBART requires language codes
                    if source_lang and target_lang:
                        result = pipe(text, src_lang=source_lang, tgt_lang=target_lang)
                    else:
                        result = pipe(text)
                else:
                    # OPUS-MT models
                    result = pipe(text)
                
                show_progress_bar(1.0, "HF translation completed!")
                
                translated_text = result[0]["translation_text"] if isinstance(result, list) else result["translation_text"]
                
                return {
                    "text": translated_text,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "confidence": 0.8,
                    "service": "huggingface",
                    "model": model_name
                }
                
        except Exception as e:
            logger.error(f"HuggingFace translation error: {e}")
            return {
                "text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "error": str(e),
                "service": "huggingface"
            }

class AzureTranslate:
    """Azure Translator service wrapper."""
    
    def __init__(self):
        self.api_key = API_KEYS.get("AZURE_TRANSLATOR_KEY")
        self.region = API_KEYS.get("AZURE_TRANSLATOR_REGION", "global")
        self.endpoint = "https://api.cognitive.microsofttranslator.com"
        
    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> Dict:
        """Translate using Azure Translator."""
        try:
            if not self.api_key:
                return {"text": text, "error": "Azure API key not configured", "service": "azure"}
            
            if not text.strip():
                return {"text": "", "source_lang": source_lang, "target_lang": target_lang}
            
            with Timer("Azure Translate"):
                show_progress_bar(0.3, "Translating with Azure...")
                
                headers = {
                    'Ocp-Apim-Subscription-Key': self.api_key,
                    'Ocp-Apim-Subscription-Region': self.region,
                    'Content-type': 'application/json',
                    'X-ClientTraceId': str(uuid.uuid4())
                }
                
                params = {'api-version': '3.0'}
                if source_lang != "auto":
                    params['from'] = source_lang
                params['to'] = target_lang
                
                body = [{'text': text}]
                
                response = requests.post(
                    f"{self.endpoint}/translate",
                    params=params,
                    headers=headers,
                    json=body
                )
                
                show_progress_bar(1.0, "Azure translation completed!")
                
                if response.status_code == 200:
                    result = response.json()[0]
                    translation = result['translations'][0]
                    
                    return {
                        "text": translation['text'],
                        "source_lang": result.get('detectedLanguage', {}).get('language', source_lang),
                        "target_lang": target_lang,
                        "confidence": result.get('detectedLanguage', {}).get('score', 0.9),
                        "service": "azure"
                    }
                else:
                    return {"text": text, "error": f"Azure API error: {response.status_code}", "service": "azure"}
                    
        except Exception as e:
            logger.error(f"Azure Translate error: {e}")
            return {"text": text, "error": str(e), "service": "azure"}

class TranslationManager:
    """Manager for different translation services."""
    
    def __init__(self):
        self.services = {
            "google": GoogleTranslate()
        }

        # Try to add HuggingFace translation if available
        try:
            self.services["huggingface"] = HuggingFaceTranslate()
            logger.info("HuggingFace translation engine loaded successfully")
        except Exception as e:
            logger.warning(f"HuggingFace translation not available: {e}")

        # Try to add Azure translation if available
        try:
            self.services["azure"] = AzureTranslate()
            logger.info("Azure translation engine loaded successfully")
        except Exception as e:
            logger.warning(f"Azure translation not available: {e}")

        self.current_service = "google"
        self.current_hf_model = "helsinki-nlp/opus-mt-en-es"
        
    def set_service(self, service_name: str) -> bool:
        """Set the current translation service."""
        if service_name in self.services:
            self.current_service = service_name
            logger.info(f"Translation service set to: {service_name}")
            return True
        else:
            logger.error(f"Unknown translation service: {service_name}")
            return False
    
    def set_hf_model(self, model_name: str) -> bool:
        """Set Hugging Face model."""
        self.current_hf_model = model_name
        logger.info(f"HF translation model set to: {model_name}")
        return True
    
    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> Dict:
        """
        Translate text using the current service.
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language
            target_lang (str): Target language
            
        Returns:
            Dict: Translation result
        """
        try:
            if not text.strip():
                return {"text": "", "source_lang": source_lang, "target_lang": target_lang}
            
            # Auto-detect language if needed
            if source_lang == "auto":
                detected_lang = detect_language(text)
                if detected_lang:
                    source_lang = detected_lang
                    logger.info(f"Detected language: {detected_lang}")
            
            service = self.services[self.current_service]
            
            if self.current_service == "huggingface":
                result = service.translate(text, self.current_hf_model, source_lang, target_lang)
            else:
                result = service.translate(text, source_lang, target_lang)
            
            result["service"] = self.current_service
            return result
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {
                "text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "error": str(e),
                "service": self.current_service
            }
    
    def translate_batch(self, texts: List[str], source_lang: str = "auto", target_lang: str = "en") -> List[Dict]:
        """Translate multiple texts."""
        results = []
        total_texts = len(texts)
        
        for i, text in enumerate(texts):
            show_progress_bar((i + 1) / total_texts, f"Translating text {i + 1}/{total_texts}")
            result = self.translate(text, source_lang, target_lang)
            result["batch_index"] = i
            results.append(result)
        
        return results
    
    def get_available_services(self) -> List[str]:
        """Get list of available translation services."""
        return list(self.services.keys())

# Global translation manager instance
translation_manager = TranslationManager()
