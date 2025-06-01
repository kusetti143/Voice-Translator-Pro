"""
Speech-to-Text module supporting multiple STT engines.
"""
import streamlit as st
import numpy as np
import tempfile
import os
from typing import Optional, Dict, List, Tuple
import speech_recognition as sr
import whisper
import torch
import requests
import json

# Optional imports
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    pipeline = None
from config.settings import STT_ENGINES, WHISPER_MODELS, API_KEYS, MODELS_DIR
from src.utils import save_audio_file, Timer, logger, show_progress_bar, clear_progress_bar

class WhisperSTT:
    """OpenAI Whisper Speech-to-Text engine."""
    
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self) -> bool:
        """
        Load Whisper model.
        
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if self.model is None:
                with Timer(f"Loading Whisper model '{self.model_name}'"):
                    show_progress_bar(0.3, f"Loading Whisper {self.model_name} model...")
                    self.model = whisper.load_model(self.model_name, device=self.device)
                    show_progress_bar(1.0, "Model loaded successfully!")
                logger.info(f"Whisper model '{self.model_name}' loaded on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            st.error(f"Error loading Whisper model: {e}")
            return False
    
    def transcribe(self, audio_data: np.ndarray, language: str = None) -> Dict:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_data (np.ndarray): Audio data
            language (str): Optional language hint
            
        Returns:
            Dict: Transcription result with text and confidence
        """
        try:
            if not self.load_model():
                return {"text": "", "confidence": 0.0, "error": "Model loading failed"}
            
            with Timer("Whisper transcription"):
                show_progress_bar(0.1, "Preparing audio for transcription...")
                
                # Whisper expects audio in specific format
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.flatten()
                
                # Ensure audio is float32 and normalized
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                if np.max(np.abs(audio_data)) > 1.0:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                show_progress_bar(0.3, "Running Whisper transcription...")
                
                # Transcribe
                options = {"language": language} if language and language != "auto" else {}
                result = self.model.transcribe(audio_data, **options)
                
                show_progress_bar(1.0, "Transcription completed!")
                
                # Extract confidence from segments if available
                confidence = 0.0
                if "segments" in result and result["segments"]:
                    confidences = [seg.get("avg_logprob", 0) for seg in result["segments"]]
                    # Convert log probabilities to confidence scores
                    confidence = np.mean([np.exp(c) for c in confidences if c is not None])
                
                return {
                    "text": result["text"].strip(),
                    "confidence": confidence,
                    "language": result.get("language", "unknown"),
                    "segments": result.get("segments", [])
                }
                
        except Exception as e:
            logger.error(f"Error in Whisper transcription: {e}")
            return {"text": "", "confidence": 0.0, "error": str(e)}

class GoogleSTT:
    """Google Speech Recognition engine."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
    def transcribe(self, audio_data: np.ndarray, language: str = "en-US") -> Dict:
        """
        Transcribe audio using Google Speech Recognition.
        
        Args:
            audio_data (np.ndarray): Audio data
            language (str): Language code
            
        Returns:
            Dict: Transcription result
        """
        try:
            with Timer("Google STT transcription"):
                show_progress_bar(0.2, "Preparing audio for Google STT...")
                
                # Save audio to temporary file
                temp_file = save_audio_file(audio_data, 16000, "wav")
                
                show_progress_bar(0.5, "Sending to Google Speech API...")
                
                # Load audio file
                with sr.AudioFile(temp_file) as source:
                    audio = self.recognizer.record(source)
                
                # Transcribe
                if language == "auto":
                    text = self.recognizer.recognize_google(audio)
                else:
                    text = self.recognizer.recognize_google(audio, language=language)
                
                show_progress_bar(1.0, "Google STT completed!")
                
                # Clean up
                os.unlink(temp_file)
                
                return {
                    "text": text,
                    "confidence": 0.9,  # Google doesn't provide confidence scores
                    "language": language
                }
                
        except sr.UnknownValueError:
            return {"text": "", "confidence": 0.0, "error": "Could not understand audio"}
        except sr.RequestError as e:
            logger.error(f"Google STT request error: {e}")
            return {"text": "", "confidence": 0.0, "error": f"Service error: {e}"}
        except Exception as e:
            logger.error(f"Error in Google STT: {e}")
            return {"text": "", "confidence": 0.0, "error": str(e)}
        finally:
            # Clean up temp file if it exists
            try:
                if 'temp_file' in locals():
                    os.unlink(temp_file)
            except:
                pass

class HuggingFaceSTT:
    """Hugging Face Speech-to-Text using transformers."""

    def __init__(self, model_name: str = "openai/whisper-base"):
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers not available")
        self.model_name = model_name
        self.pipe = None

    def load_model(self) -> bool:
        """Load Hugging Face STT model."""
        try:
            if not HF_AVAILABLE:
                return False
            if self.pipe is None:
                with Timer(f"Loading HF model '{self.model_name}'"):
                    show_progress_bar(0.3, f"Loading {self.model_name}...")
                    device = 0 if torch.cuda.is_available() else -1
                    self.pipe = pipeline(
                        "automatic-speech-recognition",
                        model=self.model_name,
                        device=device
                    )
                    show_progress_bar(1.0, "HF model loaded!")
            return True
        except Exception as e:
            logger.error(f"Error loading HF STT model: {e}")
            return False
    
    def transcribe(self, audio_data: np.ndarray, language: str = None) -> Dict:
        """Transcribe using Hugging Face model."""
        try:
            if not self.load_model():
                return {"text": "", "confidence": 0.0, "error": "Model loading failed"}
            
            with Timer("HuggingFace STT transcription"):
                show_progress_bar(0.5, "Running HF transcription...")
                
                # Ensure correct format
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                result = self.pipe(audio_data, return_timestamps=True)
                show_progress_bar(1.0, "HF STT completed!")
                
                return {
                    "text": result["text"],
                    "confidence": 0.8,  # Default confidence
                    "language": language or "unknown"
                }
                
        except Exception as e:
            logger.error(f"Error in HF STT: {e}")
            return {"text": "", "confidence": 0.0, "error": str(e)}

class STTManager:
    """Manager for different STT engines."""
    
    def __init__(self):
        self.engines = {
            "whisper": WhisperSTT(),
            "google": GoogleSTT()
        }

        # Try to add HuggingFace STT if available
        try:
            self.engines["huggingface"] = HuggingFaceSTT()
            logger.info("HuggingFace STT engine loaded successfully")
        except Exception as e:
            logger.warning(f"HuggingFace STT not available: {e}")

        self.current_engine = "whisper"
        
    def set_engine(self, engine_name: str) -> bool:
        """
        Set the current STT engine.
        
        Args:
            engine_name (str): Name of the engine
            
        Returns:
            bool: True if engine set successfully
        """
        if engine_name in self.engines:
            self.current_engine = engine_name
            logger.info(f"STT engine set to: {engine_name}")
            return True
        else:
            logger.error(f"Unknown STT engine: {engine_name}")
            return False
    
    def set_whisper_model(self, model_name: str) -> bool:
        """Set Whisper model."""
        if "whisper" in self.engines:
            self.engines["whisper"] = WhisperSTT(model_name)
            logger.info(f"Whisper model set to: {model_name}")
            return True
        return False
    
    def transcribe(self, audio_data: np.ndarray, language: str = "auto") -> Dict:
        """
        Transcribe audio using the current engine.
        
        Args:
            audio_data (np.ndarray): Audio data
            language (str): Language hint
            
        Returns:
            Dict: Transcription result
        """
        try:
            engine = self.engines[self.current_engine]
            
            # Handle language parameter for different engines
            if self.current_engine == "google" and language == "auto":
                language = "en-US"
            elif self.current_engine == "whisper" and language == "auto":
                language = None
            
            result = engine.transcribe(audio_data, language)
            result["engine"] = self.current_engine
            
            return result
            
        except Exception as e:
            logger.error(f"Error in STT transcription: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "error": str(e),
                "engine": self.current_engine
            }
    
    def transcribe_chunks(self, audio_chunks: List[np.ndarray], language: str = "auto") -> List[Dict]:
        """
        Transcribe multiple audio chunks.
        
        Args:
            audio_chunks (List[np.ndarray]): List of audio chunks
            language (str): Language hint
            
        Returns:
            List[Dict]: List of transcription results
        """
        results = []
        total_chunks = len(audio_chunks)
        
        for i, chunk in enumerate(audio_chunks):
            show_progress_bar((i + 1) / total_chunks, f"Transcribing chunk {i + 1}/{total_chunks}")
            result = self.transcribe(chunk, language)
            result["chunk_index"] = i
            results.append(result)
        
        return results
    
    def get_available_engines(self) -> List[str]:
        """Get list of available STT engines."""
        return list(self.engines.keys())

# Global STT manager instance
stt_manager = STTManager()
