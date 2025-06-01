"""
Utility functions for the Voice Translator application.
"""
import os
import time
import tempfile
import streamlit as st
from typing import Optional, List
import numpy as np
import soundfile as sf
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException as LangDetectError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_language(text: str) -> Optional[str]:
    """
    Detect the language of the given text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        Optional[str]: Detected language code or None if detection fails
    """
    try:
        if not text or len(text.strip()) < 3:
            return None
        return detect(text)
    except LangDetectError:
        logger.warning(f"Could not detect language for text: {text[:50]}...")
        return None

def chunk_audio(audio_data: np.ndarray, sample_rate: int, chunk_duration: int = 30) -> List[np.ndarray]:
    """
    Split audio into chunks of specified duration.
    
    Args:
        audio_data (np.ndarray): Audio data
        sample_rate (int): Sample rate of the audio
        chunk_duration (int): Duration of each chunk in seconds
        
    Returns:
        List[np.ndarray]: List of audio chunks
    """
    chunk_samples = chunk_duration * sample_rate
    chunks = []
    
    for i in range(0, len(audio_data), chunk_samples):
        chunk = audio_data[i:i + chunk_samples]
        if len(chunk) > sample_rate:  # Only include chunks longer than 1 second
            chunks.append(chunk)
    
    return chunks

def save_audio_file(audio_data: np.ndarray, sample_rate: int, format: str = "wav") -> str:
    """
    Save audio data to a temporary file.
    
    Args:
        audio_data (np.ndarray): Audio data
        sample_rate (int): Sample rate
        format (str): Audio format
        
    Returns:
        str: Path to the saved file
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}") as tmp_file:
        sf.write(tmp_file.name, audio_data, sample_rate)
        return tmp_file.name





def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds (float): Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"



def cleanup_temp_files(file_paths: List[str]) -> None:
    """
    Clean up temporary files.
    
    Args:
        file_paths (List[str]): List of file paths to delete
    """
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Could not delete temporary file {file_path}: {e}")

def show_progress_bar(progress: float, text: str = "") -> None:
    """
    Show progress bar in Streamlit.
    
    Args:
        progress (float): Progress value between 0 and 1
        text (str): Progress text
    """
    if 'progress_bar' not in st.session_state:
        st.session_state.progress_bar = st.progress(0)
        st.session_state.progress_text = st.empty()
    
    st.session_state.progress_bar.progress(progress)
    if text:
        st.session_state.progress_text.text(text)

def clear_progress_bar() -> None:
    """Clear the progress bar."""
    if 'progress_bar' in st.session_state:
        st.session_state.progress_bar.empty()
        st.session_state.progress_text.empty()
        del st.session_state.progress_bar
        del st.session_state.progress_text



def create_download_link(file_path: str, filename: str, link_text: str) -> str:
    """
    Create a download link for a file.
    
    Args:
        file_path (str): Path to the file
        filename (str): Name for the downloaded file
        link_text (str): Text for the download link
        
    Returns:
        str: HTML download link
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        
        import base64
        b64 = base64.b64encode(data).decode()
        
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'
    except Exception as e:
        logger.error(f"Error creating download link: {e}")
        return f"Error creating download link: {e}"

class Timer:
    """Simple timer context manager."""
    
    def __init__(self, description: str = ""):
        self.description = description
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Suppress unused parameter warnings
        _ = exc_type, exc_val, exc_tb
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.elapsed = duration
        if self.description:
            logger.info(f"{self.description}: {duration:.2f}s")
    
    @property
    def duration(self) -> float:
        """Get the duration of the timer."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
