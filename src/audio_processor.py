"""
Audio processing module for handling uploaded audio files.
Focused on file upload and processing - no real-time recording.
"""
import streamlit as st
import numpy as np
from typing import Optional
import tempfile
import os
import librosa
import soundfile as sf
from config.settings import AUDIO_CONFIG
from src.utils import chunk_audio, Timer, logger

# Audio recording functionality removed - using file upload only
HAS_AUDIO_RECORDER = False
RECORDER_TYPE = None

class AudioProcessor:
    """Audio processing utilities for uploaded files."""
    
    def __init__(self):
        self.sample_rate = AUDIO_CONFIG["sample_rate"]
        self.chunk_duration = AUDIO_CONFIG["chunk_duration"]
    
    def process_uploaded_file(self, uploaded_file) -> Optional[np.ndarray]:
        """
        Process uploaded audio file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Optional[np.ndarray]: Processed audio data
        """
        try:
            with Timer() as timer:
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # Load audio file
                audio_data, sr = librosa.load(tmp_file_path, sr=self.sample_rate, mono=True)

                # Clean up temporary file
                os.unlink(tmp_file_path)

            # Log timing after context manager exits
            logger.info(f"Audio file loading: {timer.elapsed:.2f}s")
            logger.info(f"Processed uploaded audio file. Duration: {len(audio_data) / sr:.2f}s")

            return audio_data
                
        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            st.error(f"Error processing audio file: {e}")
            return None
    
    def chunk_long_audio(self, audio_data: np.ndarray) -> list:
        """
        Chunk long audio into smaller segments.
        
        Args:
            audio_data: Audio data to chunk
            
        Returns:
            List[np.ndarray]: List of audio chunks
        """
        try:
            duration = len(audio_data) / self.sample_rate
            
            if duration <= self.chunk_duration:
                return [audio_data]
            
            chunks = chunk_audio(audio_data, self.sample_rate, self.chunk_duration)
            logger.info(f"Audio chunked into {len(chunks)} segments")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking audio: {e}")
            return [audio_data]
    
    def get_audio_info(self, audio_data: np.ndarray) -> dict:
        """
        Get information about audio data.
        
        Args:
            audio_data: Audio data to analyze
            
        Returns:
            dict: Audio information
        """
        try:
            duration = len(audio_data) / self.sample_rate
            peak_level = np.max(np.abs(audio_data))
            rms_level = np.sqrt(np.mean(audio_data**2))
            
            return {
                'duration': duration,
                'sample_rate': self.sample_rate,
                'samples': len(audio_data),
                'peak_level': peak_level,
                'rms_level': rms_level,
                'channels': 1  # Always mono after processing
            }
            
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return {
                'duration': 0,
                'sample_rate': self.sample_rate,
                'samples': 0,
                'peak_level': 0,
                'rms_level': 0,
                'channels': 1
            }
    
    def apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply basic noise reduction to audio data.

        Args:
            audio_data: Audio data to process

        Returns:
            np.ndarray: Processed audio data
        """
        try:
            # Simple noise reduction: normalize and apply basic filtering
            # Normalize audio to [-1, 1] range
            if len(audio_data) == 0:
                return audio_data

            # Remove DC offset
            audio_data = audio_data - np.mean(audio_data)

            # Normalize amplitude
            max_amplitude = np.max(np.abs(audio_data))
            if max_amplitude > 0:
                audio_data = audio_data / max_amplitude * 0.95

            # Simple high-pass filter to remove very low frequencies (< 80 Hz)
            # This helps remove some background noise
            from scipy import signal
            try:
                # Design high-pass filter
                nyquist = self.sample_rate / 2
                low_cutoff = 80 / nyquist
                b, a = signal.butter(1, low_cutoff, btype='high')
                audio_data = signal.filtfilt(b, a, audio_data)
            except ImportError:
                # If scipy is not available, skip filtering
                logger.warning("Scipy not available, skipping noise reduction filtering")

            logger.info("Applied basic noise reduction")
            return audio_data

        except Exception as e:
            logger.warning(f"Error applying noise reduction: {e}")
            # Return original audio if noise reduction fails
            return audio_data

    def validate_audio_file(self, uploaded_file) -> bool:
        """
        Validate uploaded audio file.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            bool: True if file is valid
        """
        try:
            # Check file size (100MB limit)
            max_size = 100 * 1024 * 1024  # 100MB
            if uploaded_file.size > max_size:
                st.error(f"File too large: {uploaded_file.size / 1024 / 1024:.1f}MB. Maximum size: 100MB")
                return False

            # Check file extension
            valid_extensions = ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'wma']
            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension not in valid_extensions:
                st.error(f"Unsupported file format: {file_extension}. Supported: {', '.join(valid_extensions)}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating audio file: {e}")
            st.error(f"Error validating file: {e}")
            return False

# Create global instances
audio_processor = AudioProcessor()

# For backward compatibility - these are no longer used but kept to avoid import errors
class StreamlitAudioRecorder:
    """Deprecated - kept for backward compatibility."""
    
    def __init__(self):
        pass
    
    def record_audio(self, key: str = "audio_recorder") -> Optional[np.ndarray]:
        """
        Deprecated recording method.
        
        Returns:
            None: Always returns None as recording is disabled
        """
        st.warning("⚠️ Real-time recording has been disabled.")
        st.info("📁 **Please use the file upload feature** to upload pre-recorded audio files.")
        st.info("🎤 **Recording Tips**: Use your phone's voice recorder, Windows Voice Recorder, or any audio recording app.")
        return None

# Create instance for backward compatibility
streamlit_recorder = StreamlitAudioRecorder()
