"""
Real-Time Voice Translator - Streamlit Application
A comprehensive voice translation app with multiple STT, translation, and TTS engines.
"""
import streamlit as st
import numpy as np
import pandas as pd
import time
import os
from typing import Dict, List, Optional
import plotly.graph_objects as go
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="AI Voice Translator Pro",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom modules
try:
    from config.settings import (
        SUPPORTED_LANGUAGES, STT_ENGINES, TRANSLATION_SERVICES,
        TTS_ENGINES, WHISPER_MODELS, HF_TRANSLATION_MODELS, DEFAULT_SETTINGS
    )
    from src.audio_processor import audio_processor
    from src.speech_to_text import stt_manager
    from src.translator import translation_manager
    from src.text_to_speech import tts_manager
    from src.utils import (
        format_duration, cleanup_temp_files,
        create_download_link, clear_progress_bar
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please run 'python setup.py' to install dependencies and set up the project.")
    st.stop()

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        'translation_history': [],
        'temp_files': [],
        'current_audio': None,
        'current_transcription': '',
        'current_translation': '',
        'processing': False,
        'settings': DEFAULT_SETTINGS.copy()
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_sidebar():
    """Render the sidebar with settings and controls."""
    st.sidebar.title("🎛️ Settings")

    # Get available engines dynamically
    try:
        from config.settings import get_available_engines
        available_stt, available_translation, available_tts = get_available_engines()
    except:
        # Fallback to static definitions
        available_stt, available_translation, available_tts = STT_ENGINES, TRANSLATION_SERVICES, TTS_ENGINES

    # STT Settings
    st.sidebar.subheader("🎤 Speech-to-Text")

    # Ensure current engine is available, fallback to first available if not
    current_stt = st.session_state.settings['stt_engine']
    if current_stt not in available_stt:
        current_stt = list(available_stt.keys())[0]
        st.session_state.settings['stt_engine'] = current_stt

    stt_engine = st.sidebar.selectbox(
        "STT Engine",
        options=list(available_stt.keys()),
        index=list(available_stt.keys()).index(current_stt),
        format_func=lambda x: available_stt[x],
        key="stt_engine_select"
    )
    st.session_state.settings['stt_engine'] = stt_engine
    
    if stt_engine == "whisper":
        whisper_model = st.sidebar.selectbox(
            "Whisper Model",
            options=list(WHISPER_MODELS.keys()),
            index=list(WHISPER_MODELS.keys()).index(st.session_state.settings['whisper_model']),
            format_func=lambda x: WHISPER_MODELS[x],
            key="whisper_model_select"
        )
        st.session_state.settings['whisper_model'] = whisper_model
        stt_manager.set_whisper_model(whisper_model)
    
    stt_manager.set_engine(stt_engine)
    
    # Translation Settings
    st.sidebar.subheader("🌐 Translation")

    # Ensure current translation service is available
    current_translation = st.session_state.settings['translation_service']
    if current_translation not in available_translation:
        current_translation = list(available_translation.keys())[0]
        st.session_state.settings['translation_service'] = current_translation

    translation_service = st.sidebar.selectbox(
        "Translation Service",
        options=list(available_translation.keys()),
        index=list(available_translation.keys()).index(current_translation),
        format_func=lambda x: available_translation[x],
        key="translation_service_select"
    )
    st.session_state.settings['translation_service'] = translation_service
    translation_manager.set_service(translation_service)

    if translation_service == "huggingface" and "huggingface" in available_translation:
        hf_model = st.sidebar.selectbox(
            "HuggingFace Model",
            options=list(HF_TRANSLATION_MODELS.keys()),
            format_func=lambda x: HF_TRANSLATION_MODELS[x],
            key="hf_model_select"
        )
        translation_manager.set_hf_model(hf_model)
    
    # Language Settings
    col1, col2 = st.sidebar.columns(2)
    with col1:
        source_lang = st.selectbox(
            "Source Language",
            options=["auto"] + list(SUPPORTED_LANGUAGES.keys()),
            index=0 if st.session_state.settings['source_language'] == "auto" else 
                  list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.settings['source_language']) + 1,
            format_func=lambda x: "Auto-detect" if x == "auto" else SUPPORTED_LANGUAGES.get(x, x),
            key="source_lang_select"
        )
        st.session_state.settings['source_language'] = source_lang
    
    with col2:
        target_lang = st.selectbox(
            "Target Language",
            options=list(SUPPORTED_LANGUAGES.keys()),
            index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.settings['target_language']),
            format_func=lambda x: SUPPORTED_LANGUAGES[x],
            key="target_lang_select"
        )
        st.session_state.settings['target_language'] = target_lang
    
    # TTS Settings
    st.sidebar.subheader("🔊 Text-to-Speech")

    # Ensure current TTS engine is available
    current_tts = st.session_state.settings['tts_engine']
    if current_tts not in available_tts:
        current_tts = list(available_tts.keys())[0]
        st.session_state.settings['tts_engine'] = current_tts

    tts_engine = st.sidebar.selectbox(
        "TTS Engine",
        options=list(available_tts.keys()),
        index=list(available_tts.keys()).index(current_tts),
        format_func=lambda x: available_tts[x],
        key="tts_engine_select"
    )
    st.session_state.settings['tts_engine'] = tts_engine
    tts_manager.set_engine(tts_engine)
    
    # Processing Options
    st.sidebar.subheader("⚙️ Processing Options")
    st.session_state.settings['chunk_audio'] = st.sidebar.checkbox(
        "Chunk long audio", 
        value=st.session_state.settings['chunk_audio'],
        help="Split audio longer than 30 seconds into chunks"
    )
    
    st.session_state.settings['show_confidence'] = st.sidebar.checkbox(
        "Show confidence scores", 
        value=st.session_state.settings['show_confidence']
    )
    
    st.session_state.settings['auto_translate'] = st.sidebar.checkbox(
        "Auto-translate", 
        value=st.session_state.settings['auto_translate'],
        help="Automatically translate after transcription"
    )
    
    st.session_state.settings['auto_speak'] = st.sidebar.checkbox(
        "Auto-speak translation", 
        value=st.session_state.settings['auto_speak'],
        help="Automatically generate speech for translations"
    )
    
    # Clear History
    st.sidebar.subheader("🗑️ Cleanup")
    if st.sidebar.button("Clear History", type="secondary"):
        st.session_state.translation_history = []
        cleanup_temp_files(st.session_state.temp_files)
        st.session_state.temp_files = []
        st.success("History cleared!")

def render_audio_input():
    """Render audio input section."""
    st.header("📁 Audio File Upload")

    # Add helpful recording information
    with st.expander("🎤 Need to record audio first? Click here for recording options"):
        st.markdown("**� Quick Recording Options:**")

        # Create tabs for different recording methods
        rec_tab1, rec_tab2, rec_tab3 = st.tabs(["🌐 Online Recorders", "📱 Mobile Apps", "💻 Desktop Apps"])

        with rec_tab1:
            st.markdown("**🎤 Free Online Voice Recorders:**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("- 🔗 [Online Voice Recorder](https://online-voice-recorder.com/) - No signup required")
                st.markdown("- 🔗 [Rev Voice Recorder](https://www.rev.com/onlinevoicerecorder) - Professional quality")
                st.markdown("- 🔗 [Vocaroo](https://vocaroo.com/) - Simple and fast")
            with col2:
                st.markdown("- 🔗 [RecordMP3](https://recordmp3online.com/) - Direct MP3 recording")
                st.markdown("- 🔗 [SpeechNotes](https://speechnotes.co/dictate/) - With transcription")
                st.markdown("- 🔗 [AudioTrimmer](https://audiotrimmer.com/online-voice-recorder/) - With editing")

        with rec_tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📱 iOS (iPhone/iPad):**")
                st.markdown("- 🎤 Voice Memos (built-in)")
                st.markdown("- 🎤 GarageBand")
                st.markdown("- 🎤 Just Press Record")
                st.markdown("- 🎤 AudioShare")
            with col2:
                st.markdown("**🤖 Android:**")
                st.markdown("- 🎤 Voice Recorder (built-in)")
                st.markdown("- 🎤 Easy Voice Recorder")
                st.markdown("- 🎤 Smart Recorder")
                st.markdown("- 🎤 RecForge II")

        with rec_tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🪟 Windows:**")
                st.markdown("- 🎤 Voice Recorder (built-in)")
                st.markdown("- 🎤 Audacity (free)")
                st.markdown("- 🎤 Windows Sound Recorder")
            with col2:
                st.markdown("**🍎 macOS:**")
                st.markdown("- 🎤 QuickTime Player")
                st.markdown("- 🎤 GarageBand")
                st.markdown("- 🎤 Audacity (free)")

        st.info("💡 **Workflow**: Record audio with any of these tools → Download/save the file → Upload below for translation!")

    # Main upload section
    st.subheader("📁 Upload Your Audio File")

    # Add helpful information
    st.info("🎯 **Ready to translate?** Upload your audio file below and let AI do the magic!")

    # File size and format information
    with st.expander("📋 Supported Formats & Guidelines"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**✅ Supported Formats:**")
            st.markdown("- 🎵 WAV (best quality)")
            st.markdown("- 🎵 MP3 (most common)")
            st.markdown("- 🎵 FLAC (lossless)")
            st.markdown("- 🎵 OGG (open source)")
            st.markdown("- 🎵 M4A (Apple format)")
            st.markdown("- 🎵 AAC (Advanced Audio)")
            st.markdown("- 🎵 WMA (Windows Media)")
        with col2:
            st.markdown("**📏 Guidelines:**")
            st.markdown("- 📦 Max file size: 100MB")
            st.markdown("- ⏱️ Max duration: 5 minutes")
            st.markdown("- 🎤 Clear speech recommended")
            st.markdown("- 🔇 Minimal background noise")
            st.markdown("- 📢 Normal speaking volume")
            st.markdown("- 🌍 Any language supported")

    uploaded_file = st.file_uploader(
        "🎤 Choose your audio file",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac', 'wma'],
        help="Drag and drop your audio file here, or click to browse"
    )

    if uploaded_file is not None:
        with st.spinner("Processing uploaded file..."):
            audio_data = audio_processor.process_uploaded_file(uploaded_file)

            if audio_data is not None:
                st.session_state.current_audio = audio_data
                st.success(f"✅ File '{uploaded_file.name}' loaded successfully!")

                # Display file info
                audio_info = audio_processor.get_audio_info(audio_data)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Duration", format_duration(audio_info['duration']))
                with col2:
                    st.metric("Sample Rate", f"{audio_info['sample_rate']} Hz")
                with col3:
                    st.metric("File Size", f"{uploaded_file.size / 1024 / 1024:.1f} MB")
                with col4:
                    st.metric("Peak Level", f"{audio_info['peak_level']:.3f}")

                # Audio visualization
                if len(audio_data) > 0:
                    st.subheader("🎵 Audio Waveform")
                    fig = go.Figure()
                    time_axis = np.linspace(0, audio_info['duration'], len(audio_data))
                    fig.add_trace(go.Scatter(
                        x=time_axis,
                        y=audio_data,
                        mode='lines',
                        name='Audio Waveform',
                        line=dict(color='#1f77b4', width=1)
                    ))
                    fig.update_layout(
                        title="Audio Waveform Visualization",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Amplitude",
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

def process_audio():
    """Process the current audio through the STT pipeline."""
    if st.session_state.current_audio is None:
        st.warning("No audio to process. Please record or upload audio first.")
        return
    
    audio_data = st.session_state.current_audio
    settings = st.session_state.settings
    
    with st.spinner("Processing audio..."):
        try:
            # Apply noise reduction if needed
            processed_audio = audio_processor.apply_noise_reduction(audio_data)
            
            # Chunk audio if it's too long and chunking is enabled
            if settings['chunk_audio']:
                chunks = audio_processor.chunk_long_audio(processed_audio)
                
                if len(chunks) > 1:
                    st.info(f"Audio split into {len(chunks)} chunks for processing")
                    
                    # Process each chunk
                    all_results = stt_manager.transcribe_chunks(chunks, settings['source_language'])
                    
                    # Combine results
                    combined_text = " ".join([result.get('text', '') for result in all_results])
                    combined_confidence = np.mean([result.get('confidence', 0) for result in all_results])
                    
                    result = {
                        'text': combined_text,
                        'confidence': combined_confidence,
                        'chunks': all_results,
                        'engine': settings['stt_engine']
                    }
                else:
                    result = stt_manager.transcribe(processed_audio, settings['source_language'])
            else:
                result = stt_manager.transcribe(processed_audio, settings['source_language'])
            
            clear_progress_bar()
            
            if result.get('error'):
                st.error(f"Transcription error: {result['error']}")
                return
            
            st.session_state.current_transcription = result.get('text', '')
            
            # Display results
            st.success("Transcription completed!")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text_area(
                    "Transcription",
                    value=st.session_state.current_transcription,
                    height=100,
                    key="transcription_display"
                )
            
            with col2:
                if settings['show_confidence']:
                    confidence = result.get('confidence', 0)
                    st.metric("Confidence", f"{confidence:.2%}")
                
                detected_lang = result.get('language', 'unknown')
                if detected_lang != 'unknown':
                    lang_name = SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)
                    st.metric("Detected Language", lang_name)
            
            # Auto-translate if enabled
            if settings['auto_translate'] and st.session_state.current_transcription:
                translate_text()
                
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            clear_progress_bar()

def translate_text():
    """Translate the current transcription."""
    if not st.session_state.current_transcription:
        st.warning("No text to translate. Please transcribe audio first.")
        return
    
    settings = st.session_state.settings
    
    with st.spinner("Translating..."):
        try:
            result = translation_manager.translate(
                st.session_state.current_transcription,
                settings['source_language'],
                settings['target_language']
            )
            
            clear_progress_bar()
            
            if result.get('error'):
                st.error(f"Translation error: {result['error']}")
                return
            
            st.session_state.current_translation = result.get('text', '')
            
            # Display translation
            st.success("Translation completed!")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text_area(
                    "Translation",
                    value=st.session_state.current_translation,
                    height=100,
                    key="translation_display"
                )
            
            with col2:
                service = result.get('service', 'unknown')
                st.metric("Service", service.title())
                
                if 'confidence' in result:
                    st.metric("Confidence", f"{result['confidence']:.2%}")
            
            # Add to history
            history_entry = {
                'timestamp': datetime.now(),
                'source_text': st.session_state.current_transcription,
                'translated_text': st.session_state.current_translation,
                'source_lang': settings['source_language'],
                'target_lang': settings['target_language'],
                'stt_engine': settings['stt_engine'],
                'translation_service': settings['translation_service']
            }
            st.session_state.translation_history.append(history_entry)
            
            # Auto-speak if enabled
            if settings['auto_speak']:
                generate_speech()
                
        except Exception as e:
            st.error(f"Error translating text: {e}")
            clear_progress_bar()

def generate_speech():
    """Generate speech from the current translation."""
    if not st.session_state.current_translation:
        st.warning("No translation to speak. Please translate text first.")
        return
    
    settings = st.session_state.settings
    
    with st.spinner("Generating speech..."):
        try:
            audio_file = tts_manager.synthesize(
                st.session_state.current_translation,
                settings['target_language']
            )
            
            clear_progress_bar()
            
            if audio_file:
                st.success("Speech generated!")
                
                # Play audio
                with open(audio_file, 'rb') as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format='audio/mp3')
                
                # Add to temp files for cleanup
                st.session_state.temp_files.append(audio_file)
                
                # Download link
                download_link = create_download_link(
                    audio_file, 
                    f"translation_audio_{int(time.time())}.mp3",
                    "Download Audio"
                )
                st.markdown(download_link, unsafe_allow_html=True)
            else:
                st.error("Failed to generate speech")
                
        except Exception as e:
            st.error(f"Error generating speech: {e}")
            clear_progress_bar()

def render_processing_controls():
    """Render processing control buttons."""
    st.header("🔄 Processing Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🎤 Transcribe", type="primary", disabled=st.session_state.current_audio is None):
            process_audio()
    
    with col2:
        if st.button("🌐 Translate", disabled=not st.session_state.current_transcription):
            translate_text()
    
    with col3:
        if st.button("🔊 Speak", disabled=not st.session_state.current_translation):
            generate_speech()
    
    with col4:
        if st.button("🔄 Process All", type="secondary", disabled=st.session_state.current_audio is None):
            process_audio()

def render_results():
    """Render current results section."""
    st.header("📝 Current Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Text")
        transcription = st.text_area(
            "Edit transcription if needed:",
            value=st.session_state.current_transcription,
            height=150,
            key="editable_transcription"
        )
        
        if transcription != st.session_state.current_transcription:
            st.session_state.current_transcription = transcription
            if st.button("🔄 Re-translate", key="retranslate_btn"):
                translate_text()
    
    with col2:
        st.subheader("Translation")
        st.text_area(
            "Translation result:",
            value=st.session_state.current_translation,
            height=150,
            disabled=True,
            key="readonly_translation"
        )

def render_history():
    """Render translation history."""
    if not st.session_state.translation_history:
        return
    
    st.header("📚 Translation History")
    
    # Create DataFrame for history
    history_df = pd.DataFrame(st.session_state.translation_history)
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Translations", len(history_df))
    with col2:
        unique_langs = len(set(history_df['target_lang'].tolist()))
        st.metric("Languages Used", unique_langs)
    with col3:
        if len(history_df) > 0:
            avg_length = history_df['source_text'].str.len().mean()
            st.metric("Avg Text Length", f"{avg_length:.0f} chars")
    
    # Display history table
    display_df = history_df.copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['source_lang'] = display_df['source_lang'].map(lambda x: SUPPORTED_LANGUAGES.get(x, x))
    display_df['target_lang'] = display_df['target_lang'].map(lambda x: SUPPORTED_LANGUAGES.get(x, x))
    
    st.dataframe(
        display_df[['timestamp', 'source_text', 'translated_text', 'source_lang', 'target_lang']],
        use_container_width=True,
        hide_index=True
    )

def main():
    """Main application function."""
    # Initialize
    initialize_session_state()
    
    # Header
    st.title("🎤 AI Voice Translator Pro")
    st.markdown("*Professional voice translation powered by advanced AI engines - Upload audio files for instant translation*")
    
    # Sidebar
    render_sidebar()
    
    # Main content
    render_audio_input()
    render_processing_controls()
    render_results()
    render_history()
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit, Whisper, Transformers, and multiple AI services")

if __name__ == "__main__":
    main()
