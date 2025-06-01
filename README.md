# 🎤 AI Voice Translator Pro

A professional voice translation application built with Streamlit that supports multiple AI engines for Speech-to-Text, Translation, and Text-to-Speech. Upload audio files for instant, high-quality translation with advanced audio processing and noise reduction.

## 🌟 Key Highlights

- **🎯 File Upload Focus** - Reliable audio file processing instead of browser recording
- **🔧 Professional Audio Processing** - Built-in noise reduction and audio enhancement
- **🤖 Multiple AI Engines** - Choose from Whisper, Google, and HuggingFace models
- **🌍 100+ Languages** - Comprehensive language support for global communication
- **📊 Real-time Visualization** - Audio waveform display and processing feedback
- **💾 Download Everything** - Save transcriptions, translations, and generated audio
- **⚙️ Fully Configurable** - Customize all engines and processing parameters
- **🚀 Production Ready** - Clean, optimized codebase with professional features

## ✨ Features

### 📁 Audio File Processing
- **Multiple format support** - WAV, MP3, FLAC, OGG, M4A, AAC, WMA
- **Drag & drop interface** for easy file upload
- **Audio visualization** with waveform display
- **File validation** and format conversion
- **Professional audio processing** pipeline

### 🎤 Speech-to-Text (STT)
- **OpenAI Whisper** (Local) - Multiple model sizes (tiny, base, small, medium, large)
- **Google Speech Recognition** - Cloud-based STT with high accuracy
- **HuggingFace Models** - Transformer-based STT with various model options (optional)
- **Automatic audio chunking** for longer files (>30 seconds)
- **Language detection** and confidence scores
- **Professional transcription** with punctuation and formatting
- **Noise reduction** preprocessing for better accuracy

### 🌐 Translation
- **Google Translate** - 100+ languages supported with high accuracy
- **HuggingFace Models** - OPUS-MT and mBART transformer models (optional)
- **Automatic language detection** from audio content
- **Batch translation support** for multiple texts
- **Editable transcription review** before translation
- **Real-time translation** with instant results

### 🔊 Text-to-Speech (TTS)
- **Google TTS (gTTS)** - Natural-sounding voices with multiple language support
- **System TTS** - Built-in OS text-to-speech (Windows SAPI)
- **Multiple voice options** for each supported language
- **Adjustable speech rate and volume** controls
- **High-quality audio generation** in WAV format
- **Automatic language detection** for optimal voice selection

### 🎛️ User Interface
- **Streamlit Web App** - Modern, responsive interface
- **File Upload Interface** - Drag & drop audio files
- **Audio Visualization** - Waveform display and analysis
- **Progress Indicators** - Visual feedback for all operations
- **Download Options** - Save audio and text results
- **Translation History** - Track all translations
- **Comprehensive Settings** - Customize all engines and parameters
- **Recording Guides** - Built-in help for audio recording

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Audio files to translate (or recording device)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd voice-translator
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up API keys (optional):**
Create a `.env` file in the root directory for enhanced features:
```env
# Google Cloud (for Google TTS and enhanced features)
GOOGLE_API_KEY=your_google_api_key

# HuggingFace (for enhanced transformer models)
HUGGINGFACE_API_KEY=your_hf_api_key
```

**Note:** The application works perfectly without API keys using:
- **Whisper** (local STT)
- **Google Translate** (free tier)
- **System TTS** (built-in voices)

**Optional Dependencies:**
- **HuggingFace Transformers** - Install with `pip install transformers` for additional STT and translation models
- **Azure/AWS Services** - Require API keys for enhanced features

4. **Run the application:**
```bash
streamlit run app.py
```

5. **Open your browser:**
Navigate to `http://localhost:8501`

## 📖 Usage Guide

### Basic Workflow
1. **Configure Settings** - Choose your preferred STT, translation, and TTS engines
2. **Upload Audio File** - Drag & drop or browse for your audio file
3. **Transcribe** - Convert speech to text automatically
4. **Translate** - Translate to your target language
5. **Generate Speech** - Create audio from translation
6. **Review & Download** - Edit results and download files

### Recording Audio (Before Upload)
- Use any recording app on your device (Voice Memos, Voice Recorder, etc.)
- Online recorders: [Online Voice Recorder](https://online-voice-recorder.com/), [Vocaroo](https://vocaroo.com/)
- Desktop apps: Audacity, Windows Voice Recorder, QuickTime Player
- Speak clearly and minimize background noise

### Uploading Files
- **Supported formats:** WAV, MP3, FLAC, OGG, M4A, AAC, WMA
- **File size limit:** 100MB maximum
- **Duration limit:** 5 minutes (automatically chunked if longer)
- **Quality tips:** Clear speech, minimal background noise

### Language Support
- **100+ languages** supported for translation
- **Auto-detection** for source language
- **Popular languages** include English, Spanish, French, German, Italian, Portuguese, Russian, Japanese, Korean, Chinese, Arabic, Hindi, and many more

### Engine Options

#### STT Engines
- **Whisper (Recommended)** - Best accuracy, works offline, multiple model sizes
- **Google Speech Recognition** - Fast, cloud-based, high accuracy
- **HuggingFace Models** - Customizable transformer models (optional: requires `transformers` package)

#### Translation Services
- **Google Translate (Recommended)** - 100+ languages, high quality, free tier
- **HuggingFace Models** - Specialized OPUS-MT and mBART models (optional: requires `transformers` package)

#### TTS Engines
- **Google TTS (Recommended)** - Natural voices, 40+ languages, cloud-based
- **System TTS** - Built-in OS voices, works offline, instant generation

## 🔧 Configuration

### Audio Settings
- **Sample Rate:** 16kHz (optimized for speech)
- **Channels:** Mono
- **Chunk Duration:** 30 seconds (for long audio)
- **Max Duration:** 5 minutes per file

### Processing Options
- **Chunk Audio:** Split long audio automatically
- **Show Confidence:** Display confidence scores
- **Auto-translate:** Translate immediately after transcription
- **Auto-speak:** Generate speech after translation

## 📁 Project Structure

```
AI Voice Translator Pro/
├── 📄 .env.example              # Environment variables template
├── 📄 app.py                    # Main Streamlit application
├── 📋 requirements.txt          # Python dependencies
├── 📖 README.md                 # Project documentation
├── 📁 config/
│   └── ⚙️ settings.py          # Configuration settings
├── 📁 src/
│   ├── 🎤 audio_processor.py   # Audio file processing & noise reduction
│   ├── 🎙️ speech_to_text.py    # STT engines (Whisper, Google, HuggingFace)
│   ├── 🌐 translator.py        # Translation services (Google, HuggingFace)
│   ├── 🔊 text_to_speech.py    # TTS engines (Google TTS, System TTS)
│   └── 🛠️ utils.py             # Utility functions and helpers
├── 📁 temp/                    # Temporary files (auto-managed)
├── 📁 models/                  # AI models cache (auto-populated)
├── 📁 downloads/               # Generated files (user downloads)
├── 📁 __pycache__/             # Python cache files (auto-generated)
└── 📁 myenv/                   # Python virtual environment
```

### 📂 Folder Overview

#### 🏠 **Root Directory**
- **`.env.example`** - Template for environment variables with API keys and configuration options
- **`app.py`** - Main Streamlit application file containing the web interface and core workflow
- **`requirements.txt`** - Python package dependencies required for the project
- **`README.md`** - Comprehensive project documentation with setup and usage instructions

#### ⚙️ **config/** - Configuration Management
**Purpose**: Centralized configuration and settings management
- **`settings.py`** - Contains all application constants, engine configurations, file paths, and default settings
- **Features**: STT/Translation/TTS engine definitions, directory paths, audio processing parameters
- **Benefits**: Easy customization, single point of configuration changes

#### 🧩 **src/** - Source Code Modules
**Purpose**: Core application logic organized by functionality
- **`audio_processor.py`** - Handles audio file upload, validation, noise reduction, and preprocessing
- **`speech_to_text.py`** - Manages STT engines (Whisper, Google Speech, HuggingFace models)
- **`translator.py`** - Handles translation services (Google Translate, HuggingFace models)
- **`text_to_speech.py`** - Manages TTS engines (Google TTS, System TTS)
- **`utils.py`** - Common utility functions, logging, timing, and helper classes

#### 🗂️ **temp/** - Temporary Files (Auto-Managed)
**Purpose**: Temporary storage for audio processing pipeline
- **Usage**: Stores intermediate audio files during STT processing
- **Content**: Uploaded audio files, format conversions, audio chunks
- **Management**: Files automatically created and cleaned up after processing
- **Size**: Varies based on uploaded file sizes (typically MB range)

#### 🤖 **models/** - AI Models Cache (Auto-Populated)
**Purpose**: Local storage for downloaded AI models to improve performance
- **Whisper Models**: OpenAI Whisper models (39MB to 1.5GB depending on size)
- **HuggingFace Models**: Transformer models for STT and translation
- **Benefits**: Faster loading times, offline capability, bandwidth savings
- **Management**: Models downloaded automatically on first use and cached locally

#### 💾 **downloads/** - Generated Files (User Downloads)
**Purpose**: Storage for files generated by the application for user download
- **Content**: Generated TTS audio files, exported transcriptions, translation results
- **Formats**: WAV audio files, text files, JSON exports
- **Access**: Files made available through Streamlit download buttons
- **Cleanup**: Periodic cleanup of old generated files

#### 🐍 **__pycache__/** - Python Cache (Auto-Generated)
**Purpose**: Python bytecode cache for faster module loading
- **Content**: Compiled Python bytecode (.pyc files)
- **Management**: Automatically created and managed by Python interpreter
- **Benefits**: Faster application startup and module imports
- **Note**: Can be safely deleted; will be regenerated automatically

#### 🌐 **myenv/** - Virtual Environment
**Purpose**: Isolated Python environment for project dependencies
- **Content**: Python interpreter, installed packages, project-specific libraries
- **Benefits**: Dependency isolation, version control, clean development environment
- **Management**: Created with `python -m venv myenv`, activated before running the application
- **Size**: Typically 200-500MB depending on installed packages

## 🛠️ Advanced Features

### Batch Processing
- Process multiple audio files
- Batch translation of text lists
- Export results to CSV/JSON

### Custom Models
- Load custom Whisper models
- Use specialized Hugging Face models
- Fine-tuned translation models

### API Integration
- RESTful API endpoints
- Webhook support
- Integration with external services

## 🔍 Troubleshooting

### Common Issues

**Audio file upload errors:**
- Check file format (supported: WAV, MP3, FLAC, OGG, M4A, AAC, WMA)
- Verify file size is under 100MB
- Ensure audio file is not corrupted

**Model loading errors:**
- Check internet connection for model downloads
- Ensure sufficient disk space (2GB+ recommended)
- Verify Python dependencies are installed correctly

**Translation errors:**
- Check internet connection for Google Translate
- Verify language is supported
- Try different translation engine

**Performance issues:**
- Use smaller Whisper models (tiny/base) for faster processing
- Enable audio chunking for files longer than 30 seconds
- Close other applications to free memory
- Consider using Google STT for faster cloud processing

### System Requirements
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 2GB free space for models
- **CPU:** Multi-core processor recommended
- **GPU:** CUDA-compatible GPU for faster processing (optional)

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for the incredible Whisper speech recognition models
- **Google** for translation services and text-to-speech APIs
- **Hugging Face** for transformer models and the amazing model hub
- **Streamlit** for the powerful and intuitive web framework
- **SciPy** for signal processing and audio filtering
- **Librosa** for professional audio processing capabilities

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**Built with ❤️ using Streamlit, OpenAI Whisper, HuggingFace Transformers, Google Services, and advanced audio processing**
