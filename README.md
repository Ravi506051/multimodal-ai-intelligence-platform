# Multimodal AI Intelligence Platform

An extensible, local-first workspace for searching and asking questions across PDFs, documents, images, audio, and video. It extracts text, indexes it with lightweight TF-IDF retrieval, returns source-aware context, and optionally uses OpenAI for transcription, vision analysis, and grounded answer generation.

## Features

- **Documents:** PDF, DOCX, TXT, Markdown, and CSV ingestion.
- **Images:** OCR with Tesseract when installed; optional OpenAI vision analysis.
- **Audio:** WAV/MP3/M4A transcription through the OpenAI API (optional).
- **Video:** key-frame extraction with OpenCV and OCR on extracted frames.
- **Search:** CPU-friendly TF-IDF retrieval; no vector database or GPU required.
- **Q&A:** grounded answers with source references; OpenAI chat is optional, with an extractive fallback.
- **Privacy-minded:** files are processed by the app; cloud API calls happen only when you configure and use them.

## Quick start

Python 3.10–3.12 is recommended.

```bash
git clone https://github.com/Ravi506051/multimodal-ai-intelligence-platform.git
cd multimodal-ai-intelligence-platform
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL shown in your terminal.

## Optional integrations

Create a `.env` file from `.env.example` and add an OpenAI API key to enable audio transcription, image analysis, and LLM answers. Without a key, document extraction, OCR (if Tesseract is installed), indexing, and extractive retrieval still work.

For image OCR, install the Tesseract OCR engine separately and ensure its executable is on PATH. Video support requires FFmpeg only for some media formats; OpenCV handles frame extraction.

## Supported formats

| Type | Formats | Processing |
|---|---|---|
| Documents | PDF, DOCX, TXT, MD, CSV | Text extraction |
| Images | PNG, JPG, JPEG, WEBP | OCR; optional vision description |
| Audio | WAV, MP3, M4A | Optional API transcription |
| Video | MP4, MOV, MKV | Key frames + OCR |

## Project structure

```
app.py                  # Streamlit interface
src/
  ingestion.py          # Multimodal file extraction
  retrieval.py          # TF-IDF indexing and source ranking
  ai_services.py        # Optional OpenAI integrations
  config.py             # Environment configuration
tests/
  test_retrieval.py
.env.example
requirements.txt
```

## Configuration

| Variable | Purpose |
|---|---|
| OPENAI_API_KEY | Enables optional OpenAI features |
| OPENAI_CHAT_MODEL | Chat model (default: gpt-4o-mini) |
| OPENAI_TRANSCRIBE_MODEL | Audio transcription model (default: whisper-1) |

Never commit API keys or private documents.

## Run tests

```bash
pytest -q
```

## Roadmap

- Persistent vector database and multi-user workspaces
- Advanced OCR/document restoration pipeline
- Timestamp-aware video and audio retrieval
- Authentication, deployment configuration, and evaluation dashboard

## License

MIT. See [LICENSE](LICENSE).
