"""Extract text from common document, image, audio, and video formats."""
from pathlib import Path
from io import BytesIO
import tempfile
import pandas as pd
from PIL import Image

TEXT_EXTS = {".txt", ".md", ".py", ".json", ".log"}

def _ocr_image(data: bytes) -> str:
    try:
        import pytesseract
        return pytesseract.image_to_string(Image.open(BytesIO(data)))
    except Exception as exc:
        return f"OCR unavailable: {exc}. Install the Tesseract engine and pytesseract."

def extract_file(filename: str, data: bytes) -> tuple[str, str]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages), "PDF text extraction"
    if ext == ".docx":
        from docx import Document
        doc = Document(BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs), "DOCX text extraction"
    if ext in TEXT_EXTS:
        return data.decode("utf-8", errors="replace"), "Text extraction"
    if ext == ".csv":
        frame = pd.read_csv(BytesIO(data))
        return frame.to_string(index=False), "CSV table extraction"
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return _ocr_image(data), "Image OCR"
    if ext in {".wav", ".mp3", ".m4a", ".mpeg", ".mpga", ".webm"}:
        from .ai_services import transcribe_audio
        return transcribe_audio(data, filename), "Audio transcription (OpenAI)"
    if ext in {".mp4", ".mov", ".mkv", ".avi"}:
        return extract_video_text(data, filename), "Video key-frame OCR"
    raise ValueError(f"Unsupported file type: {ext or 'unknown'}")

def extract_video_text(data: bytes, filename: str) -> str:
    import cv2
    from .config import MAX_VIDEO_FRAMES
    suffix = Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(data)
        temp.flush()
        cap = cv2.VideoCapture(temp.name)
        if not cap.isOpened():
            raise ValueError("Could not open video. Check that the file is a supported video format.")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        positions = sorted(set(int(i * max(0, total - 1) / max(1, MAX_VIDEO_FRAMES - 1))
                               for i in range(min(MAX_VIDEO_FRAMES, total))))
        texts = []
        for frame_no in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok:
                continue
            ok, encoded = cv2.imencode(".png", frame)
            if ok:
                text = _ocr_image(encoded.tobytes())
                if text.strip():
                    seconds = frame_no / max(cap.get(cv2.CAP_PROP_FPS), 1)
                    texts.append(f"[Frame at {seconds:.1f}s]\n{text}")
        cap.release()
    return "\n\n".join(texts) if texts else "No readable text found in sampled video frames."
