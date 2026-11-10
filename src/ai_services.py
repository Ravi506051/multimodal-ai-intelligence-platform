"""Optional OpenAI-powered transcription, vision, and grounded answers."""
import base64
from pathlib import Path
from openai import OpenAI
from . import config

def _client():
    if not config.OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY in your .env file to enable this feature.")
    return OpenAI(api_key=config.OPENAI_API_KEY)

def transcribe_audio(data: bytes, filename: str) -> str:
    import io
    client = _client()
    file_obj = io.BytesIO(data)
    file_obj.name = Path(filename).name
    result = client.audio.transcriptions.create(model=config.TRANSCRIBE_MODEL, file=file_obj)
    return result.text

def describe_image(data: bytes, mime_type: str = "image/jpeg") -> str:
    client = _client()
    encoded = base64.b64encode(data).decode("ascii")
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "Describe this image accurately. Extract visible text and explain important visual details."},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}
        ]}],
        max_tokens=700,
    )
    return response.choices[0].message.content or ""

def answer_question(question: str, contexts: list[dict]) -> str:
    client = _client()
    context = "\n\n".join(
        f"[Source: {item['source']}]\n{item['text']}" for item in contexts
    )
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Answer using only the supplied context. If the answer is not present, say so. Cite sources using [Source: filename]."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""
