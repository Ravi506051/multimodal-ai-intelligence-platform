"""Lightweight, CPU-friendly retrieval over extracted text chunks."""
from dataclasses import dataclass
from typing import Iterable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Chunk:
    source: str
    text: str

class LocalRetriever:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
        self.matrix = None

    def add_documents(self, documents: Iterable[tuple[str, str]], chunk_size: int = 900, overlap: int = 140):
        for source, text in documents:
            clean = " ".join((text or "").split())
            if not clean:
                continue
            step = max(1, chunk_size - overlap)
            for start in range(0, len(clean), step):
                part = clean[start:start + chunk_size]
                if part:
                    self.chunks.append(Chunk(source=source, text=part))
        self._fit()

    def _fit(self):
        if not self.chunks:
            self.matrix = None
            return
        try:
            self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])
        except ValueError:
            self.matrix = None

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.chunks or self.matrix is None or not query.strip():
            return []
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        return [{"source": self.chunks[i].source, "text": self.chunks[i].text,
                 "score": float(scores[i])} for i in ranked if scores[i] > 0]
