"""Retrieval over the (fictional) policy docs in knowledge/: chunk -> embed -> FAISS -> cited hits.

Why: an LLM answering from memory makes things up. Retrieval finds the passage that actually
contains the answer, so the model can quote it and we can show the source (a citation).

Build the index:  make rag-index   (writes data/processed/rag_index/, gitignored)
The embedder is swappable: real runs use all-MiniLM-L6-v2; tests pass a tiny fake one,
so CI never downloads a model.
"""
from __future__ import annotations

import json
import re
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

# isort: off
# Order matters: torch and faiss each bundle their own libomp. On macOS, if faiss's copy loads
# first, torch segfaults when it encodes text. Loading torch first is safe (verified both ways).
import torch  # noqa: F401
import faiss
# isort: on

from shipsense.config import ROOT

KNOWLEDGE_DIR = ROOT / "knowledge"
INDEX_DIR = ROOT / "data" / "processed" / "rag_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_WORDS, OVERLAP = 150, 30  # MiniLM reads ~256 tokens (~190 words); stay under it


@dataclass
class Chunk:
    chunk_id: int
    doc: str       # file name, e.g. "03_refund_and_credit_policy.md"
    title: str     # the doc's "# " heading
    section: str   # the "## " heading the passage sits under
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc} § {self.section}"


@dataclass
class Hit:
    chunk: Chunk
    score: float   # cosine similarity, higher = closer


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Wraps the Hugging Face model; imported lazily because it pulls in torch."""

    def __init__(self, model_name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")


def chunk_markdown(path: Path, max_words: int = MAX_WORDS, overlap: int = OVERLAP) -> list[Chunk]:
    """One chunk per '## ' section; long sections become overlapping word windows."""
    title, section, sections = path.stem, "Introduction", {}
    for line in path.read_text().splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            section = line[3:].strip()
        elif line.startswith("> **FICTIONAL"):
            continue  # the disclaimer banner is on every doc; indexing it only adds noise
        else:
            sections.setdefault(section, []).append(line)

    chunks: list[Chunk] = []
    for name, lines in sections.items():
        words = " ".join(lines).split()
        step = max_words - overlap
        for start in range(0, max(len(words) - overlap, 1), step):
            text = " ".join(words[start:start + max_words])
            if text:
                chunks.append(Chunk(len(chunks), path.name, title, name, text))
    return chunks


def load_knowledge(folder: Path = KNOWLEDGE_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(folder.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        for c in chunk_markdown(path):
            c.chunk_id = len(chunks)
            chunks.append(c)
    return chunks


def embed_text(c: Chunk) -> str:
    """What gets embedded: title + section + text, so a passage keeps its context."""
    return f"{c.title}. {c.section}. {c.text}"


class PolicyIndex:
    """FAISS inner-product index over normalised vectors (inner product = cosine similarity)."""

    def __init__(self, chunks: list[Chunk], index: faiss.Index, embedder: Embedder):
        self.chunks, self.index, self.embedder = chunks, index, embedder

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> PolicyIndex:
        vectors = embedder.encode([embed_text(c) for c in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])  # exact search; fine for a few hundred chunks
        index.add(vectors)
        return cls(chunks, index, embedder)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        scores, ids = self.index.search(self.embedder.encode([query]), min(k, len(self.chunks)))
        return [Hit(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0], strict=True) if i >= 0]

    def save(self, folder: Path = INDEX_DIR) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "index.faiss"))
        (folder / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks], indent=1))

    @classmethod
    def load(cls, embedder: Embedder, folder: Path = INDEX_DIR) -> PolicyIndex:
        chunks = [Chunk(**c) for c in json.loads((folder / "chunks.json").read_text())]
        return cls(chunks, faiss.read_index(str(folder / "index.faiss")), embedder)


class HashingEmbedder:
    """Deterministic bag-of-words embedder for tests: no model download, same words -> close vectors."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9$%]+", text.lower()):
                # crc32 is stable across runs (Python's built-in hash() is randomised per process)
                out[row, zlib.crc32(word.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


if __name__ == "__main__":
    chunks = load_knowledge()
    idx = PolicyIndex.build(chunks, SentenceTransformerEmbedder())
    idx.save()
    print(f"indexed {len(chunks)} chunks from {len({c.doc for c in chunks})} docs -> {INDEX_DIR.relative_to(ROOT)}")
    for q in ["How much credit does a customer get if the order is 2 days late?",
              "When is a shipment considered lost?"]:
        top = idx.search(q, k=2)
        print(f"\nQ: {q}")
        for h in top:
            print(f"  {h.score:.3f}  {h.chunk.citation}")
