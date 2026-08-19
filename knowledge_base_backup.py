
import os

from dotenv import load_dotenv

load_dotenv()
import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rich.console import Console
from config import (
    KB_DIR, CASES_FILE,
    EMBEDDING_MODEL, EMBEDDING_REVISION,
    TOP_K_CHUNKS, SIMILARITY_THRESHOLD
)

console = Console()

class KnowledgeBase:
    def __init__(self):
        self.chunks = []
        self.metadata = []
        self.index = None
        self.embedder = None
        self._load_embedder()
        self._load_documents()
        self._build_index()

    def _load_embedder(self):
        console.log(
            f"[cyan]Loading embedding model: "
            f"{EMBEDDING_MODEL}[/cyan]")
        t0 = time.time()
        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL,
            revision=EMBEDDING_REVISION
        )
        elapsed = round(time.time() - t0, 2)
        console.log(
            f"[green]✅ Embedder loaded in "
            f"{elapsed}s[/green]")

    def _chunk_text(self, text: str,
                    chunk_size: int = 300,
                    overlap: int = 50) -> list:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(
                words[i:i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap
        return chunks

    def _load_documents(self):
        """Load all KB markdown files"""
        console.log("[cyan]Loading knowledge base...[/cyan]")

        
        if os.path.exists(KB_DIR):
            for fname in sorted(os.listdir(KB_DIR)):
                if fname.endswith(".md"):
                    fpath = os.path.join(KB_DIR, fname)
                    with open(fpath, "r") as f:
                        content = f.read()

                    chunks = self._chunk_text(content)
                    for i, chunk in enumerate(chunks):
                        self.chunks.append(chunk)
                        self.metadata.append({
                            "source": fname,
                            "chunk_id": i,
                            "type": "knowledge_base"
                        })

        # Load resolved cases
        if os.path.exists(CASES_FILE):
            with open(CASES_FILE) as f:
                data = json.load(f)

            for case in data.get("cases", []):
                # Skip superseded cases
                if case.get("status") == "superseded":
                    continue

                case_text = (
                    f"Case {case['case_id']}: "
                    f"{case['title']}. "
                    f"Symptoms: "
                    f"{'. '.join(case.get('symptoms', []))}. "
                    f"Resolution: "
                    f"{'. '.join(case.get('resolution', []))}"
                )
                self.chunks.append(case_text)
                self.metadata.append({
                    "source": case["case_id"],
                    "chunk_id": 0,
                    "type": "resolved_case",
                    "status": case.get("status")
                })

        console.log(
            f"[green]✅ Loaded {len(self.chunks)} "
            f"chunks[/green]")

    def _build_index(self):
        """Build FAISS vector index"""
        console.log("[cyan]Building FAISS index...[/cyan]")
        t0 = time.time()

        embeddings = self.embedder.encode(
            self.chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Normalize for cosine similarity
        norms = np.linalg.norm(
            embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

        elapsed = round(time.time() - t0, 2)
        console.log(
            f"[green]✅ Index built in "
            f"{elapsed}s — {len(self.chunks)} "
            f"vectors[/green]")

    def retrieve(self, query: str,
                 top_k: int = TOP_K_CHUNKS) -> list:
        """Retrieve top-k relevant chunks"""
        q_emb = self.embedder.encode(
            [query], convert_to_numpy=True)
        q_emb = q_emb / (
            np.linalg.norm(q_emb, keepdims=True) + 1e-8)

        scores, indices = self.index.search(
            q_emb.astype(np.float32), top_k)

        results = []
        for score, idx in zip(
                scores[0], indices[0]):
            if idx < 0:
                continue
            if score < SIMILARITY_THRESHOLD:
                continue
            results.append({
                "text": self.chunks[idx],
                "score": float(score),
                "source": self.metadata[idx]["source"],
                "type": self.metadata[idx]["type"]
            })

        return results