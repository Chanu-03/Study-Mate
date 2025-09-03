# -------------------------
# vector_store.py — Simple in-memory vector store
# -------------------------
import numpy as np
from typing import List, Dict

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

class SimpleVectorStore:
    def __init__(self):
        self.vectors: List[np.ndarray] = []
        self.metadatas: List[Dict] = []

    def add(self, embeddings: List[List[float]], metadatas: List[Dict]):
        for emb, md in zip(embeddings, metadatas):
            self.vectors.append(np.array(emb))
            self.metadatas.append(md)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        query_vec = np.array(query_vector)
        if len(self.vectors) == 0:
            return []
        similarities = [cosine_similarity(query_vec, vec) for vec in self.vectors]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "score": similarities[idx],
                "metadata": self.metadatas[idx]
            })
        return results

    def reset(self):
        self.vectors = []
        self.metadatas = []
