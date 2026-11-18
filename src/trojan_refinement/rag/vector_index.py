"""Exact FAISS cosine retrieval with filtering before nearest-neighbor selection."""

from pathlib import Path


class CosineIndex:
    def __init__(self, vectors):
        import faiss
        import numpy as np

        vectors = np.array(vectors, dtype="float32", order="C", copy=True)
        if vectors.ndim != 2 or not all(vectors.shape) or not np.isfinite(vectors).all():
            raise ValueError("Embedding matrix must be finite and nonempty")
        if (np.linalg.norm(vectors, axis=1) <= 1e-12).any():
            raise ValueError("Zero embeddings cannot be used for cosine retrieval")
        faiss.normalize_L2(vectors)
        self.vectors = vectors
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, queries, top_k, allowed=None):
        import faiss
        import numpy as np

        queries = np.array(queries, dtype="float32", order="C", copy=True)
        if queries.ndim != 2 or queries.shape[1] != self.index.d or not np.isfinite(queries).all():
            raise ValueError("Query dimensions or values do not match the index")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if (np.linalg.norm(queries, axis=1) <= 1e-12).any():
            raise ValueError("Zero query embeddings cannot be normalized")
        faiss.normalize_L2(queries)
        rows = list(range(self.index.ntotal)) if allowed is None else sorted(set(allowed))
        if any(type(i) is not int or not 0 <= i < self.index.ntotal for i in rows):
            raise ValueError("Unknown vector row")
        if not rows:
            return [[] for _ in queries]
        index = self.index
        if rows != list(range(index.ntotal)):
            index = faiss.IndexFlatIP(self.index.d)
            index.add(np.ascontiguousarray(self.vectors[rows]))
        scores, neighbors = index.search(queries, min(top_k, len(rows)))
        return [
            [(rows[int(i)], float(score)) for i, score in zip(ids, values) if i >= 0]
            for ids, values in zip(neighbors, scores)
        ]

    def save(self, path):
        import faiss

        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path):
        import faiss

        index = faiss.read_index(str(Path(path)))
        if not isinstance(index, faiss.IndexFlatIP) or index.ntotal < 1:
            raise ValueError("Expected a nonempty FAISS IndexFlatIP")
        return cls(index.reconstruct_n(0, index.ntotal))
