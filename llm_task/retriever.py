from similarity import cosine_similarity

class SemanticRetriever:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query, top_k=3):
        query_vec = self.embedder.encode([query])[0]
        doc_vecs, metadata = self.vector_store.get_all()

        scores = cosine_similarity(query_vec, doc_vecs)
        ranked = sorted(
            zip(scores, metadata),
            key=lambda x: x[0],
            reverse=True
        )

        return ranked[:top_k]
