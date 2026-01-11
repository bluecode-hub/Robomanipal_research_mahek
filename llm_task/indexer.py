# indexer.py
from embedding import EmbeddingModel
from vector_store import VectorStore

docs = [
    {
        "id": 1,
        "text": "Backpropagation is an algorithm used to train neural networks by computing gradients.",
        "source": "ML Book"
    },
    {
        "id": 2,
        "text": "Gradient descent minimizes a loss function by updating parameters iteratively.",
        "source": "ML Book"
    },
    
   {
        "id": 3,
        "text": "turnover of my company is 100cr",
        "source": "Company report"
    }
]

def build_index():
    embedder = EmbeddingModel()
    store = VectorStore()

    for doc in docs:
        emb = embedder.encode([doc["text"]])[0]
        store.add(emb, doc)

    return embedder, store
