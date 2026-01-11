import numpy as np

def cosine_similarity(query_vec, doc_vecs):
    return np.dot(doc_vecs, query_vec)
