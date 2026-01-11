import numpy as np
class VectorStore:
    def __init__(self):
        self.embeddings=[]
        self.metadata=[]
    def add(self,embedding,meta):
        self.embeddings.append(embedding)
        self.metadata.append(meta)
    def get_all(self):
        return np.array(self.embeddings),self.metadata