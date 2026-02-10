from ._pyvecdb import IndexFlat, IndexIVF, IndexHNSW

def read_fvecs(filename):
    import numpy as np
    a = np.fromfile(filename, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view('float32')

class VectorDatabase:
    """
    High-level Python wrapper for pyvecdb indexes.
    """
    def __init__(self, method="flat", dim=128, **kwargs):
        self.method = method
        self.dim = dim
        self.index = None

        if method.lower() == "flat":
            self.index = IndexFlat(dim)
        elif method.lower() == "ivf":
            nlist = kwargs.get("nlist", 100)
            self.index = IndexIVF(dim, nlist)
        elif method.lower() == "hnsw":
            M = kwargs.get("M", 16)
            efCheck = kwargs.get("efConstruction", 200)
            self.index = IndexHNSW(dim, M, efCheck)
        else:
            raise ValueError(f"Unknown method: {method}")

    def add(self, vectors):
        self.index.add(vectors)

    def train(self, vectors):
        self.index.train(vectors)

    def search(self, query, k=5):
        return self.index.search(query, k)

    @property
    def ntotal(self):
        return self.index.get_ntotal()
