import numpy as np
import pyvecdb
import time

def test_flat():
    print("Testing Flat Index...")
    d = 128
    ntotal = 10000
    xq = np.random.random((5, d)).astype('float32')
    xb = np.random.random((ntotal, d)).astype('float32')

    index = pyvecdb.VectorDatabase("flat", dim=d)
    index.add(xb)

    print(f"Index size: {index.ntotal}")

    start = time.time()
    dists, labels = index.search(xq, k=5)
    print(f"Search time: {time.time() - start:.4f}s")

    print("Dists:\n", dists)
    print("Labels:\n", labels)
    print("Flat Index test passed.\n")

def test_ivf():
    print("Testing IVF Index...")
    d = 128
    ntotal = 20000
    xq = np.random.random((5, d)).astype('float32')
    xb = np.random.random((ntotal, d)).astype('float32') # larger dataset for training

    index = pyvecdb.VectorDatabase("ivf", dim=d, nlist=100)

    print("Training...")
    index.train(xb)
    print("Adding...")
    index.add(xb)

    print(f"Index size: {index.ntotal}")

    start = time.time()
    dists, labels = index.search(xq, k=5)
    print(f"Search time: {time.time() - start:.4f}s")

    print("Dists:\n", dists)
    print("Labels:\n", labels)
    print("IVF Index test passed.\n")

def test_hnsw():
    print("Testing HNSW Index...")
    d = 128
    ntotal = 5000
    xq = np.random.random((5, d)).astype('float32')
    xb = np.random.random((ntotal, d)).astype('float32')

    index = pyvecdb.VectorDatabase("hnsw", dim=d, M=16, efConstruction=100)

    print("Adding...")
    index.add(xb)

    print(f"Index size: {index.ntotal}")

    start = time.time()
    dists, labels = index.search(xq, k=5)
    print(f"Search time: {time.time() - start:.4f}s")

    print("Dists:\n", dists)
    print("Labels:\n", labels)
    print("HNSW Index test passed.\n")

if __name__ == "__main__":
    test_flat()
    test_ivf()
    test_hnsw()
