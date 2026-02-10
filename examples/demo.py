import numpy as np
import pyvecdb
import time

def main():
    print("=== PyVecDB Demo ===\n")

    # Parameters
    dim = 128
    nb = 10000  # Database size
    nq = 5      # Number of queries

    # Generate random data
    print(f"Generating random data (d={dim}, n={nb})...")
    np.random.seed(42)
    xb = np.random.random((nb, dim)).astype('float32')
    xq = np.random.random((nq, dim)).astype('float32')

    # --- Flat Index ---
    print("\n[1] Testing Flat Index (Exact Search)")
    index_flat = pyvecdb.VectorDatabase("flat", dim=dim)
    index_flat.add(xb)
    print(f"Total vectors: {index_flat.ntotal}")

    start = time.time()
    dists, labels = index_flat.search(xq, k=5)
    t = time.time() - start
    print(f"Search time: {t:.4f}s")
    print("Nearest neighbors for first query:", labels[0])

    # --- IVF Index ---
    print("\n[2] Testing IVF Index (Inverted File)")
    # Train heavily relies on data distribution, random data is hard for KMeans but works for demo
    nlist = 100
    index_ivf = pyvecdb.VectorDatabase("ivf", dim=dim, nlist=nlist)
    print("Training IVF...")
    index_ivf.train(xb)
    print("Populating index...")
    index_ivf.add(xb)

    start = time.time()
    dists_ivf, labels_ivf = index_ivf.search(xq, k=5)
    t = time.time() - start
    print(f"Search time: {t:.4f}s")
    print("Nearest neighbors for first query:", labels_ivf[0])

    # --- HNSW Index ---
    print("\n[3] Testing HNSW Index (Graph)")
    index_hnsw = pyvecdb.VectorDatabase("hnsw", dim=dim, M=16, efConstruction=200)
    print("Building graph...")
    index_hnsw.add(xb)

    start = time.time()
    dists_hnsw, labels_hnsw = index_hnsw.search(xq, k=5)
    t = time.time() - start
    print(f"Search time: {t:.4f}s")
    print("Nearest neighbors for first query:", labels_hnsw[0])

    print("\n=== Demo Completed Successfully ===")

if __name__ == "__main__":
    main()
