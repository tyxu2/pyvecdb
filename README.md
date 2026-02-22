# pyvecdb: A Lightweight Hybrid Vector Database

`pyvecdb` is a high-performance, lightweight vector database implemented in C++ with user-friendly Python bindings. It supports state-of-the-art indexing algorithms and hardware acceleration.

## Features

*   **Algorithms**:
    *   **Flat**: Brute-force exact search (Baseline).
    *   **IVF (Inverted File Index)**: Fast approximate search using K-Means clustering (reducers search space).
    *   **HNSW (Hierarchical Navigable Small World)**: Graph-based index for ultra-fast approximate nearest neighbor search.
*   **Performance**:
    *   **C++ Core**: All heavy lifting (distance calcs, graph traversal) is done in optimized C++.
    *   **CUDA Acceleration**: Optional GPU path; current default build is CPU-only (see Configuration).

## Project Structure

```text
pyvecdb/
├── CMakeLists.txt       # CMake build configuration (handles C++/CUDA)
├── setup.py             # Python packaging (invokes CMake)
├── pyvecdb/             # Python package source
│   ├── __init__.py      # High-level Python wrapper
│   └── ...
├── src/                 # C++ Core Source
│   ├── bindings.cpp     # pybind11 definitions
│   ├── Index.h          # Base class
│   ├── IndexFlat.cpp    # Brute force index
│   ├── IndexIVF.cpp     # IVF implementation
│   ├── IndexHNSW.cpp    # HNSW implementation
│   ├── Distance.cu      # CUDA kernels (optional)
│   └── ...
```

## Installation

**Prerequisites**:
*   C++ Compiler (GCC/Clang/MSVC) with C++17 support.
*   CMake >= 3.18
*   Python 3.8+
*   (Optional) CUDA Toolkit for GPU support.

```bash
# Clone and install (CPU-only by default)
pip install .
```

If you want GPU support, see the CUDA section below and rebuild.

## Configuration & Build Options

### 1. CUDA Acceleration
CUDA support exists, but the current build defaults to CPU-only.
*   **Default**: `CMakeLists.txt` sets `DISABLE_CUDA` to `ON`.
*   **Enable**: Set `DISABLE_CUDA` to `OFF`, ensure `nvcc` is in your `$PATH`, then reinstall.
*   **Verify**:
    ```python
    from pyvecdb import is_cuda_enabled
    print(f"CUDA Active: {is_cuda_enabled()}")
    ```
*   **Fallback**: If CUDA is not found or disabled, it runs in CPU-only mode.

### 2. Index Hyperparameters
You can tune the performance/accuracy trade-off using these parameters:

| Algorithm | Parameter | Default | Description | Tuning Tip |
|-----------|-----------|---------|-------------|------------|
| **IVF** | `nlist` | 100 | Number of clusters (centroids). | Approx ~`4 * sqrt(N)` |
| **IVF** | `nprobe` | 1 | Clusters to visit during search. | Increase to `10-50` for higher recall. |
| **HNSW** | `M` | 16 | Neighbors per node in graph. | `16-64`. Higher = better accuracy, more memory. |
| **HNSW** | `efConstruction` | 200 | Search depth during index build. | `100-500`. Higher = better graph quality. |
| **HNSW** | `ef` | 50 | Search depth during query. | Increase to `100+` for higher recall. |

## Usage

### Quick Start
```python
import numpy as np
import pyvecdb

# Create a small index and search
vectors = np.random.random((100, 128)).astype("float32")
query = np.random.random((1, 128)).astype("float32")

db = pyvecdb.VectorDatabase("flat", dim=128)
db.add(vectors)

dists, labels = db.search(query, k=5)
print(labels)
```

### 1. Flat Index (Exact Search)
```python
import numpy as np
import pyvecdb

# Create data
d = 128
data = np.random.random((10000, d)).astype('float32')
query = np.random.random((5, d)).astype('float32')

# Initialize and Add
db = pyvecdb.VectorDatabase("flat", dim=d)
db.add(data)

# Search
dists, indices = db.search(query, k=5)
print(indices)
```

### 2. IVF Index (Fast Approximate Search)
Suitable for large datasets. Requires training.

```python
# Initialize (nlist = number of clusters)
db = pyvecdb.VectorDatabase("ivf", dim=128, nlist=100)

# Train and Add
db.train(data) # Computes centroids
db.add(data)   # Assigns vectors to clusters

# Search
results = db.search(query, k=5)
```

### 3. HNSW Index (Graph Based)
Best trade-off between speed and accuracy.

```python
# Initialize (M=neighbors per node, efConstruction=build depth)
db = pyvecdb.VectorDatabase("hnsw", dim=128, M=16, efConstruction=200)

db.add(data)
results = db.search(query, k=5)
```

## Troubleshooting

*   **ImportError: undefined symbol: fatbinData**: Build likely picked CUDA without proper linking; keep CPU-only by leaving `DISABLE_CUDA` set to `ON` and reinstall.
*   **AttributeError: module 'pyvecdb' has no attribute 'is_cuda_enabled'**: Reinstall the package and restart Python so the updated `pyvecdb/__init__.py` is picked up.
