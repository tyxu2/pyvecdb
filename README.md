# pyvecdb: A Lightweight Hybrid Vector Database

`pyvecdb` is a high-performance, lightweight vector database implemented in C++ with user-friendly Python bindings. It supports state-of-the-art indexing algorithms and hardware acceleration.

## Features

*   **Algorithms**:
    *   **Flat**: Brute-force exact search (Baseline).
    *   **IVF (Inverted File Index)**: Fast approximate search using K-Means clustering (reducers search space).
    *   **HNSW (Hierarchical Navigable Small World)**: Graph-based index for ultra-fast approximate nearest neighbor search.
*   **Performance**:
    *   **C++ Core**: All heavy lifting (distance calcs, graph traversal) is done in optimized C++.
    *   **CUDA Acceleration**: Automatically detects CUDA capable GPUs to accelerate distance matrix calculations (essential for Flat index and IVF training).
*   **Usability**:
    *   **Pythonic API**: Clean Python wrapper (`VectorDatabase`) that behaves like standard libraries.
    *   **Seamless Integration**: Takes numpy arrays as input.

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
# Clone and install
pip install .
```

If CMake finds `nvcc`, it will automatically compile the GPU kernels.

## Configuration & Build Options

### 1. CUDA Acceleration
CUDA support is **automatically detected** at build time.
*   **Enable**: Ensure `nvcc` is in your system `$PATH` before installing.
*   **Verify**:
    ```python
    import pyvecdb
    from pyvecdb._pyvecdb import is_cuda_enabled
    print(f"CUDA Active: {is_cuda_enabled()}")
    ```
*   **Fallback**: If CUDA is not found, it gracefully degrades to CPU-only mode.

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

## Implementation Details

*   **Hybrid Build**: We use `scikit-build` style logic (via custom `setup.py`) to bridge CMake and Python setuptools.
*   **Memory Management**: Vectors are stored in contiguous C++ `std::vector` memory, exposed to Python as needed.
*   **Extensibility**: The `Index` base class makes it easy to add new algorithms (e.g., PQ, LSH) in the future.
