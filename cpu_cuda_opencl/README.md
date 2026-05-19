# Parallel Graph Algorithms: CPU vs CUDA vs OpenCL

Benchmarking three classic graph algorithms — **BFS**, **PageRank**, and **SSSP (Bellman-Ford)** — across CPU, CUDA, and OpenCL on an NVIDIA A100-SXM4-40GB GPU.

> CS259 Term Project · Google Colab · CUDA 13.0 · OpenCL 3.0

---

## Table of Contents

- [Overview](#overview)
- [Algorithms](#algorithms)
- [Results Summary](#results-summary)
- [Hardware & Software](#hardware--software)
- [Getting Started](#getting-started)
- [Benchmark Details](#benchmark-details)
- [Key Findings](#key-findings)

---

## Overview

This project implements and benchmarks three fundamental graph algorithms in three execution environments:

| Backend | Language | Parallelism |
|---------|----------|-------------|
| CPU | C / C++ | Single-threaded |
| CUDA | CUDA C | GPU (NVIDIA-native) |
| OpenCL | OpenCL C | GPU (cross-platform) |

Graphs are represented in **Compressed Sparse Row (CSR)** format. Synthetic random graphs are used for BFS and PageRank; Bellman-Ford is tested on larger random graphs. PageRank also includes the real-world **web-Stanford** dataset (281K nodes, ~2.3M edges) from [SNAP](https://snap.stanford.edu/data/web-Stanford.html).

---

## Algorithms

### Breadth-First Search (BFS)
- Frontier-based level-synchronous BFS
- GPU version: one thread per frontier node, `atomicCAS` for visited marking
- Datasets: Small (1K), Medium (64K), Large (256K), XLarge (1M nodes)

### PageRank
- Iterative power-iteration with damping factor `d = 0.85`, convergence threshold `ε = 1e-6`, max 200 iterations
- GPU version: 4 kernels — dangling sum, rank init, scatter via in-edges, convergence diff
- Datasets: Small (1K) → XLarge (1M) + web-Stanford (281K real-world)

### SSSP — Bellman-Ford
- Parallel edge-relaxation with early-exit on no updates
- Negative cycle detection included
- Datasets: Medium (100K), Large (1M), XLarge (10M nodes)

---

## Results Summary

### BFS Execution Time

| Dataset | CPU (ms) | CUDA (ms) | OpenCL (ms) |
|---------|----------|-----------|-------------|
| Small (1K) | 0.19 | 1.40 | 0.87 |
| Medium (64K) | 20.60 | 1.46 | 2.29 |
| Large (256K) | 117.11 | 3.70 | 9.52 |
| XLarge (1M) | 823.86 | 11.76 | 19.40 |

### PageRank Execution Time

| Dataset | CPU (ms) | CUDA (ms) | OpenCL (ms) |
|---------|----------|-----------|-------------|
| Small (1K) | 0.25 | 0.76 | 1.02 |
| Medium (64K) | 9.05 | 0.73 | 1.06 |
| Large (256K) | 69.13 | 1.57 | 2.52 |
| XLarge (1M) | 565.70 | 2.99 | 3.11 |
| web-Stanford (281K) | 790.00 | 508.04 | 530.38 |

### SSSP (Bellman-Ford) Execution Time

| Dataset | CPU (ms) | CUDA (ms) | OpenCL (ms) |
|---------|----------|-----------|-------------|
| Medium (100K) | 63.47 | 1.41 | 5.24 |
| Large (1M) | 1,808 | 8.80 | 40.65 |
| XLarge (10M) | ~25,000 | 101 | 257 |

---

## Hardware & Software

```
GPU:    NVIDIA A100-SXM4-40GB (108 SMs, ~13,824 CUDA cores, 40 GB VRAM)
Driver: 580.82.07
CUDA:   13.0
OpenCL: 3.0 (via pocl + ocl-icd)
OS:     Ubuntu 24 (Google Colab)
```

**Compiler flags:**
```bash
# CUDA
nvcc -O2 -arch=sm_80

# CPU / OpenCL
g++ -O2 -std=c++17
gcc -O2 -DCL_TARGET_OPENCL_VERSION=300
```

---


## Getting Started

### Requirements

```bash
# CUDA toolkit
nvcc --version   # >= 11.0 recommended

# OpenCL
sudo apt-get install ocl-icd-opencl-dev opencl-headers pocl-opencl-icd
```

### Build & Run

**BFS**
```bash
g++ -O2 -o bfs bfs.cpp && ./bfs
nvcc -O2 -arch=sm_80 -o bfs_cuda bfs_cuda.cu && ./bfs_cuda
gcc -O2 -o bfs_opencl bfs_opencl.c -lOpenCL && ./bfs_opencl
```

**PageRank**
```bash
# Download real-world graph (optional)
wget https://snap.stanford.edu/data/web-Stanford.txt.gz && gunzip web-Stanford.txt.gz

g++ -O2 -std=c++17 -o pagerank_cpu page_rank.c && ./pagerank_cpu
nvcc -O2 -arch=sm_80 -o pagerank_cuda pagerank_cuda.cu && ./pagerank_cuda
g++ -O2 -std=c++17 -o pagerank_opencl pagerank_opencl.cpp -lOpenCL && ./pagerank_opencl
```

**SSSP (Bellman-Ford)**
```bash
gcc -O2 -o bf_cpu sssp_cpu.c -lm && ./bf_cpu
nvcc -O2 -o bf_cuda sssp_cuda.cu && ./bf_cuda
gcc -O3 -DCL_TARGET_OPENCL_VERSION=300 -o bf_ocl sssp_opencl.c -lOpenCL -lm && ./bf_ocl
```

### Run Everything in Colab

Open `APP_Term_project.ipynb` in Google Colab with a GPU runtime (A100 or T4) and run all cells in order.

---

## Benchmark Details

### Graph Generation

All synthetic graphs use a fixed seed (`srand(42)`) for reproducibility. Each node is assigned `avg_degree = 8` random out-edges, stored in CSR format.

### Memory Measurement

| Backend | Method |
|---------|--------|
| CPU | `/proc/self/status` — VmRSS delta |
| CUDA | `cudaMemGetInfo()` before/after allocation |
| OpenCL | `clGetMemObjectInfo(CL_MEM_SIZE)` summed across buffers |

### PageRank Parameters

```
Damping factor:       0.85
Convergence epsilon:  1e-6
Max iterations:       200
Dangling node handling: rank redistributed uniformly
```

---

## Key Findings

**CUDA dominates at scale.** On the XLarge BFS graph (1M nodes), CUDA is ~70× faster than CPU and ~1.6× faster than OpenCL.

**GPU overhead matters for small graphs.** On the Small BFS dataset (1K nodes), CPU (0.19 ms) outperforms both CUDA (1.40 ms) and OpenCL (0.87 ms) due to kernel launch and data transfer overhead dominating actual compute time.

**OpenCL is competitive but slower than CUDA.** Across all three algorithms, OpenCL runs 1.6–4.6× slower than CUDA on the same A100 hardware, likely due to the abstraction overhead of the OpenCL runtime and less aggressive compiler optimization compared to `nvcc`.

**Bellman-Ford sees the largest GPU speedup.** At XLarge (10M nodes, 50M edges), CUDA achieves roughly a 250× speedup over CPU — the algorithm's high edge count makes it extremely parallelism-friendly.

**Memory scales predictably.** BFS and PageRank GPU memory footprints mirror CPU footprints (CSR graph + rank arrays), while Bellman-Ford's edge-list representation requires ~3 separate arrays (src, dst, weight) on the GPU.


