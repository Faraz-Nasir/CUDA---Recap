> Note: Before beginning, its important to note that doing warmup and benchmark runs is a good way to get a more accurate measurement of time it takes to execute a function. Without doing any warmup runs, cuBLAS will have a lot of overhead from the first run and that will skew the results (~45ms). Benchmark runs are used to get a more accurate average time.

# cuBLAS

- NVIDIA CUDA Basic Linear Algebra Subprograms, is a GPU accelerated library for accelerating AI and HPC applications.

- It includes several APIs and GEMM(General Matrix Multiplication) APIs with support for fusions that are highly optimized for NVIDIA GPUs.

-
Run with the `-lcublas` tag
 ```bash
PS> nvcc '.\CUDAProgramming\06_CUDA_APIs\01 cuBLAS\01_Hgemm_Sgemm.cu' -o 00 -lcublas 
```

## cuBLAS-Lt
- **cuBLASLt (cuda BLAS Lightweight)** is an extension of the cuBLAS library that provides a more flexible API, primarily aimed at improving performance for specific workloads like deep learning models. Close to all the datatypes and API calls are tried back to matmul.

- In cases where a problem cannot be run by a single kernel, cuBLASLt will attempt to decompose the problem into multiple sub-problems and solve it by running the kernel on each subproblem.

- this is where fp16/fp8 and int8 come in.

## cuBLAS-Xt
- cublas-xt for host + gpu solving (this is way slower)
- since we have to transfer between motherboard DRAM and GPU VRAM, we get memory bandwidth bottlenecks and can't compute as fast as doing everything on-chip;.
- **cuBLAS-Xt** is an extension to cuBLAS that enables multi-GPU support.
- Key features include:
    - **Multiple GPUs:** Ability to run BLAS operation across multiple GPUs, allowing for GPU scaling and potentially significant performance improvements on large dataset.
    - **Thread Safety:** Designed to be thread-safe, enabling concurrent execution of multiple BLAS operations on different GPUs.

Ideal for large-scale computations that can benefit from distributing workloads across multiple GPUs.

Choose Xt for large-scale linear algebra that exceeds GPU VRAM memory

![alt text](image.png)

## cuBLASDx

- The cuBLASDx library is a device side API extension for performing BLAS calculations inside the CUDA kernels. By fusing numerical operations you can decrease latency and further improve performance of your applications.
    - This however is not part of the CUDA Toolkit.

## cuTLASS

- cuBLAS and its variants are run on the host and whatever cuBLAS-DX isnt super well documented or optimized.
- matrix multiplication is the most import operation in deep learning and **cuBLAS doesn't let us easily fuse operations together.**
- **cuTLASS (CUDA Templates for Linear Algebra Subroutines) on the other hand does**