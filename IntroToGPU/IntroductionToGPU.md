# Introduction to GPU's

## Hardware
- CPU: Central Processing Unit
    - General Purpose
    - Higher Clock Speed
    - Fewer Cores
    - High Cache
    - Low Latency
    - Low throughput
- GPU: Graphical Processing Unit
    - Specialized
    - Low Clock Speed
    - More Cores
    - Low Cache
    - High Latency 
    - High throughput


## Terminology
- CPU (Host)
    - Minimizes time taken by one task
    - Metric: Latency in seconds
- GPU (Device)
    - Maximize Throughput
    -Metric: Throughput in tasks per seconds (eg: pixels per ms)

## Typical CUDA Program
- CPU allocates CPU Memory 
- CPU copies data to GPU
- CPU launches kernel on GPU (All processing is done here)
- CPU copies result back from GPU back to CPU for next steps.


Kernel looks like a serial program; says nothing about parallelism. Imagine that you are trying to solve a jigsaw puzzle and all you are given is the location of each puzzle piece. The high level algorithm would be designed to take these individual pieces and solve a single problem for each of them. As long as all the pieces are assembled in the right place at the end, it would work. You would'nt need to start at one corner and work your way across the puzzle. Instead we can solve multiple pieces at the same time, as long as they don't interfere with each other.

## Things to remember
- GPU is the device, which executes GPU functions called kernels
- CPU is the host, this shifts data from its memory to GPU memory.
- While writing Kernel algorithm, look at the whole picture and examine how each piece can be changed at the same time to get the final result instead of writing sequentially like convential CPU algorithms.