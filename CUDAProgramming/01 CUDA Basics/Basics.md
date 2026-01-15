# Nuts & Bolts of the Operation

## Kernel
A kernel is a special function that runs on your GPU instead of the CPU. Think of it like giving instructions to a large team at once, who all can work at the same time. You mark a kernel with `__global__` keyword, and it can only return void.

Example:
```cpp
___global___ void addNumbers(int* a,int* b,int* result){
    *result=*a+*b;
}
```
## Grid
The Grid represents the entire set of threads launched for a single kernel invocation. Think of it as the overall execution space. It's a collection of thread blocks.

When you launch a kernel, you specify the dimensions of the grid, essentially how many blocks will be created which can be 1D,2D or 3D(like a line,sheet or cube of blocks). Used for Organizing really large computations.
- Example: When processing a large image,each block might handle one section of the image.

## Block
A block is a group of threads that can cooperate and share data quickly through shared memory. It can also be 1D,2D or 3D.
- Threads within a block can:
    - Share Memory
    - Synchronize with each other
    - Cooperate on tasks
- Example: If processing an image, a block migh handle a 16x16 pixel region

## Threads
The thread is the smallest unit of exection in CUDA. Each thread executes the kernel code independently. Within a block, threads are idenitifed by a unique thread ID. This ID allows you to access specific data or perform different operations based on the threads position within it's block.
- Threads have their own unique ID to know which data to work on.

## Understanding CUDA Thread Indexing.
In CUDA, each threads has a unique identifier that can be used to determine its position within the grid and block.The following variables are commonly used for this purpose
- threadIdx:
    - A 3 component vector(1,2,3D) with threadIdx.x,threadIdx.y,threadIdx.z that gives the thread's position within its ***block***.
    - Example:If you have a 1D block of 256 threads, threadIdx.x would range from 0 to 255.

- blockDim:
    - A 3 component vector with blockDim.x, blockDim.y, blockDim.z that specifies dimensions of the block.
    - If the block has 256 threads in the x-direction, then blockDim.x is 256

- blockIdx:
    - A 3 component vector that gives the block's position within its grid.
    - If its a 1D grid of 10 blocks then blockIdx.x ranges from 0 to 9

- gridDim:
    - A 3 component vector(gridDim.x,gridDim.y,gridDim.z) that specifies the dimensions of the grid.
    - Example: If the grid has 10 blocks in the x-direction, gridDim.x is 10.


## Important- How to calculate global thread ID

- To compute a unique global thread ID (for accessing elements in a 1D array).

```cpp
int globalThreadId=blockIdx.x*blockDim.x+threadIdx.x
```
- blockIdx.x*blockDim.x gives the number of starting for the current block
- threadIdx.x gives the threads position within that block.

### Helper Types & Functions
- dim3
    - A simple way to specifiy 3D dimensions
    - Used for grid and block sizes
    - Example:
    ```cpp
    dim3 blockSize(16,16,1); //16x16x1 threads in the block
    dim3 gridSize(8,8,1);//8x8x1 blocks in the grid
    ```

- <<<>>>

This is used to configure and launch kernels on the GPU. It defines the grid and block dimensions, shared memory size and stream for kernel execution. Properly configuring these parameters are essentials for efficient GPU programming and maximizing performance.

-   Special brackets for launching kernels
-   Specifies grid and block dimensions
-   Example:
```cpp
addNumbers<<<gridSize,BlockSize>>>(a,b,result);
```

Here `addNumbers` is the name of the kernel and `gridSize` & `blockSize` is the size of the grid and block respectively. Rest are the parameters passed to the kernel.

## Memory Management
- ### cudaMalloc
    - Allocates memory on the GPU
    - Similar to regular malloc but for GPU memory.
    - Example
    ```cpp
    int *device_array;
    cudaMalloc(&deviceArray,size*sizeof(int));
    ```
-   ### cudaMemcpy

It can copy data from device to host and vice versa.

-   cudaMemcpyHostToDevice,
-   cudaMemcpyDeviceToHost,
-   cudaMemcpyDeviceToDevice,

-   ### `cudaFree` will free memory on the device.

```cpp
cudaMemcpy(deviceArray,Host_array,size*sizeof(int),cudaMemcpyHostToDevice);
```

-   ### cudaDeviceSynchronize() 

    - By default CPU and GPU work together to minimize time but sometime we need to make the CPU wait until all GPU operations are complete there we use `cudaDeviceSynchronize()`
        - Useful for when you need results before continuing
        - We will use it while performing benchmarks
        - Example:
        ```cpp
        kernel<<<gridSize,blockSize>>>(data);
        cudaDeviceSynchronize(); //Waits for the kernel above to finish
        printf("Kernel Completed");
        ``` 


## Memory
CUDA provides several types of memory, each with different speeds and usecases.

- ### Global Memory:
    - The main memory of the GPU, accessible by all threads.
    - Slowest but largest in size.
    - Use for data shared across all threads.
    - Example: Arrays or large datasets.
- ### Shared Memory:
    - Memory shared by all threads in a **block**.
    - Very fast but small in size.
    - Use for data that threads in a block need to share frequently

- ### Registers:
    - Fastest memory, private to each **thread**.
    - Used for local variables.
    - Limited in number.
    - Example: Loop counters or intermediate calculations.

- ### Constant Memory:
    - Read-only memory for all the threads.
    - Cached for fast access.
    - Used for data that doesn't change.

- ### Local Memory:
    - Used when registers are not enough.
    - Slow (stored in global memory).
    - Avoid if possible
    - Example: Large arrays or variables that don't fit in registers.


# Conclusion

```cpp
//Kernel Definition
___global___ void addArrays(int* a,int* b,int* c,int size){
    //Calculating unique index for the threadID;
    int index=blockDim.x*blockIdx.x+threadIdx.x;

    if(index<size>){
        c[index]=a[index]+b[index];
    }
}

int main(){
    dim3 blockSize(256);
    dim3 gridSize((size+blockSize.x-1)/blockSize.x);

    addArrays<<<gridSize,blockSize>>>(d_a,d_b,d_c,size);
}
```







