# CUDA Streams

## Intuition
 You can think of streams as "river streams" where the direction of operation flows only forward in time. For example:
    - Copy some data over
    - Computation
    - Copy the data back

We can have multiple streams at once in CUDA, and each stream can have it's own timeline. This allows us to overlap operations and make better use of the GPU.

When training a massive language model, it would be silly to spend a ton of time loading all the tokens in and out of the GPU. Streams allow us to move data while also doing computations at alltimes. Streams introduced a software abstraction called "prefetching", which is a way to move data aroung before it is needed. This is a way to hide the latency of moving data around.

# Exmaples.

- Default Stream = Stream 0 = Null Stream

```cpp
//Kernel launches using the null stream(0)
myKernel<<<gridSize,blockSize>>>(args);

//The above statement is equivalent to 
myKernel<<<gridSize,blockSize,0,0>>>(args);
```

- The execution configuration of a global function call is specified by inserting an expression of the form `<<<gridDim,blockDim,Ns,S>>>` where:
    - gridDim(dim3 or int) specifies the number of blocks in the grid
    - blockDim(dim3 or int) specifies the dimension and size of each block
    - Ns(size_t) specifies the number of Bytes in the shared memory that is dynamically allocated per block for this call in addition to the statically allocated memory.
    - S(cudaStream_t) specifies the associated stream, this is an option parameter which defaults to 0

- Stream 1 and 2 are created with different priorities, this mean that they are executed in a certain order at runtime. This essentially gives us more control over the concurrent execution of our kernels.

```cpp

    // Creating streams with priorities;
    int leastPriority, greatestPriority;


    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority,&greatestPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1,cudaStreamNonBlocking,leastPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2,cudaStreamNonBlocking,greatestPriority));
```

## Pinned Memory
- Pinned Memory is a memory that is locked in place and cannot be moved around by the OS. This is useful for when you want to move data to the GPU and do some computation on it. If the OS moves the data around, the GPU will be looking for that data in the wrong place and will return a segfault.

```cpp
//Allocating Pinned Memory
float* h_data;
cudaMallocHost((void**)&h_data,sizeof(float));
```

## Events
- Measuring Kernel Execution time: Events are placed before and after kernel launches to measure the execution time accurately.

- Synchronization between streams: Events can be used to create dependencies between different stream, ensuring one operation starts only after another has completed.

- Overlapping computation and Data transfer: Events can mark the completion of a data transfer, signaling the computation can begin on the data.

```cpp
cudaEvent_t start,stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start,stream);
kernel<<<grid,block,0,stream>>>(args);
cudaEventRecord(stop,stream);

cudaEventSynchronize(stop);
float milliseconds=0;
cudaEventElapsedTime(&milliseconds,start,stop);
```

## Callbacks
- By using callbacks, you can setup a pipeline where the completion of one operation on the GPU  triggers the start of another operation on the CPU, which might then queue more work on for the GPU

```cpp
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status,void* userData){
    printf("GPU Operation completed\n");
    //Trigger next batch of work
}
kernel<<<grid,block,0,stream>>>(args);
cudaStreamAddCallback(stream,myCallback,nullptr,0);

```