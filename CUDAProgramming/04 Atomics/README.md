# What are Atomic Operations

- by **atomic** we are referring to the indivisibility concept in physics where a thing cannot be broken down further

- An **atomic operation**  ensures that a paticular operation on a memory location is completed entirely by one thread before another thread can access or modify the same memory location. This prevents race conditions.

- Since we limit the amount of work done on a single piece of memory per unit time throughout an atomic operation, we lose slightly to speed.It is hardware guaranteed to be memory safe at a cost of speed.

## Integer Atomic Operations

-   ```cpp
       atomicAdd(int* address,int val) 
    ```
    Atomically adds value to the value at the address and returns the old value
-   ```cpp
       atomicSub(int* address,int val) 
    ```
    Atomically subtracts value to the value at the address and returns the old value
-   ```cpp
       atomicExch(int* address,int val) 
    ```
    Atomically exchanges value to the value at the address and returns the old value
-   ```cpp
       atomicMax(int* address,int val) 
    ```
    Atomically sets the value at the address and returns the old value

# Floating-Point Atomic Operations

-  `atomicAdd(float* address,float val)`: Atomically addsval to the value at address and returns the old value.


# Basic Principle

- Modern GPU's have special hardware instructions to perform these operations efficiently. They use techniques like Compare-And-Swap(CAS) at the hardware level.

- Think of atomics as a very fast, hardware-level mutex operation. It's as if each atomic operation performs this:
    - lock(memory_location)
    - old_value= *memory_location
    - *memory_location = old_value + increment
    - unlock(memory_location)
    - return old_value

```cpp

    __device__ int softwareAtomicAdd(int* address,int increment){
        __shared__ int lock;
        int old;

        if(threadIdx.x==0) lock=0;
        __syncthreads();

        while(atomicCAS(&lock,0,1)!=0); //Acquire's lock

        old=*address;
        *address=old+increment;

        __threadfence(); //Ensures that write is visible ot other threads

        atomicExch(&lock,0); //This releases the lock
        return old;
    }
```

- Mutual Exclusion
    - This implies a reciprocal or shared relationship between entities.
    - Suggests that the exclusion applies equally to all the parties involved.
    - Refers to the act of keeping something out or preventing access.
    - In this context, it means preventing simultaneous access to a resource.