# Optimizing Matrix Multiplication

- cuBLAS expects matrices to be in column major format so we have to transpose them beforehand
- RowMajor: A[i][j] is stored in A[i*N+j]
- ColumnMajor: A[i][j] is stored in A[j*N+i]

```py
#Row Major
A=[[1,2,3],
   [4,5,6],
   [7,8,9]]

# How its stored in the memory
A=[1,2,3,4,5,6,7,8,9]

#Column Major
B=[[1,4,7],
   [2,5,8],
   [3,6,9]
]

#How its stored in the memory
B=[1,4,7,2,5,8,3,6,9]
```

# Purpose of `prgama #unroll`

- Ideally, you would want a more useful compute per iteration. If you can do 4 math operations inside of 1 per iteration thats good.
- In some contexts, the compiler will actually unroll the loop without explicitly telling it do so.
- You can check the PTX assembly code with `nvcc -ptx v1.cu -o - | less` to see if the compiler has unrolled the loop
- By writing a kernel without unrolling and benchmarking it with a kernel that has unrolling, you cansee if the unrolling is actually beneficial, then check the PTX assemble code to see if the compiler has unrolled the loop, only beneficial if you are not getting the benefits that you wanted and need to investigate further.
- Then quickly benchmark, just take the average time of the kernel and compre it to the unrolled version. If the unrolled version is faster, then the unrolling was beneficial, if not then the unrolling was not beneficial. Always make sure to verify the results so your kernel outputs what it should.

## pragma unroll

### What is Loop Unrolling?

- Loop unrolling is an optimizing technique where the compiler replaces a loop with repeated sequences of code.

```cpp
for(int i=0;i<4;i++){
    a[i]=b[i]+c[i];
}
```

- Every time a `for` loop iterates, teh computer has to do 3 extra steps/tasks  that aren't really computational work
    - Increment the counter
    - Compare the counter to the limit
    - Branch - Jump back to the start of the loop
- This slows the whole process


- **The solution**: If we unroll the loop you do more ***real work*** before checking those conditions again
- **Fully Unrolled Loop**
```cpp
    a[0]=b[0]+c[0];
    a[1]=b[1]+c[1];
    a[2]=b[2]+c[2];
    a[3]=b[3]+c[3];
```

## `#pragma unroll`

-  `#pragma unroll` is a compiler directive. It is a hint that is given to the NVCC compiler saying "Please try to unroll the loop immediately following this line"

- There are 2 ways to do this:
    - `#pragma unroll`   (No Number)
    - `#prgama unroll N` (Number)

    - Using `#pragma unroll`, the compiler attempts to fully unroll the loop. This requires the loop count to be a **compile-time-constant**. i.e i<4. If the compiler doesn't know the exact number of iterations at compile time. It cannot fully unroll it.

    - Using `#pragma roll N`, the compiler partially unrolls the loop by a factor of N.
        - `#pragma unroll 4` inside a loop of 100 iterations.
        - The compiler will generate a block of code that does the work of 4 iterations. and then the loop that block for 25 times(25*4=100)

- Without unrolling, the GPU pauses to check the loop counter, This breaks the flow of math instructions.
- With unrolling, the GPU doesn't have to pass to check these loop counter conditions. It can keep the cores busy with pure math computations.

### How to use `#pragma unroll`

- Place it strictly before the loop that you want to affect.
- For standard full unrolling, the `LOOP_COUNT` must be a time constant.
- The compiler cannot copy-paste code if it doesn't know how many times to paste it while it is compiling.

### Downside of `#prgama unroll`
- Unrolling complex lops often requires more variables to be active at once. If you run out of registers,the data can spill into VRAM which is very slow.   
- If `LOOP_COUNT` is massive like `10000`, unrolling it fully would generate a massive kernel binary and this might not fit completely in the GPU's instruction cache, causing slowdowns.

















# What is occupancy

- Occupancy is defined as the ratio between the number of active warps per SM and the maximum possible number of active warps per SM.

- There are three main limits to keeping more active blocks loaded on a SM: 
    - Register Count,
    - Warp Count,
    - SMEM Capacity

# Assembly Instructions:

## Why would we want to dig into OR write assembly code
- This allows us to understand the operations we are bound by
    - Eg: Warp Divergence, waiting for data to arrive in registers, time expensive operations etc.
- Allows for clock-cycle optimization(closest to bare metal that you can get)
 
## Unrolling Loop logic doesn't always make the code faster.
- Modern GPU's have a specialized hardware feature called a **Loop Instruction Buffer**
- Without Unrolling: When the loop body is tiny. The GPU sees this small loop and loads the instructions into the Loop Buffer once, and then re-executes them a 100 times with 0 instruction fetch latency.

- With unrolling: We force the compiler to copy paste the instruction N number of times, making the resulting code physically huge(in terms of B). It is now too large to fit in the Loop Buffer.The GPU has to constantly fetch new instructions from the L1 cache which adds latency which wasn't present before.

- Loop Unrolling is most effective when you need to hide latency dont use when the math is too simple or when the math is unaffected by the loop counters.

- Also unrolling a large loop can confuse the compiler to force it to use more registers to schedule the instructions.

- If the unrolled kernel uses even a few more registers per thread than the looped version then it might reduce occupancy. Lower occupancy means the GPU has fewer threads to switch when it gets bored leading to lower performance.