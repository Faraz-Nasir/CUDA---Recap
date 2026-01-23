#include<cuda_runtime.h>
#include<stdio.h>

// Remember that in C -> technically just means *.
// Remember that every piece of data that the GPU touches has to be located on GPU memory only. GPU cannot read anything that resides on CPU memory.

//So Summarising, regardless of host or device kernel/functions. if im performing an operation then the location that im performing it on must reside on their memory. This can be CPU or GPU 

// Dont pass cpu memory location to gpu kernel and expect that it will work because gpu cannot access cpu memory and cpu cannot access gpu memory.


//This whole code can be rewritten with the help of AtomicAdd() functions, so would not even need mutexes because on prinicple Atomic functions inherently lock the memory at which they are performing operations to ensure mutual Exclusion.
typedef struct{
    int* lock;
}Mutex;

//Initializing the mutex(This is a host function used to allocate memory on the gpu for the lock)
__host__ void initMutex(Mutex *m){
    cudaMalloc((void**)&m->lock,sizeof(int));
    int inital=0;
    cudaMemcpy(m->lock,&inital,sizeof(int),cudaMemcpyHostToDevice);
}

//Acquiring the Mutex
__device__ void lock(Mutex* m){
    while(atomicCAS(m->lock,0,1)!=0); //Locks the mutex
}

__device__ void unlock(Mutex* m){
    atomicExch(m->lock,0);//Unlocks the Mutex
}

//Kernel to demonstrate mutex usage
__global__ void mutexKernel(int *counter,Mutex *m){
    lock(m);
    //Critical Section
    int old= *counter;
    *counter=old+1; 
    unlock(m);
}

int main(){
    Mutex h_m;  //Mutex allocated on the CPU 
    initMutex(&h_m); //Pasing the CPU Mutex to declare memory on the gpu for the lock(which is its object)

    Mutex* d_m;
    cudaMalloc((void**)&d_m,sizeof(Mutex));
    cudaMemcpy(d_m,&h_m,sizeof(Mutex),cudaMemcpyHostToDevice);

    int* d_counter;
    cudaMalloc((void**)&d_counter,sizeof(int));
    int initial=0;
    cudaMemcpy(d_counter,&initial,sizeof(int),cudaMemcpyHostToDevice);

    //Launching the Kernel with multiple threads
    // 1 Block, 1000 threads in that 1 block
    mutexKernel<<<1,1000>>>(d_counter,d_m);

    int result;
    cudaMemcpy(&result,d_counter,sizeof(int),cudaMemcpyDeviceToHost);
    

    printf("Counter Value: %d\n",result);

    cudaFree(h_m.lock);
    cudaFree(d_m);
    cudaFree(d_counter);

    return 0;
}
