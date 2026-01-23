#include<cuda_runtime.h>
#include<stdio.h>

#define NUM_THREADS 1000
#define NUM_BLOCKS 1000

//Atomic functions within this kernel
__global__ void incrementCounterAtomic(int* counter){
    //I am not capturing whatever value it is returning
    atomicAdd(counter,1);
}

int main(){
    int h_counterAtomic=0;
    int *d_counterAtomic;

    //Allocating device memory
    cudaMalloc((void**)&d_counterAtomic,sizeof(int));

    //Copying value from host to device
    cudaMemcpy(d_counterAtomic,&h_counterAtomic,sizeof(int),cudaMemcpyHostToDevice);

    incrementCounterAtomic<<<NUM_BLOCKS,NUM_THREADS>>>(d_counterAtomic);

    //Copying value back from device to host
    cudaMemcpy(&h_counterAtomic,d_counterAtomic,sizeof(int),cudaMemcpyDeviceToHost);

    printf("Atomic Value after kernel execution: %d",h_counterAtomic);

    cudaFree(d_counterAtomic);

    return 0;

}