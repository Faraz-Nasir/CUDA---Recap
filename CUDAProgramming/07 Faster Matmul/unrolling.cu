#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<stdlib.h>

#define N 10000000
#define THREADS_PER_BLOCK 256
#define LOOP_COUNT 100
#define WARMPUP_RUNS 5
#define BENCH_RUNS 10

//Kernel without Loop unrolling
__global__ void vectorAddNoUnroll(const float* a,const float* b,float* c,int n){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;

    if(tid<n){
        float sum=0;
        for(int j=0;j<LOOP_COUNT;j++){
            sum+=a[tid]+b[tid];
        }
        c[tid]=sum;
    }
}

//Kernel with loop unrolling using #pragma unroll
__global__ void vectorAddUnroll(const float* a,const float* b,float* c,int n){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;

    if(tid<n){
        float sum=0;
        #pragma unroll
        for(int j=0;j<LOOP_COUNT;j++){
            sum+=a[tid]+b[tid];
        }
        c[tid]=sum;
    }
}

//Function to run the kernel
// The first parameter is a way to pass one function as argument to another.
//void is the return type of the kernel passed as argument
//*kernel is the pointer to that kernel.
float runKernel(void (*kernel)(const float*,const float*,float*,int),float *d_a, float* d_b,float* d_c,int n){
    int numBlocks=(n+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    cudaEvent_t start,stop;
    float milliseconds;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    kernel<<<numBlocks,THREADS_PER_BLOCK>>>(d_a,d_b,d_c,n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main(){
    float *a,*b,*c;
    float *d_a,*d_b,*d_c;
    size_t size=N*sizeof(float);

    //Allocating Host PageLocked Memory
    cudaMallocHost(&a,size);
    cudaMallocHost(&b,size);
    cudaMallocHost((void**)&c,size); //Im just checking if it is necessary to type convert to void** or if I can just pass the address of the pointer to float

    //Initializing Host arrays
    for(int i=0;i<N;i++){
        a[i]=1.0f;
        b[i]=2.0f;
    }

    //Allocating device memory
    cudaMalloc(&d_a,size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);

    //Copying input data from host to device
    cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);
    
    //Warmup Runs
    // #pragma unroll
    for(int i=0;i<WARMPUP_RUNS;i++){
        runKernel(vectorAddNoUnroll,d_a,d_b,d_c,N);
        runKernel(vectorAddUnroll,d_a,d_b,d_c,N);
    }

    //Benchmarking Runs
    float totalTimeNoUnroll=0,totalTimeUnroll=0;
    for(int i=0;i<BENCH_RUNS;i++){
        totalTimeNoUnroll+=runKernel(vectorAddNoUnroll,d_a,d_b,d_c,N);
        totalTimeUnroll+=runKernel(vectorAddUnroll,d_a,d_b,d_c,N);
    }

    //Calculating Average times
    float avgTimeNoUnroll=totalTimeNoUnroll/BENCH_RUNS;
    float avgTimeUnroll=totalTimeUnroll/BENCH_RUNS;


    printf("Average time for kernel without unrolling: %f ms\n",avgTimeNoUnroll);
    printf("Average time for kernel with unrolling: %f ms\n",avgTimeUnroll);

    //Clean Up
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("End of Program\n");
    return 0;
}