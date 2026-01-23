#include<cuda_runtime.h>
#include<stdio.h>

#define CHECK_CUDA_ERROR(val) check((val),#val,__FILE__,__LINE__)

template<typename T>
void check(T err,const char* const func,const char* const file,const int line){
    if(err!=cudaSuccess){
        fprintf(stderr,"CUDA error at %s:%d code=%d(%s) \"%s\"\n",file,line,static_cast<unsigned int>(err),cudaGetErrorString(err),func);
        exit(EXIT_FAILURE);
    }
}

__global__ void vectorAdd(const float *A,const float *B,float *C,int numElements){
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<numElements){
        C[i]=A[i]+B[i];
    }
}

int main(void){
    int numElements=50000;
    size_t size=numElements*sizeof(float);
    float *h_A,*h_B,*h_C;
    float *d_A,*d_B,*d_C;
    cudaStream_t stream1,stream2;

    //Allocating Host Memory
    h_A=(float*)malloc(size);
    h_B=(float*)malloc(size);
    h_C=(float*)malloc(size);

    //Initialing these arrays
    for(int i=0;i<numElements;i++){
        h_A[i]=rand()/(float)RAND_MAX;
        h_B[i]=rand()/(float)RAND_MAX;
    }

    //Allocating Device Memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A,size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B,size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C,size));

    //Creating streams
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

    //Copying data to device asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A,h_A,numElements,cudaMemcpyHostToDevice,stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B,h_B,numElements,cudaMemcpyHostToDevice,stream2));

    //Synchronizing here to make sure that all the data is copied before starting computations
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    //Launching the Kernel
    int threadsPerBlock=256;
    int blocksPerGrid=(numElements+threadsPerBlock-1)/threadsPerBlock;
    vectorAdd<<<blocksPerGrid,threadsPerBlock,0,stream1>>>(d_A,d_B,d_C,numElements);

    printf("Execution Completed\n");
    //Copying the Results back asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C,d_C,size,cudaMemcpyDeviceToHost,stream1));

    //Synchronizing Streams
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    for(int i=0;i<numElements;i++){
        printf("%f + %f = %f\n",*h_A,*h_B,*h_C);
        h_A++;
        h_B++;
        h_C++;
    }

    printf("Streams Destroyed\n");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;



}