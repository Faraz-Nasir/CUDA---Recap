#include<cuda_runtime.h>
#include<stdio.h>
#include<iostream>

__global__ void kernel1(float* data,int n){
    int idx=threadIdx.x+blockDim.x*blockIdx.x;
    if(idx<n){
        data[idx]*=2.0f;
    }
}

__global__ void kernel2(float* data,int N){
    int idx=threadIdx.x+blockDim.x*blockIdx.x;
    if(idx<N){
        data[idx]+=1.0f;
    }
}

void CUDART_CB myStreamCallback(cudaStream_t stream,cudaError_t status,void* userData){
    printf("Stream Callback: Operation Completed\n");
}

int main(){
    const int N=1000000;
    size_t size=N*sizeof(float);
    float* h_data,*d_data;
    cudaStream_t stream1,stream2;
    cudaEvent_t event;

    std::cout<<"Event created: "<<event<<std::endl; //This will print a garbage value because it has not been assigned yet

    //Allocating Host and Device memory
    //Using cudaMallocHost to get pageLocked Memory on the CPU so it cant be moved around by the OS. Malloc memory on the other hand can be moved around by the OS. and there can be a case where if the memory is shifted then code is pointing to an address with possible invalid/garbage value. 
    //Also while using malloc() because it is pageable memory the GPU cant talk to it. So while using something cudaMemcpy , the function firsts transfers pageable memory into secret buffer and then from there it shifts to the GPU VRAM which slows down the program.
    cudaMallocHost((void**)&h_data,size);  // This is non-pageable locked CPU memory
    cudaMalloc(&d_data,size);

    //Initializing data
    for(int i=0;i<N;i++){
        h_data[i]=static_cast<float>(i);
    }

    //Creating Steams with different priorities
    int leastPriority,greatestPriority;
    cudaDeviceGetStreamPriorityRange(&leastPriority,&greatestPriority);
    cudaStreamCreateWithPriority(&stream1,cudaStreamNonBlocking,leastPriority);
    cudaStreamCreateWithPriority(&stream2,cudaStreamNonBlocking,greatestPriority);

    //Creating an Event
    cudaEventCreate(&event);
    
    std::cout<<"Event Assigned: "<<event<<std::endl;
    
    //Copying data from CPU mem to GPU VRAM Async
    cudaMemcpyAsync(d_data,h_data,size,cudaMemcpyHostToDevice,stream1);
    kernel1<<<(N+255)/256,256,0,stream1>>>(d_data,N);
    cudaEventRecord(event,stream1);

    //Making stream2 wait for the event to trigger
    cudaStreamWaitEvent(stream2,event,0);
    kernel2<<<(N+255)/256,256,0,stream2>>>(d_data,N);
    cudaStreamAddCallback(stream2,myStreamCallback,NULL,0);

    //Copying data back
    cudaMemcpyAsync(h_data,d_data,size,cudaMemcpyDeviceToHost,stream2);

    //Synchronizing Streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    for(int i=0;i<N;i++){
        fprintf(stdout,"%f\n",h_data[i]);
    }
    
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFreeHost(h_data);
    cudaFree(d_data);
    cudaEventDestroy(event);

}