#include<cuda_runtime.h>
#include<cublasLt.h>
#include<cuda_fp16.h>
#include<iostream>
#include<vector>
#include<iomanip>

//Run command: nvcc '.\CUDAProgramming\06_CUDA_APIs\02 cuBLASLt\01_LtMatmul.cu' -o 00 -lcublaslt


void cpu_matmul(const float* A,const float* B,float* C,int M,int N,int K){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            float sum=0.0f;
            for(int k=0;k<K;k++){
                sum+=A[i*K+k]*B[k*N+j];
            }
            C[i*N+j]=sum;
        }
    }
}

void print_matrix(const float* matrix, int rows, int cols,const char* name){
    std::cout<<"Matrix"<<name<<":"<<std::endl;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            std::cout<<std::setw(8)<<std::fixed<<std::setprecision(2)<<matrix[i*cols+j]<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}


int main(){
    const int M=4,K=4,N=4;

    float h_A[M*K]={
        1.0f,2.0f,3.0f,4.0f,
        5.0f,6.0f,7.0f,8.0f,
        9.0f,10.0f,11.0f,12.0f,
        13.0f,14.0f,15.0f,16.0f
    };
    float h_B[K*N]={
        1.0f,2.0f,3.0f,4.0f,
        5.0f,6.0f,7.0f,8.0f,
        9.0f,10.0f,11.0f,12.0f,
        17.0f,18.0f,19.0f,20.0f
    };

    float h_C_cpu[M*N]={0};
    float h_C_gpu_fp32[M*N]={0};
    float h_C_gpu_fp16[M*N]={0};

    print_matrix(h_A,M,K,"A");
    print_matrix(h_B,K,N,"B");

    //Allocating device memory for FP32
    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    cudaMalloc(&d_A_fp32,M*K*sizeof(float));
    cudaMalloc(&d_B_fp32,K*N*sizeof(float));
    cudaMalloc(&d_C_fp32,M*N*sizeof(float));
    
    //Allocating device memory for FP16
    float *d_A_fp16, *d_B_fp16, *d_C_fp16;
    cudaMalloc(&d_A_fp16,M*K*sizeof(half));
    cudaMalloc(&d_B_fp16,K*N*sizeof(half));
    cudaMalloc(&d_C_fp16,M*N*sizeof(half));

    //Copying data to device (FP32)
    cudaMemcpy(d_A_fp32,h_A,M*K*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp32,h_B,K*N*sizeof(float),cudaMemcpyHostToDevice);
    
    //Convert and copy data to device(FP16)
    std::vector<half> h_A_half(M*K);
    std::vector<half> h_B_half(K*N);    

    for(int i=0;i<M*K;i++) 
        h_A_half[i]=__float2half(h_A[i]);

    for(int i=0;i<K*N;i++)
        h_B_half[i]=__float2half(h_B[i]);

    cudaMemcpy(d_A_fp16,h_A_half.data(),M*K*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp16,h_B_half.data(),K*N*sizeof(half),cudaMemcpyHostToDevice);

    //Creating a cuBLAS handle
    cublasLtHandle_t handle;
    cublasLtCreate(&handle);

    //Matrix descriptors for FP32
    cublasLtMatrixLayout_t matA_fp32,matB_fp32,matC_fp32;
    cublasLtMatrixLayoutCreate(&matA_fp32,CUDA_R_32F,K,M,K);
    cublasLtMatrixLayoutCreate(&matB_fp32,CUDA_R_32F,N,K,N);
    cublasLtMatrixLayoutCreate(&matC_fp32,CUDA_R_32F,N,M,N);
    
    //Matrix descriptors for FP16
    cublasLtMatrixLayout_t matA_fp16,matB_fp16,matC_fp16;
    cublasLtMatrixLayoutCreate(&matA_fp16,CUDA_R_16F,K,M,K);
    cublasLtMatrixLayoutCreate(&matB_fp16,CUDA_R_16F,N,K,N);
    cublasLtMatrixLayoutCreate(&matC_fp16,CUDA_R_16F,N,M,N);

    //Matrix Multiplication descriptor for FP32
    cublasLtMatmulDesc_t matmulDesc_fp32;
    cublasLtMatmulDescCreate(&matmulDesc_fp32,CUBLAS_COMPUTE_32F,CUDA_R_32F);
    
    //Matrix Multiplication descriptor for FP16
    cublasLtMatmulDesc_t matmulDesc_fp16;
    cublasLtMatmulDescCreate(&matmulDesc_fp16,CUBLAS_COMPUTE_16F,CUDA_R_16F);

    //Matrix Operation for A and B
    cublasOperation_t transa=CUBLAS_OP_N;
    cublasOperation_t transb=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(matmulDesc_fp32,CUBLASLT_MATMUL_DESC_TRANSA,&transa,sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(matmulDesc_fp32,CUBLASLT_MATMUL_DESC_TRANSB,&transb,sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(matmulDesc_fp16,CUBLASLT_MATMUL_DESC_TRANSA,&transa,sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(matmulDesc_fp16,CUBLASLT_MATMUL_DESC_TRANSB,&transb,sizeof(cublasOperation_t));

    //Setting up Alpha and Beta

    const float alpha=1.0f;
    const float beta=0.0f;
    cublasLtMatmul(handle, matmulDesc_fp32, &alpha, d_B_fp32, matB_fp32, d_A_fp32, matA_fp32, &beta, d_C_fp32, matC_fp32, d_C_fp32, matC_fp32, nullptr, nullptr, 0, 0);
    
    const float alpha_half=__float2half(1.0f);
    const float beta_half=__float2half(0.0f);

    cublasLtMatmul(handle, matmulDesc_fp16, &alpha_half, d_B_fp16, matB_fp16, d_A_fp16, matA_fp16, &beta_half, d_C_fp16, matC_fp16, d_C_fp16, matC_fp16, nullptr, nullptr, 0, 0);

    //Copying results back to host
    cudaMemcpy(h_C_gpu_fp32,d_C_fp32,M*N*sizeof(float),cudaMemcpyDeviceToHost);
    std::vector<half> h_C_gpu_fp16_half(M*N);
    cudaMemcpy(h_C_gpu_fp16_half.data(),d_C_fp16,M*N*sizeof(half),cudaMemcpyDeviceToHost);

    //Converting half precision results to single precision
    for(int i=0;i<M*N;i++){
        h_C_gpu_fp16[i]=__half2float(h_C_gpu_fp16_half[i]);
    }

    //Performing CPU matrix multiplication
    cpu_matmul(h_A,h_B,h_C_cpu,M,N,K);

    print_matrix(h_C_cpu,M,N,"C (CPU)");
    print_matrix(h_C_gpu_fp32,M,N,"C (GPU FP32)");
    print_matrix(h_C_gpu_fp16,M,N,"C (GPU FP16)");

    cublasLtMatrixLayoutDestroy(matA_fp32);
    cublasLtMatrixLayoutDestroy(matB_fp32);
    cublasLtMatrixLayoutDestroy(matC_fp32);
    cublasLtMatrixLayoutDestroy(matA_fp16);
    cublasLtMatrixLayoutDestroy(matB_fp16);
    cublasLtMatrixLayoutDestroy(matC_fp16);
    cublasLtMatmulDescDestroy(matmulDesc_fp32);
    cublasLtMatmulDescDestroy(matmulDesc_fp16);
    cublasLtDestroy(handle);
    cudaFree(d_A_fp32);
    cudaFree(d_B_fp32);
    cudaFree(d_C_fp32);
    cudaFree(d_A_fp16);
    cudaFree(d_B_fp16);
    cudaFree(d_C_fp16);

    return 0;
}