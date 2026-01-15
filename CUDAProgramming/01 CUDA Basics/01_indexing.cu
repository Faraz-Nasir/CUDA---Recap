#include <stdio.h>


/*
    For this example,
    Consider Blocks as Buildings
             Threads as People
    
    1. First part is to find the building number out of a 3d landscape of buildings
    // Multiplication is used to skip over that line of buildings in that axis.
        int block_id=blockIdx.x+
        blockIdx.y*gridDim.x+  
        blockIdx.z*gridDim.x*gridDim.y
    
    2. Second Part is to find the person in that building once the building number is known.
        We can repeat the same analogy to find the person.

        int thread_offset=threadIdx.x+
            threadIdx.y*blockDim.x+
            threadIdx.z*blockDim.x*blockDim.y


*/
__global__ void whoami(void){

    //Blocks are buildings
    //Threads are people living in  those buildings
    int block_id=
        blockIdx.x+
        blockIdx.y*gridDim.x+
        blockIdx.z*gridDim.x*gridDim.y;

    int block_offset=
        block_id*
        blockDim.x*blockDim.y*blockDim.z;

    int thread_offset=
        threadIdx.x+
        threadIdx.y*blockDim.x+
        threadIdx.z*blockDim.x*blockDim.z;

    int id=block_offset+thread_offset;

    printf("%04d | Block(%d %d %d) = %3d | Thread (%d %d %d) = %3d\n",
    id,
    blockIdx.x,blockIdx.y,blockIdx.z,block_id,
    threadIdx.x,threadIdx.y,threadIdx.z,thread_offset
    );
}

int main(){
    const int b_x=2,b_y=3,b_z=4;
    const int t_x=4,t_y=4,t_z=4;

    int threadsPerBlock=t_x*t_y*t_z;
    int blocksPerGrid=b_x*b_y*b_z;

    printf("%d Threads/Block\n",threadsPerBlock);
    printf("%d Blocks/Grid\n",blocksPerGrid);

    whoami<<<blocksPerGrid,threadsPerBlock>>>();
    cudaDeviceSynchronize();

}