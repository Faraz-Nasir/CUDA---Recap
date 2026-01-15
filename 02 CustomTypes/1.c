#include <stdio.h>
#include <stdlib.h>

//size_t, is a datatype used for memory allocation, used to represent the size of objects in Bytes(B)
//Guaranteed to be big enough to contain the size of the biggest object of the host system
// %z -> size_t
// %u -> unsigned int

int main(){
    int arr[]={12,24,36,48,60};

    size_t size=sizeof(arr)/sizeof(arr[0]);
    printf("Size of the whole Array: %zu\n",size);

    return 0;
}