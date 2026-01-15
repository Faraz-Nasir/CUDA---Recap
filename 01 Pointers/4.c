#include<stdio.h>
#include<stdlib.h>

int main(){
    int* ptr=NULL;
    printf("1. Initial Pointer Value: %p\n",(void*)ptr);

    if(ptr==NULL){
        printf("2. Pointer is NULL, cannot dereference\n");
    }

    //Allocating Memory
    ptr=malloc(sizeof(int));
    if(ptr==NULL){
        printf("3. Dynamic Memory Allocation failed\n");
        return 1;
    }
    printf("4. After allocation, ptr value: %p\n",(void*)ptr);

    *ptr=42;
    printf("5. Value stored at pointer: %d\n",*ptr);

    //CleanUp
    free(ptr);
    printf("6. Address pointed by pointer: %p\n",ptr);
    ptr=NULL;
    printf("7. Address pointed by pointer: %p\n",ptr);

    if(ptr==NULL){
        printf("8. Pointer is NULL");
    }

    return 0;
}