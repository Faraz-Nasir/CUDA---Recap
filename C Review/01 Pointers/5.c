#include<stdio.h>

int main(){
    int arr[]={12,24,36,48,60};
    int* ptr=arr;
    
    for(int i=0;i<5;i++){
        printf("Address: %p, Value: %d\n",ptr,*ptr);
        ptr++;
    }
}