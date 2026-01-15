#include<stdio.h>

struct Point{
    float x;
    float y;
};

int main(){
    struct Point p={1.1,2.2};
    printf("Size of Point object: %zu\n",sizeof(struct Point));
}