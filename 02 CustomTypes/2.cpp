#include<iostream>
using namespace std;

typedef struct{
    float x;
    float y;
} Point;

int main(){
    Point p={1.1,2.2};
    printf("Size of Point: %zu\n",sizeof(Point));
}
