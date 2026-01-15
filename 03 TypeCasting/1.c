#include<stdio.h>

int main(){
    float f=69.69;
    int i=(int) f;
    printf("%d\n",i);  //Decimal part will be truncated

    char c=(char) f;
    printf("%c\n",c);  //Translation via ASCII
}