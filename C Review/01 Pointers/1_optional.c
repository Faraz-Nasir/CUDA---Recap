#include <stdio.h>
#include <stdlib.h>

struct Person{
    int age;
    char* name;
};

int main(){
    struct Person person1;
    person1.age=21;
    person1.name="Faraz";

    printf("age: %d\t name: %s\n",person1.age,person1.name);
    
    struct Person* person2=malloc(sizeof(struct Person));
    
    person2->age=29;
    person2->name="Not Faraz";
    
    printf("age: %d\t name: %s\n",person2->age,person2->name);
    return 0;
}