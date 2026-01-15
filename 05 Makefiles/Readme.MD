### Makefiles


### CMakeLists.txt
CMake is a tool that **generates Makefiles**. It is a build system generator used to build test and package software. 

### `:=` vs `=` in Makefiles

`=` is used for defining variables. Value of that variable is re-evaluated from the memory everytime the variable is used.

`:=`is used for defining variables too but value of that variable wont be re-evaluated, meaning that if I use 
b=2
a := b, and in the future
b=3

The value of a will still remain 2 regardless of the updates to value of b.

### Purpose of using @ in Makefile

The @ symbol is used to execute commands silently, without printing anything to the console.