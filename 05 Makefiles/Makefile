.PHONY: 1 1_obj 1_obj_exe_run 2 3 clean

GCC = gcc
NVCC = nvcc
CUDA_FLAGS= -arch=sm_86
PWD = pwd

1:
	@$(GCC) -o 1 1.c

1_obj:
	@$(GCC) -c 1.c -o 1.o

1_obj_exe_run:
	@$(GCC) 1.o -o 1
	@./1

2:
	@$(GCC) -o 2 2.c
	@./2

3:
	@$(NVCC) $(CUDA_FLAGS) -o 3_cu 3.cu
	@./3_cu

clean:
	rm *.o