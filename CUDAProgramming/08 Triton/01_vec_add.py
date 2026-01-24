import torch
import triton
import triton.language as tl
import time

@triton.jit
def add_kernel(
    x_ptr,  # Pointer to the first input vector.
    y_ptr,  # Pointer to the second input vector.
    output_ptr,  # Pointer to the output vector.
    n_elements,  #Size of the vector.
    BLOCK_SIZE: tl.constexpr #Number of elements each program should process.
): 
    # There are going to be multiple programs processing different data. We identify which program.
    pid=tl.program_id(axis=0) # We use a 1D launch grid so acis is 0.
                              # This is the same as cuda blockIdx.x

    # This program will process inputs that are offset from the initial data
    # For instance if you had a vector of length 256 and block_size of 64
    # The program would access the elements [0:64, 64:128,128:192,192:256]
    # Note that offsets is a list of pointers:
    block_start=pid*BLOCK_SIZE
    offsets=block_start+tl.arange(0,BLOCK_SIZE) #Which element in the block

    # Creating a mask to guard memory operations against out-of-bounds accesses.
    mask=offsets<n_elements

    # All of this is the same as 
    # __global__ void add_kernel(float* x, float* y, float* output,int n){
    #     int idx=blockIdx.x*blockDim.x+threadIdx.x;
    #     if(idx<n){
    #         output[idx]=x[idx]+y[idx]
    #     }
    # }

    # Loading x and y from the DRAM, masking out any extra elements in case the input is not a multiple of the blockSize

    x=tl.load(x_ptr+offsets,mask=mask)
    y=tl.load(y_ptr+offsets,mask=mask)
    output=x+y

    #Writing x and y back to the DRAM
    tl.store(output_ptr+offsets,output,mask=mask)

def add(x:torch.Tensor,y:torch.Tensor):
    output=torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements=output.numel()
    # The SPMD launch grid denotes the number of kernel instances that will run in parallel.
    # It is analogous to CUDA launch grids. It can either be a Tuple[int] or Callable(metaparameters) -> Tuple[int].
    # In this case,we use a 1D grid where the size is the number of blocks.
    grid=lambda meta:(triton.cdiv(n_elements,meta['BLOCK_SIZE']),)
    # Ceiling division(n_elements/BLOCK_SIZE)

    # Each torch.tensor object is implicitly converted into a pointer to its first element.
    # triton.jit'ed functions can be indexed with a launch grid to obtain a callable GPU Kernel.
    # Don't forget to pass meta-parameters as keyword arguments.
    add_kernel[(grid)](x,y,output,n_elements,BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still running asynchronously at this point.
    return output

torch.manual_seed(0)
size=2**25
x=torch.rand(size,device='cuda')
y=torch.rand(size,device='cuda')



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], # Arg name for X axis for the plot
        x_vals=[2**i for i in range(12,28,1)],
        x_log=True,
        line_arg='provider', #Arg name whose value corresponds to a diff line in the plot
        line_vals=['triton','torch'], #Possible vals for line_arg
        line_names=['Triton','Torch'],
        styles=[('blue','-'),('green','-')], #Line styles
        ylabel="GB/s",
        plot_name='vector-add-performance', #Name of the plot also used to save the plot as a file
        args={} # Values for function arguments not in x_names and y_name
    ))

def Benchmark(size,provider):
    x=torch.rand(size,device='cuda',dtype=torch.float32)
    y=torch.rand(size,device='cuda',dtype=torch.float32)
    quantiles=[0.5,0.2,0.8]
    if provider=='torch':
        ms,min_ms,max_ms=triton.testing.do_bench(lambda:x+y,quantiles=quantiles)
    elif provider=='triton':
        ms,min_ms,max_ms=triton.testing.do_bench(lambda:add(x,y),quantiles=quantiles)
    gbps=lambda ms:3*x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms),gbps(max_ms),gbps(min_ms)

Benchmark.run(print_data=True,show_plots=True)