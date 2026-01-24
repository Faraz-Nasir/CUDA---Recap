import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Getting the programID
    row_idx=tl.program_id(axis=0)

    #Computing memory offsets
    row_start_ptr=input_ptr+row_idx*input_row_stride
    out_row_start_ptr=output_ptr+row_idx*output_row_stride

    # Load the row into SRAM
    row=tl.load(row_start_ptr+tl.arange(0,BLOCK_SIZE),mask=tl.arange(0,BLOCK_SIZE)<n_cols,other=-float('inf'))

    #  Compute max for numerical stability
    row_max=tl.max(row,axis=0)

    #Subtracting max
    numerator=tl.exp(row-row_max)

    #Computing sum for row and exponents
    denominator=tl.sum(numerator,axis=0)

    #Normalizing
    softmax_output=numerator/denominator

    #Storing the output
    tl.store(out_row_start_ptr+tl.arange(0,BLOCK_SIZE),softmax_output,mask=tl.arange(0,BLOCK_SIZE)<n_cols)

def triton_softmax(x):
    n_rows,n_cols=x.shape
    output=torch.empty_like(x)

    #Determining the block size
    BLOCK_SIZE=triton.next_power_of_2(n_cols)
    BLOCK_SIZE=min(BLOCK_SIZE,1024)

    #Launching the triton kernel
    grid=(n_rows,)
    softmax_kernel[(grid)](
        output,x,x.stride(0),output.stride(0),n_cols,BLOCK_SIZE
    )
    return output

#Setting up the input tensor
torch.manual_seed(0)
x=torch.randn(256,1024,device='cuda')
torch_result=torch.softmax(x,dim=1)
triton_result=triton_softmax(x)

#Comparing results
max_diff=torch.max(torch.abs(torch_result-triton_result))
print(f"Maximum difference between PyTorch and Triton results: {max_diff:.2e}")

#Checking if results are close
is_close=torch.allclose(torch_result,triton_result,rtol=1e-5,atol=1e-5)
print(f"Results are close: {is_close}")