using CUDA

N = 1024

# Regular host array
host_data = Array{Float32}(undef, N)
fill!(host_data, 1.0f0)

# Pin the host array for async copy
pinned_host_data = CUDA.pin(host_data)

# Allocate device memory
d_data = CUDA.zeros(Float32, N)

# Create CUDA streams
s1 = CuStream()
s2 = CuStream()

# Async host → device
CUDA.@cuda_memcpy_async! d_data pinned_host_data stream=s1

# Kernel
function kernel_add1(a)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(a)
        a[i] += 1.0f0
    end
    return
end

# Async kernel execution
CUDA.@cuda threads=256 blocks=div(N, 256) stream=s2 kernel_add1(d_data)

# Async device → host
CUDA.@cuda_memcpy_async! pinned_host_data d_data stream=s1

# Synchronize
synchronize(s1)
synchronize(s2)

# Retrieve result
result = Array(pinned_host_data)


# One thought:
# We can update for example 100 time steps, and than do one DtoH copy