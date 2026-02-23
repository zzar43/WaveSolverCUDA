# save forward and backward wavefield with marmousi model

using BenchmarkTools, JLD2, CairoMakie, ImageFiltering

# include("src/adjoint.jl")
# include("src/forward.jl")
include("src/WaveSolverCUDA.jl")
using .WaveSolverCUDA

try
    readdir("data/forward_backward_demo/")
catch err
    mkdir("data/forward_backward_demo")
end

@load "data/marmousi/marmousi20.jld2"
c = 1000 .* c
rho = 1000 .* rho
c0 = imfilter(c, Kernel.gaussian(10));
c0[1:23,:] .= 1500.

Nx, Ny = size(c)
dx, dy = 20, 20
Nt = 8000
Fs = 500
dt = 1/Fs
t = range(0, (Nt-1)*dt, Nt)
pml_len = 100
pml_coef = 200

a = 1 ./ rho
b = rho .* c .^ 2;

source_num = 15
source_position = zeros(2,source_num)
for i = 1:source_num
    source_position[1,i] = 5
    source_position[2,i] = 51 + (i-1)*51
end
source_vals = zeros(Nt, source_num)
for i = 1:source_num
    source_vals[:,i] = source_ricker_int(5,0.2,t) * 1e6
end

receiver_num = 425 * 2
receiver_position = zeros(2, receiver_num)
for i = 1:receiver_num
    receiver_position[1,i] = 1
    receiver_position[2,i] = (i-1)*1 + 1
end

println("    Computing data...")
CUDA.@time data = forward_acoustic_c(c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, idx_source=4, recordWaveField=false, saveRatio=1)
println("    Done.")

# parameters
Nx_pml = Nx + 2*pml_len
Ny_pml = Ny + 2*pml_len
source_position_pml = source_position .+ pml_len
receiver_position_pml = receiver_position .+ pml_len

# to device
source_vals_device, source_position_x, source_position_y, receiver_position_x, receiver_position_y = to_device_source_receiver(source_position_pml, source_vals, source_num, Nt, receiver_position_pml, receiver_num)

# CUDA parameters
cuthreads, cublocks = init_CUDA_grid_parameters(Nx_pml, Ny_pml; blockx, blocky)
cuthreads_source, cublocks_source = init_CUDA_source_parameters(source_num; blockx)
cuthreads_receiver, cublocks_receiver = init_CUDA_source_parameters(receiver_num; blockx)
