using BenchmarkTools, JLD2, CairoMakie, ImageFiltering
include("src/WaveSolverCUDA.jl")
using .WaveSolverCUDA

try
    readdir("data/forward_demo/")
catch err
    mkdir("data/forward_demo")
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

idx_source = 4

println("    Computing data...")
CUDA.@time data_true, U_true = forward_acoustic_c(c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, idx_source=idx_source, recordWaveField=true, saveRatio=1)
println("    Done.")

@save "forward_true.jld2" data_true U_true

# # forward
# println("    Computing forward modeling...")
# CUDA.@time data_forward, U_forward = forward_acoustic_c(c0, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, idx_source=4, recordWaveField=true, saveRatio=1)
# println("    Done.")

# # backward
# # @cuda blocks=cublocks threads=cuthreads WaveSolverCUDA.diff_twice_time_wavefield!(U_forward, dt)
# adjoint_source = data_true - data_forward

# println("    Computing backward...")
# CUDA.@time data_backward, U_backward = forward_acoustic_c(c0, Nx, Ny, Nt, dx, dy, dt, receiver_num, receiver_position, adjoint_source, source_num, source_position, source_vals, pml_len, pml_coef; blockx=16, blocky=16, idx_source=4, recordWaveField=true, saveRatio=1)
# println("    Done.")

# @save "data/forward_backward.jld2" data_true, U_true, U_forward, U_backward

CUDA.@time grad = adjoint_check_wavefield(data_true, c0, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, idx_source, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, saveRatio=1)

@save "grad.jld2" grad