
println("Loading solver")

include("src/WaveSolverCUDA.jl")
using .WaveSolverCUDA

using JLD2, ImageFiltering
import PyPlot as plt

println("Solver loaded")

println("Loading data")
@load "data/marmousi/marmousi20.jld2"
c = 1000 .* c[:, 301:650]
rho = 1000 .* rho[:, 301:650]
c0 = imfilter(c, Kernel.gaussian(10));
c0[1:23,:] .= 1500.

Nx, Ny = size(c)
dx, dy = 20, 20
Nt = 3500
Fs = 500
dt = 1/Fs
t = range(0, (Nt-1)*dt, Nt)
pml_len = 100
pml_coef = 100

a = 1 ./ rho
b = rho .* c .^ 2;

source_num = 11
source_position = zeros(2,source_num)
for i = 1:source_num
    source_position[1,i] = 1
    source_position[2,i] = 1 + (i-1)*35
end
source_vals = zeros(Nt, source_num)
for i = 1:source_num
    source_vals[:,i] = source_ricker(5,0.2,t) * 1e6
end

receiver_num = 350
receiver_position = zeros(2, receiver_num)
for i = 1:receiver_num
    receiver_position[1,i] = 1
    receiver_position[2,i] = (i-1)*1 + 1
end

@load "temp_data/true_data.jld2"
println("Data loaded")

# println("Computing data...")
# CUDA.@time data_true = forward_acoustic_c(c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16)
# @save "temp_data/true_data.jld2" data_true
# println("Done.")

eval_obj(x) = eval_obj_fwi(data_true, x, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16)

eval_grad(x) = eval_grad_fwi(data_true, x, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, cutoff=23)


# println("Computing gradient...")
# # CUDA.@time grad = adjoint_method_c(data_true, c0, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16)
# CUDA.@time val, grad = eval_grad(c0[:])
# println("    Done.")
# @save "temp_data/true_data.jld2" data_true grad



res, vals = lbfgs(c0[:], eval_obj, eval_grad; m=5, iterMax=10, etaBack=0.1, etaWWP=0.9, stepSizeBack=1., stepSizeWWP=1., LinesearchMax=5, wwp_c1=1e-4, wwp_c2=0.9, saveStep=true)