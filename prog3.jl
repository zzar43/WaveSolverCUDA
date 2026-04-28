include("src/WaveSolverCUDA.jl")
using .WaveSolverCUDA

using JLD2, ImageFiltering

Nx, Ny = 201, 201
dx, dy = 5, 5
Nt = 1000
Fs = 500
dt = 1/Fs
t = range(0, (Nt-1)*dt, Nt)
pml_len = 30
pml_coef = 100

c = 1000 .* ones(Nx, Ny)
c_true = copy(c)
c_true[85:115,50:80] .= 950
c_true[85:115,120:150] .= 1200
c = imfilter(c_true, Kernel.gaussian(15))
c = (c .- minimum(c)) ./ (maximum(c) - minimum(c)) * (maximum(c_true) - minimum(c_true)) .+ minimum(c_true)

source_num = 9
source_position = zeros(2,source_num)
for i = 1:source_num
    source_position[1,i] = 1
    source_position[2,i] = 1 + (i-1)*25
end
source_vals = zeros(Nt, source_num)
for i = 1:source_num
    # source_vals[:,i] = source_ricker(15,0.1,t) * 1e5
    source_vals[:,i] = source_ricker(10,0.1,t) * 1e5
end

receiver_num = 101
receiver_position = zeros(2, receiver_num)
for i = 1:receiver_num
    receiver_position[1,i] = 201
    receiver_position[2,i] = (i-1)*2 + 1
end

CUDA.@time data_true = forward_acoustic_c(c_true, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16)
@save "temp_data/true_data.jld2" data_true
# @load "temp_data/true_data.jld2"

# proximal method

include("src/prox.jl")

a = [950., 1000., 1200.]
b = [950., 1000., 1200.]
nu = 1e-4
# nu = 0
lamb = 3e3
iterMax = 30

x0 = copy(c)[:]
@load "temp_data/level2/30.jld2"
x0 = copy(x)[:]

eval_obj(x) = eval_obj_fwi(data_true, x, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16) + eval_theta(x, a, b, nu)

eval_grad(x) = eval_grad_fwi(data_true, x, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16)

eval_prox_op(v, lamb) = eval_prox_op(v, a, b, nu, lamb)

res = prox_grad_method(eval_obj, eval_grad, eval_prox_op, x0, lamb; max_iter=iterMax, beta=0.9, ls_time=10, saveStep=true)
# # 
# _, grad = eval_grad(x0)
# @save "temp_data/grad.jld2" grad