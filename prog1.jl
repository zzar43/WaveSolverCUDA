include("src/solver.jl")
include("src/forward.jl")

using BenchmarkTools, CairoMakie

Nx = 801
Ny = 801
dx = 5
dy = 5
Nt = 2000
Fs = 500
dt = 1/Fs
t = range(0, (Nt-1)*dt, Nt)
pml_len = 100
pml_coef = 50

source_num = 3
source_position = zeros(2,source_num)
for i = 1:source_num
    source_position[1,i] = 201
    source_position[2,i] = 401 + 100*(i-1)
end
source_vals = zeros(Nt, source_num)
for i = 1:source_num
    source_vals[:,i] = source_ricker_int(15,0.2,t)
end

receiver_num = 201
receiver_position = zeros(2, receiver_num)
for i = 1:receiver_num
    receiver_position[1,i] = 1
    receiver_position[2,i] = 1 + (i-1) * 4
end

rho = 1 .* ones(myReal, Nx, Ny)
c = 1000 .* ones(myReal, Nx, Ny)
a = 1 ./ rho
b = rho .* c .^ 2

# @time U1 = acoustic_solver_pml(a, b, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, pml_len, pml_coef; blockx=16, blocky=16, recordWaveField=true)

@time data = forward_problem_acoustic(a, b, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, idx_source=0)

data0 = data
fig = Figure(size = (800, 500))
val = 0.1 * maximum(data)
ax = Axis(fig[1, 1], aspect = 1, yreversed=true)
hm = heatmap!(data0', colormap=:gray1, colorrange=(-val, val))
Colorbar(fig[1, 2], hm)
fig

data0 = data[:,:,3]
fig = Figure(size = (800, 500))
val = 0.1 * maximum(data)
ax = Axis(fig[1, 1], aspect = 1, yreversed=true)
hm = heatmap!(data0', colormap=:gray1, colorrange=(-val, val))
Colorbar(fig[1, 2], hm)
fig


u = U[:,:,1700]
fig = Figure(size = (800, 500))
val = 0.01 * maximum(U)
ax = Axis(fig[1, 1], aspect = 1, )
hm = heatmap!(u, colormap=:gray1, colorrange=(-val, val))
Colorbar(fig[1, 2], hm)
fig

u = U1[:,:,1800]
fig = Figure(size = (800, 500))
val = 0.01 * maximum(U)
ax = Axis(fig[1, 1], aspect = 1, )
hm = heatmap!(u, colormap=:gray1, colorrange=(-val, val))
Colorbar(fig[1, 2], hm)
fig





maximum(data)