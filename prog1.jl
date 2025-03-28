include("src/solver.jl")

using BenchmarkTools, CairoMakie

Nx = 801
Ny = 801
dx = 5
dy = 5
Nt = 1800
Fs = 1000
dt = 1/Fs
t = range(0, (Nt-1)*dt, Nt)
pml_len = 50
pml_coef = 20

source_num = 3
source_position = zeros(source_num, 2)
for i = 1:source_num
    source_position[i,1] = 201 + 400 * (i-1)
    source_position[i,2] = 201 + 400 * (i-1)
end
source_vals = zeros(Nt, source_num)
for i = 1:source_num
    source_vals[:,i] = source_ricker_int(15,0.2,t)
end
source_vals = source_vals'
(source_num, 2) == size(source_position)
check_source_vals!(source_vals, Nt, source_num)
check_source_position!(source_position, source_num)

source_position
source_vals
receiver_num = 12
receiver_position = 2 .* source_position

rho = 1 .* ones(myReal, Nx, Ny)
c = 1000 .* ones(myReal, Nx, Ny)
a = 1 ./ rho
b = rho .* c .^ 2

@benchmark U = acoustic_solver_pml(a, b, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, pml_len, pml_coef; blockx=16, blocky=16, recordWaveField=true)
println("done")

@benchmark U = acoustic_solver_pml1(a, b, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, pml_len, pml_coef; blockx=16, blocky=16, recordWaveField=true)
println("done")


# u = U[1800,:,:]
fig = Figure(size = (800, 500))
val = 0.2 * maximum(u)
ax = Axis(fig[1, 1], aspect = 1, )
hm = heatmap!(u, colormap=:gray1, colorrange=(-val, val))
# Colorbar(fig[1, 2], hm)
fig

