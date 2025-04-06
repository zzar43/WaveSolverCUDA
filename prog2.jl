include("src/adjoint.jl")
include("src/forward.jl")

# DEMO 1: square domain

Nx = 401
Ny = 401
dx = 10
dy = 10
Nt = 2000
Fs = 300
dt = 1/Fs
t = range(0, (Nt-1)*dt, Nt)
pml_len = 100
pml_coef = 100

rho = 1 .* ones(myReal, Nx, Ny)
c = 1000 .* ones(myReal, Nx, Ny)
c[150:250,150:250] .= 1200
c0 = 1000 .* ones(myReal, Nx, Ny)

source_num = 9
source_position = zeros(2,source_num)
for i = 1:source_num
    source_position[1,i] = 1
    source_position[2,i] = 1 + (i-1)*51
end
source_vals = zeros(Nt, source_num)
for i = 1:source_num
    source_vals[:,i] = source_ricker_int(8,0.2,t) * 1e6
end

receiver_num = 401
receiver_position = zeros(2, receiver_num)
for i = 1:receiver_num
    receiver_position[1,i] = 401
    receiver_position[2,i] = (i-1)*1 + 1
end

println("    Computing data...")
CUDA.@time data = forward_acoustic_c(c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, idx_source=0, recordWaveField=false, saveRatio=1)
println("    Done.")

println("    Computing adjoint...")
CUDA.@profile gg = adjoint_c(data, c0, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16)
println("    Done")