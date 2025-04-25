include("base.jl")
include("utils.jl")
include("misc.jl")
include("kernels.jl")

# The adjoint method for one parameter case (velocity).

function diff_twice_time_wavefield!(U, dt)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i > 1 && i < size(U,1) && j > 1 && j < size(U,2)

        temp0 = U[i, j, 1]
        temp1 = U[i, j, 2]
        for idx_time in 2:size(U,3)-1
            temp1 = U[i, j, idx_time]
            U[i, j, idx_time] = (U[i, j, idx_time+1] - 2*U[i, j, idx_time] + temp0) / dt^2
            temp0 = temp1
        end

    end

    return nothing
end

function time_int_wavefield_c!(U, c, grad, dt)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i > 1 && i < size(U,1) && j > 1 && j < size(U,2)
        for idx_time = 2:size(U,3)-1
            grad[i,j] += U[i,j,idx_time]
        end
        grad[i,j] = grad[i,j] * dt
        grad[i,j] = -2 * grad[i,j] / c[i,j]^3
    end
    
    return nothing
end

function adj_forward_modeling!(U, u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml, source_position_x, source_position_y, source_vals_device, idx_source, sigma_x_half, sigma_y_half, a_x, a_y, receiver_position_x, receiver_position_y, forward_data, receiver_num, saveRatio, pml_len, cublocks, cublocks_source, cublocks_receiver, cuthreads, cuthreads_source, cuthreads_receiver)

    # forward
    for idx_time = 1:Nt

        @cuda blocks=cublocks threads=cuthreads update_pressure_pml_4th!(u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml)

        @cuda blocks=cublocks_source threads=cuthreads_source update_source_idx!(u, source_position_x, source_position_y, source_vals_device, idx_source, idx_time, dt)

        @cuda blocks=cublocks threads=cuthreads update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx_pml, Ny_pml)

        @cuda blocks=cublocks threads=cuthreads update_velocity_pml_4th!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx_pml, Ny_pml)

        @cuda blocks=cublocks_receiver threads=cuthreads_receiver record_wavefield!(u, receiver_position_x, receiver_position_y, forward_data, receiver_num, idx_time)

        # U[:,:,idx_time] = u[pml_len+1:end-pml_len, pml_len+1:end-pml_len]
        if idx_time % saveRatio == 0
            U[:, :, Int(ceil(idx_time/saveRatio))] = u[pml_len+1:end-pml_len, pml_len+1:end-pml_len]
        end

    end

    return nothing
end

function adj_backward_modeling!(U, u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml, receiver_position_x, receiver_position_y, adjoint_source, receiver_num, Nt, sigma_x_half, sigma_y_half, a_x, a_y, saveRatio, pml_len, cublocks, cublocks_receiver, cuthreads, cuthreads_receiver)

    for idx_time = 1:Nt

        @cuda blocks=cublocks threads=cuthreads update_pressure_pml_4th!(u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml)

        @cuda blocks=cublocks_receiver threads=cuthreads_receiver update_source!(u, receiver_position_x, receiver_position_y, adjoint_source, receiver_num, Nt-idx_time+1, dt)
    
        @cuda blocks=cublocks threads=cuthreads update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx_pml, Ny_pml)
    
        @cuda blocks=cublocks threads=cuthreads update_velocity_pml_4th!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx_pml, Ny_pml)

        # U[:,:,Nt-idx_time+1] .*= u[pml_len+1:Nx+pml_len, pml_len+1:Ny+pml_len]
        if idx_time % saveRatio == 0
            U[:,:,Int(ceil((Nt-idx_time+1)/saveRatio))] .*= u[pml_len+1:Nx+pml_len, pml_len+1:Ny+pml_len]
        end
        
    end
    return nothing
end

function adjoint_single_source_c(data, c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, idx_source, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, saveRatio=1)

    @assert (Nt, receiver_num) == size(data) "The size of data should be (Nt, receiver_num)"

    source_position = check_source_position(source_position, source_num)
    source_vals = check_source_vals(source_vals, Nt, source_num)
    receiver_position = check_receiver_position(receiver_position, receiver_num)

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

    # initialize
    rho = 1000 .* ones(size(c))
    a = -1 ./ rho
    b = -1 .* rho .* c.^2
    c = CuArray{myReal}(c)

    u, vx, vy, wx, wy = init_grid_pml(myReal, Nx, Ny, pml_len)
    a_x, a_y, b_pml = init_parameters_pml(myReal, a, b, pml_len)
    sigma_x, sigma_y, sigma_x_half, sigma_y_half = build_sigma(myReal, Nx, Ny, pml_len, pml_coef)
    forward_data = CUDA.zeros(myReal, Nt, receiver_num)
    # U = CUDA.zeros(myReal, Nx, Ny, Nt)
    U = CUDA.zeros(myReal, Nx, Ny, Int(floor(Nt/saveRatio)))
    grad = CUDA.zeros(myReal, Nx, Ny)

    # adj_forward_modeling!(U, u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml, source_position_x, source_position_y, source_vals_device, idx_source, sigma_x_half, sigma_y_half, a_x, a_y, receiver_position_x, receiver_position_y, forward_data, receiver_num, saveRatio, pml_len, cublocks, cublocks_source, cublocks_receiver, cuthreads, cuthreads_source, cuthreads_receiver)
    # forward
    for idx_time = 1:Nt

        @cuda blocks=cublocks threads=cuthreads update_pressure_pml_4th!(u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml)

        @cuda blocks=cublocks_source threads=cuthreads_source update_source_idx!(u, source_position_x, source_position_y, source_vals_device, idx_source, idx_time, dt)

        @cuda blocks=cublocks threads=cuthreads update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx_pml, Ny_pml)

        @cuda blocks=cublocks threads=cuthreads update_velocity_pml_4th!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx_pml, Ny_pml)

        @cuda blocks=cublocks_receiver threads=cuthreads_receiver record_wavefield!(u, receiver_position_x, receiver_position_y, forward_data, receiver_num, idx_time)

        # U[:,:,idx_time] = u[pml_len+1:end-pml_len, pml_len+1:end-pml_len]
        if idx_time % saveRatio == 0
            U[:, :, Int(ceil(idx_time/saveRatio))] = u[pml_len+1:end-pml_len, pml_len+1:end-pml_len]
        end

    end

    # time differential
    @cuda blocks=cublocks threads=cuthreads diff_twice_time_wavefield!(U, dt)
    # backward
    u, vx, vy, wx, wy = init_grid_pml(myReal, Nx, Ny, pml_len)
    adjoint_source = CuArray{myReal}(data) - forward_data

    # adj_backward_modeling!(U, u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml, receiver_position_x, receiver_position_y, adjoint_source, receiver_num, Nt, sigma_x_half, sigma_y_half, a_x, a_y, saveRatio, pml_len, cublocks, cublocks_receiver, cuthreads, cuthreads_receiver)
    for idx_time = 1:Nt

        @cuda blocks=cublocks threads=cuthreads update_pressure_pml_4th!(u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml)

        @cuda blocks=cublocks_receiver threads=cuthreads_receiver update_source!(u, receiver_position_x, receiver_position_y, adjoint_source, receiver_num, Nt-idx_time+1, dt)
    
        @cuda blocks=cublocks threads=cuthreads update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx_pml, Ny_pml)
    
        @cuda blocks=cublocks threads=cuthreads update_velocity_pml_4th!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx_pml, Ny_pml)

        # U[:,:,Nt-idx_time+1] .*= u[pml_len+1:Nx+pml_len, pml_len+1:Ny+pml_len]
        if idx_time % saveRatio == 0
            U[:,:,Int(ceil((Nt-idx_time+1)/saveRatio))] .*= u[pml_len+1:Nx+pml_len, pml_len+1:Ny+pml_len]
        end
        
    end
    
    @cuda blocks=cublocks threads=cuthreads time_int_wavefield_c!(U, c, grad, dt)

    return Array{myReal}(grad)
end

function adjoint_single_source_c_host_ram(data, c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, idx_source, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, saveRatio=1)

    @assert (Nt, receiver_num) == size(data) "The size of data should be (Nt, receiver_num)"

    source_position = check_source_position(source_position, source_num)
    source_vals = check_source_vals(source_vals, Nt, source_num)
    receiver_position = check_receiver_position(receiver_position, receiver_num)

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

    # initialize
    rho = 1000 .* ones(size(c))
    a = -1 ./ rho
    b = -1 .* rho .* c.^2
    # c = CuArray{myReal}(c)

    u, vx, vy, wx, wy = init_grid_pml(myReal, Nx, Ny, pml_len)
    a_x, a_y, b_pml = init_parameters_pml(myReal, a, b, pml_len)
    sigma_x, sigma_y, sigma_x_half, sigma_y_half = build_sigma(myReal, Nx, Ny, pml_len, pml_coef)
    forward_data = CUDA.zeros(myReal, Nt, receiver_num)
    U = zeros(myReal, Nx, Ny, Int(floor(Nt/saveRatio)))
    grad = zeros(myReal, Nx, Ny)

    # adj_forward_modeling!(U, u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml, source_position_x, source_position_y, source_vals_device, idx_source, sigma_x_half, sigma_y_half, a_x, a_y, receiver_position_x, receiver_position_y, forward_data, receiver_num, saveRatio, pml_len, cublocks, cublocks_source, cublocks_receiver, cuthreads, cuthreads_source, cuthreads_receiver)
    # forward
    for idx_time = 1:Nt

        @cuda blocks=cublocks threads=cuthreads update_pressure_pml_4th!(u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml)

        @cuda blocks=cublocks_source threads=cuthreads_source update_source_idx!(u, source_position_x, source_position_y, source_vals_device, idx_source, idx_time, dt)

        @cuda blocks=cublocks threads=cuthreads update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx_pml, Ny_pml)

        @cuda blocks=cublocks threads=cuthreads update_velocity_pml_4th!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx_pml, Ny_pml)

        @cuda blocks=cublocks_receiver threads=cuthreads_receiver record_wavefield!(u, receiver_position_x, receiver_position_y, forward_data, receiver_num, idx_time)

        if idx_time % saveRatio == 0
            U[:, :, Int(ceil(idx_time/saveRatio))] = Array{myReal}(u[pml_len+1:end-pml_len, pml_len+1:end-pml_len])
        end

    end

    # time differential
    # @cuda blocks=cublocks threads=cuthreads diff_twice_time_wavefield!(U, dt)
    @. @views U[:,:,2:end-1] = (U[:,:,3:end] - 2*U[:,:,2:end-1] + U[:,:,1:end-2]) / dt^2
    # backward
    u, vx, vy, wx, wy = init_grid_pml(myReal, Nx, Ny, pml_len)
    adjoint_source = CuArray{myReal}(data) - forward_data

    # adj_backward_modeling!(U, u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml, receiver_position_x, receiver_position_y, adjoint_source, receiver_num, Nt, sigma_x_half, sigma_y_half, a_x, a_y, saveRatio, pml_len, cublocks, cublocks_receiver, cuthreads, cuthreads_receiver)
    for idx_time = 1:Nt

        @cuda blocks=cublocks threads=cuthreads update_pressure_pml_4th!(u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml)

        @cuda blocks=cublocks_receiver threads=cuthreads_receiver update_source!(u, receiver_position_x, receiver_position_y, adjoint_source, receiver_num, Nt-idx_time+1, dt)
    
        @cuda blocks=cublocks threads=cuthreads update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx_pml, Ny_pml)
    
        @cuda blocks=cublocks threads=cuthreads update_velocity_pml_4th!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx_pml, Ny_pml)

        # U[:,:,Nt-idx_time+1] .*= u[pml_len+1:Nx+pml_len, pml_len+1:Ny+pml_len]
        if idx_time % saveRatio == 0
            U[:,:,Int(ceil((Nt-idx_time+1)/saveRatio))] .*= Array{myReal}(u[pml_len+1:Nx+pml_len, pml_len+1:Ny+pml_len])
        end
        
    end
    
    # @cuda blocks=cublocks threads=cuthreads time_int_wavefield_c!(U, c, grad, dt)
    grad = sum(U, dims=3)[:,:,1] .* dt
    grad = -2 .* grad ./ c.^3

    return Array{myReal}(grad)
end



function adjoint_c(data, c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, saveRatio=1)

    grad = Array{myReal}(0 .* c)
    for idx = 1:source_num
        grad0 = Array{myReal}(0 .* c)
        @views data0 = data[:,:,idx]
        grad0 = adjoint_single_source_c(data0, c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, idx, receiver_num, receiver_position, pml_len, pml_coef; blockx=blockx, blocky=blocky, saveRatio=saveRatio)
        grad += grad0
    end

    return grad
end

function adjoint_c_host_ram(data, c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, saveRatio=1)

    grad = Array{myReal}(0 .* c)
    for idx = 1:source_num
        grad0 = Array{myReal}(0 .* c)
        @views data0 = data[:,:,idx]
        grad0 = adjoint_single_source_c_host_ram(data0, c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, idx, receiver_num, receiver_position, pml_len, pml_coef; blockx=blockx, blocky=blocky, saveRatio=saveRatio)
        grad += grad0
    end

    return grad
end