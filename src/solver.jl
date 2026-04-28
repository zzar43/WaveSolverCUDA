# A new solver
include("base.jl")
include("utils.jl")
include("misc.jl")
include("kernels.jl")


# source integration

function source_integration(source_vals, Nt, dt)
    source_vals0 = 0 .* source_vals
    for idx_time = 1:Nt
        source_vals0[Nt-idx_time+1, :] = sum(source_vals[1:Nt-idx_time+1, :], dims=1) * dt
        # source_vals0[Nt-idx_time+1, :] = sum(source_vals[1:Nt-idx_time+1, :], dims=1)
    end
    return source_vals0
end

function acoustic_solver(a, b, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, idx_source=0, recordWaveField=false, waveFieldOnCUDA=false)

    source_position = check_source_position(source_position, source_num)
    source_vals = check_source_vals(source_vals, Nt, source_num)
    receiver_position = check_receiver_position(receiver_position, receiver_num)

    # parameters
    Nx_pml = Nx + 2*pml_len
    Ny_pml = Ny + 2*pml_len
    source_position_pml = source_position .+ pml_len
    receiver_position_pml = receiver_position .+ pml_len

    # to device
    source_vals_device, source_position_x, source_position_y, receiver_position_x, receiver_position_y = to_device_source_receiver(source_position_pml, source_integration(source_vals, Nt, dt), source_num, Nt, receiver_position_pml, receiver_num)
    
    # CUDA parameters
    cuthreads, cublocks = init_CUDA_grid_parameters(Nx_pml, Ny_pml; blockx, blocky)
    cuthreads_source, cublocks_source = init_CUDA_source_parameters(source_num; blockx)
    cuthreads_receiver, cublocks_receiver = init_CUDA_source_parameters(receiver_num; blockx)

    # initialize
    u, vx, vy, wx, wy = init_grid_pml(myReal, Nx, Ny, pml_len)
    a_x, a_y, b_pml = init_parameters_pml(myReal, a, b, pml_len)
    sigma_x, sigma_y, sigma_x_half, sigma_y_half = build_sigma(myReal, Nx, Ny, pml_len, pml_coef)
    data = CUDA.zeros(myReal, Nt, receiver_num)
    if recordWaveField == true
        U = CUDA.zeros(myReal, Nx, Ny, Nt)
    end

    # main
    if idx_source == 0

        for idx_time = 1:Nt

            @cuda blocks=cublocks threads=cuthreads update_pressure_pml_4th!(u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml)

            @cuda blocks=cublocks_source threads=cuthreads_source update_source!(u, source_position_x, source_position_y, source_vals_device, source_num, idx_time, dt, b_pml)
    
            @cuda blocks=cublocks threads=cuthreads update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx_pml, Ny_pml)
    
            @cuda blocks=cublocks threads=cuthreads update_velocity_pml_4th!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx_pml, Ny_pml)
    
            @cuda blocks=cublocks_receiver threads=cuthreads_receiver record_data!(u, receiver_position_x, receiver_position_y, data, receiver_num, idx_time)

            if recordWaveField == true
                @cuda blocks=cublocks threads=cuthreads record_wavefield!(u, U, Nx, Ny, pml_len, idx_time)
            end
        end

    elseif idx_source >= 1 && idx_source <= source_num

        for idx_time = 1:Nt

            @cuda blocks=cublocks threads=cuthreads update_pressure_pml_4th!(u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml)
    
            @cuda blocks=cublocks_source threads=cuthreads_source update_source_idx!(u, source_position_x, source_position_y, source_vals_device, idx_source, idx_time, dt, b_pml)
    
            @cuda blocks=cublocks threads=cuthreads update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx_pml, Ny_pml)
    
            @cuda blocks=cublocks threads=cuthreads update_velocity_pml_4th!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx_pml, Ny_pml)
    
            @cuda blocks=cublocks_receiver threads=cuthreads_receiver record_data!(u, receiver_position_x, receiver_position_y, data, receiver_num, idx_time)

            if recordWaveField == true
                @cuda blocks=cublocks threads=cuthreads record_wavefield!(u, U, Nx, Ny, pml_len, idx_time)
            end
        end

    else
        error("idx_source is out of range")
    end

    if recordWaveField == true && waveFieldOnCUDA == true
        return Array{myReal}(data), U
    elseif recordWaveField == true && waveFieldOnCUDA == false
        return Array{myReal}(data), Array{myReal}(U)
    else
        return Array{myReal}(data)
    end
    
end

function acoustic_solver_c(c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, idx_source=0, recordWaveField=false, waveFieldOnCUDA=false)

    rho = ones(size(c))
    a = -1 ./ rho
    b = -1 .* rho .* c.^2

    return acoustic_solver(a, b, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=blockx, blocky=blocky, idx_source=idx_source, recordWaveField=recordWaveField, waveFieldOnCUDA=waveFieldOnCUDA)

end

function forward_acoustic(a, b, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16)

    source_position = check_source_position(source_position, source_num)
    source_vals = check_source_vals(source_vals, Nt, source_num)
    receiver_position = check_receiver_position(receiver_position, receiver_num)

    # parameters
    Nx_pml = Nx + 2*pml_len
    Ny_pml = Ny + 2*pml_len
    source_position_pml = source_position .+ pml_len
    receiver_position_pml = receiver_position .+ pml_len

    # to device
    source_vals_device, source_position_x, source_position_y, receiver_position_x, receiver_position_y = to_device_source_receiver(source_position_pml, source_integration(source_vals, Nt, dt), source_num, Nt, receiver_position_pml, receiver_num)
    
    # CUDA parameters
    cuthreads, cublocks = init_CUDA_grid_parameters(Nx_pml, Ny_pml; blockx, blocky)
    cuthreads_source, cublocks_source = init_CUDA_source_parameters(source_num; blockx)
    cuthreads_receiver, cublocks_receiver = init_CUDA_source_parameters(receiver_num; blockx)

    # initialize
    u, vx, vy, wx, wy = init_grid_pml(myReal, Nx, Ny, pml_len)
    a_x, a_y, b_pml = init_parameters_pml(myReal, a, b, pml_len)
    sigma_x, sigma_y, sigma_x_half, sigma_y_half = build_sigma(myReal, Nx, Ny, pml_len, pml_coef)
    data = CUDA.zeros(myReal, Nt, receiver_num, source_num)

    # main
    for idx_source = 1:source_num

        for idx_time = 1:Nt

            @cuda blocks=cublocks threads=cuthreads update_pressure_pml_4th!(u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml)
    
            @cuda blocks=cublocks_source threads=cuthreads_source update_source_idx!(u, source_position_x, source_position_y, source_vals_device, idx_source, idx_time, dt, b_pml)
    
            @cuda blocks=cublocks threads=cuthreads update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx_pml, Ny_pml)
    
            @cuda blocks=cublocks threads=cuthreads update_velocity_pml_4th!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx_pml, Ny_pml)
    
            @cuda blocks=cublocks_receiver threads=cuthreads_receiver record_data!(u, receiver_position_x, receiver_position_y, data, receiver_num, idx_time, idx_source)

        end

    end

    return Array{myReal}(data)

end

function forward_acoustic_c(c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16)
    
    rho = ones(size(c))
    a = -1 ./ rho
    b = -1 .* rho .* c.^2

    return forward_acoustic(a, b, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=blockx, blocky=blocky)

end

function backward_acoustic_c(c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16)

    source_position = check_source_position(source_position, source_num)
    source_vals = check_source_vals(source_vals, Nt, source_num)
    receiver_position = check_receiver_position(receiver_position, receiver_num)

    rho = ones(size(c))
    a = -1 ./ rho
    b = -1 .* rho .* c.^2
    c = CuArray{myReal}(c)

    # parameters
    Nx_pml = Nx + 2*pml_len
    Ny_pml = Ny + 2*pml_len
    source_position_pml = source_position .+ pml_len
    receiver_position_pml = receiver_position .+ pml_len

    # to device
    source_vals_device, source_position_x, source_position_y, receiver_position_x, receiver_position_y = to_device_source_receiver(source_position_pml, source_integration(source_vals, Nt, dt), source_num, Nt, receiver_position_pml, receiver_num)
    
    # CUDA parameters
    cuthreads, cublocks = init_CUDA_grid_parameters(Nx_pml, Ny_pml; blockx, blocky)
    cuthreads_source, cublocks_source = init_CUDA_source_parameters(source_num; blockx)
    cuthreads_receiver, cublocks_receiver = init_CUDA_source_parameters(receiver_num; blockx)

    # initialize
    u, vx, vy, wx, wy = init_grid_pml(myReal, Nx, Ny, pml_len)
    a_x, a_y, b_pml = init_parameters_pml(myReal, a, b, pml_len)
    sigma_x, sigma_y, sigma_x_half, sigma_y_half = build_sigma(myReal, Nx, Ny, pml_len, pml_coef)
    U = CUDA.zeros(myReal, Nx, Ny, Nt)

    # main
    for idx_time = 1:Nt

        @cuda blocks=cublocks threads=cuthreads update_pressure_pml_4th!(u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml)

        @cuda blocks=cublocks_source threads=cuthreads_source update_source!(u, source_position_x, source_position_y, source_vals_device, source_num, idx_time, dt, b_pml)

        @cuda blocks=cublocks threads=cuthreads update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx_pml, Ny_pml)

        @cuda blocks=cublocks threads=cuthreads update_velocity_pml_4th!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx_pml, Ny_pml)

        @cuda blocks=cublocks threads=cuthreads record_adj_wavefield!(u, U, Nx, Ny, pml_len, idx_time, Nt)                                                                            

    end

    return Array{myReal}(U)
    
end

# I think I can write a full function for adjoint state method, instead of those intermediate functions.
function adjoint_method_c0(received_data, c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16)

    source_position = check_source_position(source_position, source_num)
    source_vals = check_source_vals(source_vals, Nt, source_num)
    receiver_position = check_receiver_position(receiver_position, receiver_num)

    rho = ones(size(c))
    a = -1 ./ rho
    b = -1 .* rho .* c.^2
    c = CuArray{myReal}(c)

    # parameters
    Nx_pml = Nx + 2*pml_len
    Ny_pml = Ny + 2*pml_len
    source_position_pml = source_position .+ pml_len
    receiver_position_pml = receiver_position .+ pml_len

    # to device
    source_vals_device, source_position_x, source_position_y, receiver_position_x, receiver_position_y = to_device_source_receiver(source_position_pml, source_integration(source_vals, Nt, dt), source_num, Nt, receiver_position_pml, receiver_num)
    # received_data = CuArray{myReal}(received_data)
    
    # CUDA parameters
    cuthreads, cublocks = init_CUDA_grid_parameters(Nx_pml, Ny_pml; blockx, blocky)
    cuthreads_source, cublocks_source = init_CUDA_source_parameters(source_num; blockx)
    cuthreads_receiver, cublocks_receiver = init_CUDA_source_parameters(receiver_num; blockx)

    # initialize
    a_x, a_y, b_pml = init_parameters_pml(myReal, a, b, pml_len)
    sigma_x, sigma_y, sigma_x_half, sigma_y_half = build_sigma(myReal, Nx, Ny, pml_len, pml_coef)
    grad = zeros(myReal, Nx, Ny)

    # loop over all sources
    for idx_source = 1:source_num
        
        # initialize
        u, vx, vy, wx, wy = init_grid_pml(myReal, Nx, Ny, pml_len)
        data = CUDA.zeros(myReal, Nt, receiver_num)
        U = CUDA.zeros(myReal, Nx, Ny, Nt)
        adjoint_source = zeros(myReal, Nt, receiver_num)
        grad0 = CUDA.zeros(myReal, Nx, Ny)

        # forward wavefield
        for idx_time = 1:Nt

            @cuda blocks=cublocks threads=cuthreads update_pressure_pml_4th!(u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml)
    
            @cuda blocks=cublocks_source threads=cuthreads_source update_source_idx!(u, source_position_x, source_position_y, source_vals_device, idx_source, idx_time, dt, b_pml)
    
            @cuda blocks=cublocks threads=cuthreads update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx_pml, Ny_pml)
    
            @cuda blocks=cublocks threads=cuthreads update_velocity_pml_4th!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx_pml, Ny_pml)
    
            @cuda blocks=cublocks_receiver threads=cuthreads_receiver record_data!(u, receiver_position_x, receiver_position_y, data, receiver_num, idx_time)

            @cuda blocks=cublocks threads=cuthreads record_wavefield!(u, U, Nx, Ny, pml_len, idx_time)

        end

        # adjoint source
        # @cuda blocks=cublocks_receiver threads=cuthreads_receiver build_adjoint_source!(adjoint_source, data, received_data, idx_source, Nt, receiver_num)
        # It seems do the adjoint source on CPU is faster than on GPU.
        adjoint_source = Array{myReal}(data) - received_data[:,:,idx_source]
        adjoint_source = adjoint_source[end:-1:1,:]
        adjoint_source = CuArray{myReal}(source_integration(adjoint_source, Nt, dt))

        # second order time differentiation
        @cuda blocks=cublocks threads=cuthreads diff2_time_inplace_CUDA!(U, dt, Nx, Ny, Nt)

        # adjoint wavefield
        u, vx, vy, wx, wy = init_grid_pml(myReal, Nx, Ny, pml_len)
        for idx_time = 1:Nt

            @cuda blocks=cublocks threads=cuthreads update_pressure_pml_4th!(u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml)

            @cuda blocks=cublocks_receiver threads=cuthreads_receiver update_source!(u, receiver_position_x, receiver_position_y, adjoint_source, receiver_num, idx_time, dt, b_pml)

            @cuda blocks=cublocks threads=cuthreads update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx_pml, Ny_pml)

            @cuda blocks=cublocks threads=cuthreads update_velocity_pml_4th!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx_pml, Ny_pml)

            @cuda blocks=cublocks threads=cuthreads record_adj_wavefield_inner_product!(u, U, Nx, Ny, pml_len, idx_time, Nt)
        end

        # numerical integration
        @cuda blocks=cublocks threads=cuthreads time_int_wavefield_c!(U, c, grad0, dt)

        grad += Array{myReal}(grad0)

    end

    return grad

end


function adjoint_method_c(received_data, c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16)

    source_position = check_source_position(source_position, source_num)
    source_vals = check_source_vals(source_vals, Nt, source_num)
    receiver_position = check_receiver_position(receiver_position, receiver_num)

    rho = ones(size(c))
    a = -1 ./ rho
    b = -1 .* rho .* c.^2
    c_device = CuArray{myReal}(c)
    received_data = CuArray{myReal}(received_data)

    # parameters
    Nx_pml = Nx + 2*pml_len
    Ny_pml = Ny + 2*pml_len
    source_position_pml = source_position .+ pml_len
    receiver_position_pml = receiver_position .+ pml_len

    source_vals_device = nothing
    source_position_x = nothing
    source_position_y = nothing
    receiver_position_x = nothing
    receiver_position_y = nothing
    
    # CUDA parameters
    cuthreads, cublocks = init_CUDA_grid_parameters(Nx_pml, Ny_pml; blockx, blocky)
    cuthreads_source, cublocks_source = init_CUDA_source_parameters(source_num; blockx)
    cuthreads_receiver, cublocks_receiver = init_CUDA_source_parameters(receiver_num; blockx)

    a_x = nothing
    a_y = nothing
    b_pml = nothing
    sigma_x = nothing
    sigma_y = nothing
    sigma_x_half = nothing
    sigma_y_half = nothing

    grad = zeros(myReal, Nx, Ny)

    try
        # to device
        source_vals_device, source_position_x, source_position_y, receiver_position_x, receiver_position_y = to_device_source_receiver(source_position_pml, source_integration(source_vals, Nt, dt), source_num, Nt, receiver_position_pml, receiver_num)

        # initialize
        a_x, a_y, b_pml = init_parameters_pml(myReal, a, b, pml_len)
        sigma_x, sigma_y, sigma_x_half, sigma_y_half = build_sigma(myReal, Nx, Ny, pml_len, pml_coef)
        u, vx, vy, wx, wy = init_grid_pml(myReal, Nx, Ny, pml_len)
        data = CUDA.zeros(myReal, Nt, receiver_num)
        U = CUDA.zeros(myReal, Nx, Ny, Nt)
        grad0 = CUDA.zeros(myReal, Nx, Ny)
        adjoint_source = CUDA.zeros(myReal, Nt, receiver_num)

        # loop over all sources
        for idx_source = 1:source_num
            
            # reset buffers instead of reallocating
            fill!(u, 0)
            fill!(vx, 0)
            fill!(vy, 0)
            fill!(wx, 0)
            fill!(wy, 0)
            fill!(data, 0)
            fill!(U, 0)
            fill!(grad0, 0)
            fill!(adjoint_source, 0)
            # forward wavefield
            for idx_time = 1:Nt

                @cuda blocks=cublocks threads=cuthreads update_pressure_pml_4th!(u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml)
        
                @cuda blocks=cublocks_source threads=cuthreads_source update_source_idx!(u, source_position_x, source_position_y, source_vals_device, idx_source, idx_time, dt, b_pml)
        
                @cuda blocks=cublocks threads=cuthreads update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx_pml, Ny_pml)
        
                @cuda blocks=cublocks threads=cuthreads update_velocity_pml_4th!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx_pml, Ny_pml)
        
                @cuda blocks=cublocks_receiver threads=cuthreads_receiver record_data!(u, receiver_position_x, receiver_position_y, data, receiver_num, idx_time)

                @cuda blocks=cublocks threads=cuthreads record_wavefield!(u, U, Nx, Ny, pml_len, idx_time)

            end

            # adjoint source
            @cuda blocks=cublocks_receiver threads=cuthreads_receiver build_adjoint_source!(adjoint_source, data, received_data, idx_source, Nt, receiver_num)
            @cuda blocks=cublocks_receiver threads=cuthreads_receiver source_integration_on_device!(adjoint_source, receiver_num, Nt, dt)
            # It seems do the adjoint source on CPU is faster than on GPU for small problems. But for large problems, the GPU version is faster.
            # adjoint_source = Array{myReal}(data) - received_data[:,:,idx_source]
            # adjoint_source = adjoint_source[end:-1:1,:]
            # adjoint_source = CuArray{myReal}(source_integration(adjoint_source, Nt, dt))

            # second order time differentiation
            @cuda blocks=cublocks threads=cuthreads diff2_time_inplace_CUDA!(U, dt, Nx, Ny, Nt)

            # adjoint wavefield
            fill!(u, 0)
            fill!(vx, 0)
            fill!(vy, 0)
            fill!(wx, 0)
            fill!(wy, 0)
            for idx_time = 1:Nt

                @cuda blocks=cublocks threads=cuthreads update_pressure_pml_4th!(u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml)

                @cuda blocks=cublocks_receiver threads=cuthreads_receiver update_source!(u, receiver_position_x, receiver_position_y, adjoint_source, receiver_num, idx_time, dt, b_pml)

                @cuda blocks=cublocks threads=cuthreads update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx_pml, Ny_pml)

                @cuda blocks=cublocks threads=cuthreads update_velocity_pml_4th!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx_pml, Ny_pml)

                @cuda blocks=cublocks threads=cuthreads record_adj_wavefield_inner_product!(u, U, Nx, Ny, pml_len, idx_time, Nt)
            end

            # numerical integration
            # Need to move this piece of code to GPU later.
            @cuda blocks=cublocks threads=cuthreads time_int_wavefield_c!(U, c_device, grad0, dt)

            grad += Array{myReal}(grad0)
        end

        return grad
    finally
        CUDA.synchronize()
        source_vals_device = nothing
        source_position_x = nothing
        source_position_y = nothing
        receiver_position_x = nothing
        receiver_position_y = nothing
        a_x = nothing
        a_y = nothing
        b_pml = nothing
        sigma_x = nothing
        sigma_y = nothing
        sigma_x_half = nothing
        sigma_y_half = nothing
        c_device = nothing
        GC.gc(false)
        CUDA.reclaim()
    end

end
