include("base.jl")
include("utils.jl")
include("kernels.jl")

function acoustic_solver_pml(a, b, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, pml_len, pml_coef; blockx=16, blocky=16, order=4, recordWaveField=false)

    check_order!(order)

    # parameters
    Nx_pml = Nx + 2*pml_len
    Ny_pml = Ny + 2*pml_len
    source_position_pml = source_position .+ pml_len

    # to device
    source_vals_device, source_position_x, source_position_y = to_device_source(source_position_pml, source_vals, source_num, Nt)
    
    # CUDA parameters
    cuthreads, cublocks = init_CUDA_grid_parameters(Nx_pml, Ny_pml; blockx, blocky)
    cuthreads_source, cublocks_source = init_CUDA_source_parameters(source_num; blockx)

    # initialize
    u, vx, vy, wx, wy = init_grid_pml(myReal, Nx, Ny, pml_len)
    a_x, a_y, b_pml = init_parameters_pml(myReal, a, b, pml_len)
    sigma_x, sigma_y, sigma_x_half, sigma_y_half = build_sigma(myReal, Nx, Ny, pml_len, pml_coef)
    if recordWaveField == true
        U = CUDA.zeros(myReal, Nt, Nx, Ny)
    end

    # main loop, different order
    if order == 4

        for idx_time = 1:Nt
    
            @cuda blocks=cublocks threads=cuthreads update_pressure_pml_4th!(u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml)
    
            @cuda blocks=cublocks_source threads=cuthreads_source update_source!(u, source_position_x, source_position_y, source_vals_device, source_num, idx_time, dt)
    
            @cuda blocks=cublocks threads=cuthreads update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx_pml, Ny_pml)
    
            @cuda blocks=cublocks threads=cuthreads update_velocity_pml_4th!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx_pml, Ny_pml)
        
            if recordWaveField == true
                U[idx_time, :, :] = u[pml_len+1:end-pml_len, pml_len+1:end-pml_len]
            end
    
        end

    elseif order == 2

        for idx_time = 1:Nt
    
            @cuda blocks=cublocks threads=cuthreads update_pressure_pml_2nd!(u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml)
    
            @cuda blocks=cublocks_source threads=cuthreads_source update_source!(u, source_position_x, source_position_y, source_vals_device, source_num, idx_time, dt)
    
            @cuda blocks=cublocks threads=cuthreads update_auxiliary_pml_2nd!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx_pml, Ny_pml)
    
            @cuda blocks=cublocks threads=cuthreads update_velocity_pml_2nd!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx_pml, Ny_pml)
        
            if recordWaveField == true
                U[idx_time, :, :] = u[pml_len+1:end-pml_len, pml_len+1:end-pml_len]
            end
    
        end

    end

    if recordWaveField == true
        return Array{myReal}(U)
    else
        return Array{myReal}(u)
    end
end
