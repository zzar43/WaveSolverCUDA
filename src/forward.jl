include("base.jl")
include("utils.jl")
include("kernels.jl")

# input is a and b
# only 4th order
# need to have specfic source mode
# output is a seis gather

function forward_problem_acoustic(a, b, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position; blockx=16, blocky=16, idx_source=0)

    # error check
    if idx_source < 1 || idx_source > source_num
        throw(ArgumentError("idx_source should be 0 or in [1, source_num]"))
    end

    # parameters
    Nx_pml = Nx + 2*pml_len
    Ny_pml = Ny + 2*pml_len
    source_position_pml = source_position .+ pml_len
    receiver_position_pml = receiver_position .+ pml_len

    # to device
    source_vals_device, source_position_x, source_position_y, receiver_position_x, receiver_position_y = to_device_source_receiver(source_position_pml, source_vals, source_num, Nt, receiver_position_pml, receiver_num)

    # CUDA parameters
    cuthreads, cublocks = init_CUDA_grid_parameters(Nx, Ny; blockx, blocky)
    cuthreads_source, cublocks_source = init_CUDA_source_parameters(source_num; blockx)
    cuthreads_receiver, cublocks_receiver = init_CUDA_source_parameters(receiver_num; blockx)

    # initialize
    u, vx, vy, wx, wy = init_grid_pml(myReal, Nx, Ny, pml_len)
    a_x, a_y, b_pml = init_parameters_pml(myReal, a, b, pml_len)
    sigma_x, sigma_y, sigma_x_half, sigma_y_half = build_sigma(myReal, Nx, Ny, pml_len, pml_coef)
    if idx_source == 0
        data = CUDA.zeros(myReal, source_num, Nt, receiver_num)
    else
        data = CUDA.zeros(myReal, Nt, receiver_num)
    end

    # main loop
    if idx_source == 0
        for idx = 1:source_num
            for idx_time = 1:Nt

                @cuda blocks=cublocks threads=cuthreads update_pressure_pml_4th!(u, vx, vy, wx, wy, sigma_x, sigma_y, b_pml, dx, dy, dt, Nx_pml, Ny_pml)
            
                @cuda blocks=cublocks_source threads=cuthreads_source update_source!(u, source_position_x[idx], source_position_y[idx], source_vals_device[:,idx], 1, idx_time, dt)

                @cuda blocks=cublocks threads=cuthreads update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx_pml, Ny_pml)

                @cuda blocks=cublocks threads=cuthreads update_velocity_pml_4th!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx_pml, Ny_pml)

                # record
                @cuda blocks=cublocks_receiver threads=cuthreads_receiver record_wavefield!(u, receiver_position_x, receiver_position_y, data[:,:,idx], receiver_num, idx_time)

            end
        end
    end

    return Array{myReal}(data)

end