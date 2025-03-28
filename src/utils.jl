include("base.jl")

# source function

function source_ricker(center_fre::Real, center_time::Real, t)
    x = (1 .- 2*pi^2 .* center_fre^2 .* (t .- center_time).^2) .* exp.(-pi^2*center_fre^2 .* (t .- center_time).^2)
    return Vector{myReal}(x)
end

function source_ricker_int(center_fre::Real, center_time::Real, t)
    x = (1 .- 2*pi^2 .* center_fre^2 .* (t .- center_time).^2) .* exp.(-pi^2*center_fre^2 .* (t .- center_time).^2);
    xx = 0 .* x
    for i = 1:length(t)
        xx[i] = sum(x[1:i]) * (t[2]-t[1])
    end
    return Vector{myReal}(xx ./ maximum(xx))
end

# PML utils

function extend_pml(myReal, u, pml_len)
    Nx, Ny = size(u)
    u_pml = zeros(myReal, Nx + 2 * pml_len, Ny + 2 * pml_len)
    u_pml[pml_len+1:Nx+pml_len, pml_len+1:Ny+pml_len] = u
    for i = 1:pml_len
        u_pml[i,:] = u_pml[pml_len+1,:]
        u_pml[Nx+2*pml_len-i+1,:] = u_pml[Nx+pml_len,:]
        u_pml[:,i] = u_pml[:,pml_len+1]
        u_pml[:,Ny+2*pml_len-i+1] = u_pml[:,Ny+pml_len]
    end
    return u_pml
end

function build_sigma(myReal, Nx, Ny, pml_len, pml_coef)
    sigma_x = zeros(myReal, Nx + 2*pml_len, Ny + 2*pml_len)
    sigma_y = zeros(myReal, Nx + 2*pml_len, Ny + 2*pml_len)
    sigma_x_half = zeros(myReal, Nx + 2*pml_len-1, Ny + 2*pml_len)
    sigma_y_half = zeros(myReal, Nx + 2*pml_len, Ny + 2*pml_len-1)
    vals = range(0,1,pml_len * 2 - 1) * pml_coef
    vals = vals .^ 2
    for i = 1:pml_len
        sigma_x[pml_len-i+1,:] .= vals[(i-1)*2+1]
        sigma_x[Nx + pml_len + i,:] .= vals[(i-1)*2+1]
        sigma_y[:,pml_len-i+1] .= vals[(i-1)*2+1]
        sigma_y[:,Ny + pml_len + i] .= vals[(i-1)*2+1]
    end
    for i = 1:pml_len-1
        sigma_x_half[pml_len-i,:] .= vals[(i-1)*2+2]
        sigma_x_half[Nx + pml_len + i,:] .= vals[(i-1)*2+2]
        sigma_y_half[:,pml_len-i] .= vals[(i-1)*2+2]
        sigma_y_half[:,Ny + pml_len + i] .= vals[(i-1)*2+2]
    end
    return CuArray{myReal}(sigma_x), CuArray{myReal}(sigma_y), CuArray{myReal}(sigma_x_half), CuArray{myReal}(sigma_y_half)
end

# initialize parameters for non-PML and PML cases

function init_grid(myReal, Nx, Ny)

    u = CUDA.zeros(myReal, Nx, Ny)
    vx = CUDA.zeros(myReal, Nx-1, Ny)
    vy = CUDA.zeros(myReal, Nx, Ny-1)

    return u, vx, vy
end

function init_parameters(myReal, a, b)
    a_x = CuArray{myReal}((a[1:end-1, :] .+ a[2:end, :]) / 2)
    a_y = CuArray{myReal}((a[:, 1:end-1] .+ a[:, 2:end]) / 2)
    b = CuArray{myReal}(b)
    return a_x, a_y, b
end

function init_parameters_pml(myReal, a, b, pml_len)

    a_pml = extend_pml(myReal, a, pml_len)
    b_pml = extend_pml(myReal, b, pml_len)

    a_x = CuArray{myReal}((a_pml[1:end-1, :] .+ a_pml[2:end, :]) / 2)
    a_y = CuArray{myReal}((a_pml[:, 1:end-1] .+ a_pml[:, 2:end]) / 2)

    return a_x, a_y, CuArray{myReal}(b_pml)
end

function init_grid_pml(myReal, Nx, Ny, pml_len)

    u = CUDA.zeros(myReal, Nx+2*pml_len, Ny+2*pml_len)
    vx = CUDA.zeros(myReal, Nx+2*pml_len-1, Ny+2*pml_len)
    vy = CUDA.zeros(myReal, Nx+2*pml_len, Ny+2*pml_len-1)
    wx = CUDA.zeros(myReal, Nx+2*pml_len, Ny+2*pml_len)
    wy = CUDA.zeros(myReal, Nx+2*pml_len, Ny+2*pml_len)

    return u, vx, vy, wx, wy
end

# CUDA parameters
# For 2 dimensional case

function init_CUDA_grid_parameters(Nx, Ny; blockx=16, blocky=16)

    gridx = Int(ceil((Nx / blockx)))
    gridy = Int(ceil((Ny / blocky)))
    cuthreads = (blockx, blocky, 1)
    cublocks = (gridx, gridy, 1)

    return cuthreads, cublocks
end

function init_CUDA_source_parameters(source_num; blockx=16)

    cuthreads_source = (blockx)
    cublocks_source = (Int(ceil(source_num / blockx)))

    return cuthreads_source, cublocks_source
end

# source and receiver to device
function to_device_source(source_position, source_vals, source_num, Nt)

    source_vals_device = CuArray{myReal}(reshape(source_vals, source_num, Nt))
    source_position_x = CuArray{myInt}(reshape(source_position, source_num, 2)[:,1])
    source_position_y = CuArray{myInt}(reshape(source_position, source_num, 2)[:,2])

    return source_vals_device, source_position_x, source_position_y
end

function to_device_source_receiver(source_position, source_vals, source_num, Nt, receiver_position, receiver_num)

    source_vals_device = CuArray{myReal}(reshape(source_vals, source_num, Nt))
    source_position_x = CuArray{myInt}(reshape(source_position, source_num, 2)[:,1])
    source_position_y = CuArray{myInt}(reshape(source_position, source_num, 2)[:,2])

    receiver_position_x = CuArray{myInt}(reshape(receiver_position, receiver_num, 2)[:,1])
    receiver_position_y = CuArray{myInt}(reshape(receiver_position, receiver_num, 2)[:,2])

    return source_vals_device, source_position_x, source_position_y, receiver_position_x, receiver_position_y
end


