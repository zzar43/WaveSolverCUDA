include("base.jl")

# ==============================
# SOURCE
# ==============================

function update_source_fixed!(u, source_position_x, source_position_y, source_vals, source_num, idx_time, dt)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i >= 1 && i <= source_num
        ix, iy = source_position_x[i], source_position_y[i]
        if ix >= 1 && ix <= size(u,1) && iy >= 1 && iy <= size(u,2)
            u[ix, iy] = source_vals[idx_time, i] * dt
        end
    end

    return nothing
end

function update_source!(u, source_position_x, source_position_y, source_vals, source_num, idx_time, dt)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i >= 1 && i <= source_num
        ix, iy = source_position_x[i], source_position_y[i]
        if ix >= 1 && ix <= size(u,1) && iy >= 1 && iy <= size(u,2)
            u[ix, iy] += source_vals[idx_time, i] * dt
            # u[ix, iy] += source_vals[idx_time, i]
        end
    end

    return nothing
end

function update_source_idx!(u, source_position_x, source_position_y, source_vals, idx_source, idx_time, dt)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i == idx_source
        ix, iy = source_position_x[i], source_position_y[i]
        if ix >= 1 && ix <= size(u,1) && iy >= 1 && iy <= size(u,2)
            u[ix, iy] += source_vals[idx_time, i] * dt
            # u[ix, iy] += source_vals[idx_time, i]
        end
    end

    return nothing
end

# It seems slower than CPU version. Not used.
function source_integration_on_device!(source_vals, source_num, Nt, dt)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i >= 1 && i <= source_num
        for idx_time = 1:Nt
            for j = 1:Nt-idx_time+1
                source_vals[Nt-idx_time+1, i] += source_vals[j, i]
            end
            source_vals[Nt-idx_time+1, i] *= dt
        end
    end

    return nothing
end

# build adjoint source
function build_adjoint_source!(adjoint_source, data, received_data, idx_source, Nt, receiver_num)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i >= 1 && i <= receiver_num
        for idx_time = 1:Nt
            adjoint_source[Nt-idx_time+1, i] = data[idx_time, i] - received_data[idx_time, i, idx_source]
        end
    end

    return nothing

end

# ==============================
# RECEIVER
# ==============================

function record_data!(u, receiver_position_x, receiver_position_y, receiver_vals, receiver_num, idx_time)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i <= receiver_num
        ix, iy = receiver_position_x[i], receiver_position_y[i]
        if ix >= 1 && ix <= size(u,1) && iy >= 1 && iy <= size(u,2)
            receiver_vals[idx_time, i] = u[ix, iy]
        end
    end

    return nothing
end

function record_data!(u, receiver_position_x, receiver_position_y, receiver_vals, receiver_num, idx_time, idx_source)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i <= receiver_num
        ix, iy = receiver_position_x[i], receiver_position_y[i]
        if ix >= 1 && ix <= size(u,1) && iy >= 1 && iy <= size(u,2)
            receiver_vals[idx_time, i, idx_source] = u[ix, iy]
        end
    end

    return nothing
end

function record_wavefield!(u, U, Nx, Ny, pml_len, idx_time)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i >= pml_len+1 && i <= Nx+pml_len && j >= pml_len+1 && j <= Ny+pml_len
        U[i-pml_len,j-pml_len,idx_time] = u[i,j]
    end

    return nothing
end

function record_adj_wavefield_inner_product!(v, Utt, Nx, Ny, pml_len, idx_time, Nt)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i >= pml_len+1 && i <= Nx+pml_len && j >= pml_len+1 && j <= Ny+pml_len
        Utt[i-pml_len,j-pml_len,Nt-idx_time+1] = Utt[i-pml_len,j-pml_len,Nt-idx_time+1] * v[i,j]
    end

    return nothing
end

function record_adj_wavefield!(u, U, Nx, Ny, pml_len, idx_time, Nt)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i >= pml_len+1 && i <= Nx+pml_len && j >= pml_len+1 && j <= Ny+pml_len
        U[i-pml_len,j-pml_len,Nt-idx_time+1] = u[i,j]
    end

    return nothing
end

# ==============================
# ADJOINT METHOD
# ==============================
function diff2_time_inplace_CUDA!(U, dt, Nx, Ny, Nt)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i >= 1 && i <= Nx && j >= 1 && j <= Ny

        u_prev = U[i, j, 1]
        u_curr = U[i, j, 2]
        U[i, j, 1] = zero(eltype(U))
        for idx_time in 2:Nt-1
            u_next = U[i, j, idx_time+1]
            U[i, j, idx_time] = (u_next - 2*u_curr + u_prev) / dt^2
            u_prev = u_curr
            u_curr = u_next
        end
        U[i, j, Nt] = zero(eltype(U))
    end
        
    return nothing

end

function time_int_wavefield_c!(U, c, grad, dt)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i >= 1 && i <= size(U,1) && j >= 1 && j <= size(U,2)
        for idx_time = 2:size(U,3)-1
            # grad[i,j] += 2 * U[i,j,idx_time] * dt / c[i,j]^3
            grad[i,j] += 2 * U[i,j,idx_time] / c[i,j]^3
        end
    end
    
    return nothing
end

# ==============================
# 2ND ORDER
# ==============================

function update_velocity_2nd!(u, vx, vy, a_x, a_y, dx, dy, dt, Nx, Ny)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i >= 1 && i <= Nx-1 && j >= 1 && j <= Ny
        @inbounds vx[i,j] = vx[i,j] + dt * a_x[i,j] * (u[i+1,j] - u[i,j]) / dx
    end
    if i >= 1 && i <= Nx && j >= 1 && j <= Ny-1
        @inbounds vy[i,j] = vy[i,j] + dt * a_y[i,j] * (u[i,j+1] - u[i,j]) / dy
    end

    return nothing
end

function update_pressure_2nd!(u, vx, vy, b, dx, dy, dt, Nx, Ny)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i > 1 && i < Nx && j > 1 && j < Ny
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (vx[i,j] - vx[i-1,j]) / dx + 
            (vy[i,j] - vy[i,j-1]) / dy
        )
    end

    return nothing
end

# ==============================
# 4TH ORDER
# ==============================

function update_velocity_4th!(u, vx, vy, a_x, a_y, dx, dy, dt, Nx, Ny)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i >= 2 && i <= Nx-2 && j >= 1 && j <= Ny
        @inbounds vx[i,j] = vx[i,j] + dt * a_x[i,j] * (u[i-1,j] - 27*u[i,j] + 27*u[i+1,j] - u[i+2,j]) / (24*dx)
    end
    if i == 1 && j >= 1 && j <= Ny
        @inbounds vx[i,j] = vx[i,j] + dt * a_x[i,j] * (-22*u[i,j] + 17*u[i+1,j] + 9*u[i+2,j] - 5*u[i+3,j] + u[i+4,j]) / (24*dx)
    end
    if i == Nx-1 && j >= 1 && j <= Ny
        @inbounds vx[i,j] = vx[i,j] + dt * a_x[i,j] * (-1*u[i-3,j] + 5*u[i-2,j] - 9*u[i-1,j] - 17*u[i,j] + 22*u[i+1,j]) / (24*dx)
    end

    if i >= 1 && i <= Nx && j >= 2 && j <= Ny-2
        @inbounds vy[i,j] = vy[i,j] + dt * a_y[i,j] * (u[i,j-1] - 27*u[i,j] + 27*u[i,j+1] - u[i,j+2]) / (24*dy)
    end
    if i >= 1 && i <= Nx && j == 1
        @inbounds vy[i,j] = vy[i,j] + dt * a_y[i,j] * (-22*u[i,j] + 17*u[i,j+1] + 9*u[i,j+2] - 5*u[i,j+3] + u[i,j+4]) / (24*dy)
    end
    if i >= 1 && i <= Nx && j == Ny-1
        @inbounds vy[i,j] = vy[i,j] + dt * a_y[i,j] * (-1*u[i,j-3] + 5*u[i,j-2] - 9*u[i,j-1] - 17*u[i,j] + 22*u[i,j+1]) / (24*dy)
    end

    return
end

function update_pressure_4th!(u, vx, vy, b, dx, dy, dt, Nx, Ny)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    # center
    if i > 2 && i < Nx-1 && j > 2 && j < Ny-1
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (vx[i-2,j] - 27*vx[i-1,j] + 27*vx[i,j] - vx[i+1,j]) / (24*dx) + 
            (vy[i,j-2] - 27*vy[i,j-1] + 27*vy[i,j] - vy[i,j+1]) / (24*dy)
        )
    end

    # top
    if i == 2 && j > 2 && j < Ny-1
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (-22*vx[i-1,j] + 17*vx[i,j] + 9*vx[i+1,j] - 5*vx[i+2,j] + vx[i+3,j]) / (24*dx) + 
            (vy[i,j-2] - 27*vy[i,j-1] + 27*vy[i,j] - vy[i,j+1]) / (24*dy)
        )
    end

    # bottom
    if i == Nx-1 && j > 2 && j < Ny-1
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (-1*vx[i-4,j] + 5*vx[i-3,j] - 9*vx[i-2,j] - 17*vx[i-1,j] + 22*vx[i,j]) / (24*dx) +
            (vy[i,j-2] - 27*vy[i,j-1] + 27*vy[i,j] - vy[i,j+1]) / (24*dy)
        )
    end

    # left
    if i > 1 && i < Nx && j == 2
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (vx[i-2,j] - 27*vx[i-1,j] + 27*vx[i,j] - vx[i+1,j]) / (24*dx) + 
            (-22*vy[i,j-1] + 17*vy[i,j] + 9*vy[i,j+1] - 5*vy[i,j+2] + vy[i,j+3]) / (24*dy)
        )
    end

    # right
    if i > 1 && i < Nx && j == Ny-1
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (vx[i-2,j] - 27*vx[i-1,j] + 27*vx[i,j] - vx[i+1,j]) / (24*dx) + 
            (-1*vy[i,j-4] + 5*vy[i,j-3] - 9*vy[i,j-2] - 17*vy[i,j-1] + 22*vy[i,j]) / (24*dy)
        )
    end

    # top-left
    if i == 2 && j == 2
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (-22*vx[i-1,j] + 17*vx[i,j] + 9*vx[i+1,j] - 5*vx[i+2,j] + vx[i+3,j]) / (24*dx) + 
            (-22*vy[i,j-1] + 17*vy[i,j] + 9*vy[i,j+1] - 5*vy[i,j+2] + vy[i,j+3]) / (24*dy)
        )
    end

    # top-right
    if i == 2 && j == Ny-1
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (-22*vx[i-1,j] + 17*vx[i,j] + 9*vx[i+1,j] - 5*vx[i+2,j] + vx[i+3,j]) / (24*dx) + 
            (-1*vy[i,j-4] + 5*vy[i,j-3] - 9*vy[i,j-2] - 17*vy[i,j-1] + 22*vy[i,j]) / (24*dy)
        )
    end

    # bottom-left
    if i == Nx-1 && j == 2
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (-1*vx[i-4,j] + 5*vx[i-3,j] - 9*vx[i-2,j] - 17*vx[i-1,j] + 22*vx[i,j]) / (24*dx) +
            (-22*vy[i,j-1] + 17*vy[i,j] + 9*vy[i,j+1] - 5*vy[i,j+2] + vy[i,j+3]) / (24*dy)
        )
    end

    # bottom-right
    if i == Nx-1 && j == Ny-1
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (-1*vx[i-4,j] + 5*vx[i-3,j] - 9*vx[i-2,j] - 17*vx[i-1,j] + 22*vx[i,j]) / (24*dx) +
            (-1*vy[i,j-4] + 5*vy[i,j-3] - 9*vy[i,j-2] - 17*vy[i,j-1] + 22*vy[i,j]) / (24*dy)
        )
    end

    return nothing
end

# ==============================
# PML 2ND ORDER
# ==============================

function update_pressure_pml_2nd!(u, vx, vy, wx, wy, sigma_x, sigma_y, b, dx, dy, dt, Nx, Ny)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    # center
    if i > 1 && i < Nx && j > 1 && j < Ny
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (vx[i,j] - vx[i-1,j]) / (dx) + 
            (vy[i,j] - vy[i,j-1]) / (dy)
        ) - 
        dt * (sigma_x[i,j] * u[i,j] + sigma_y[i,j] * u[i,j]) + 
        dt * b[i,j] * (wx[i,j] + wy[i,j])
    end

    return nothing
end

function update_velocity_pml_2nd!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx, Ny)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i >= 1 && i <= Nx-1 && j >= 1 && j <= Ny
        @inbounds vx[i,j] = vx[i,j] + dt * a_x[i,j] * (u[i+1,j] - u[i,j]) / (dx) - dt * sigma_x_half[i,j] * vx[i,j]
    end

    if i >= 1 && i <= Nx && j >= 1 && j <= Ny-1
        @inbounds vy[i,j] = vy[i,j] + dt * a_y[i,j] * (u[i,j+1] - u[i,j]) / (dy) - dt * sigma_y_half[i,j] * vy[i,j]
    end

    return nothing
end

function update_auxiliary_pml_2nd!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx, Ny)
    
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    # center
    if i > 1 && i <= Nx-1 && j >= 1 && j <= Ny
        @inbounds wx[i,j] = wx[i,j] + dt / dx * sigma_y[i,j] * (vx[i,j] - vx[i-1,j])
    end
    if i >= 1 && i <= Nx && j > 1 && j <= Ny-1
        @inbounds wy[i,j] = wy[i,j] + dt / dy * sigma_x[i,j] * (vy[i,j] - vy[i,j-1])
    end

    return nothing
end

function update_pressure_2ndx!(u, vx, vy, wx, wy, sigma_x, sigma_y, b, dx, dy, dt, Nx, Ny)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    # center
    if i > 1 && i < Nx && j > 1 && j < Ny
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (vx[i,j] - vx[i-1,j]) / (dx) + 
            (vy[i,j] - vy[i,j-1]) / (dy)
        ) - 
        dt * (sigma_x[i,j] * u[i,j] + sigma_y[i,j] * u[i,j]) + 
        dt * b[i,j] * (wx[i,j] + wy[i,j])
    end

    return nothing
end

function update_velocity_2ndx!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx, Ny)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i >= 1 && i <= Nx-1 && j >= 1 && j <= Ny
        @inbounds vx[i,j] = vx[i,j] + dt * a_x[i,j] * (u[i+1,j] - u[i,j]) / (dx) - dt * sigma_x_half[i,j] * vx[i,j]
    end

    if i >= 1 && i <= Nx && j >= 1 && j <= Ny-1
        @inbounds vy[i,j] = vy[i,j] + dt * a_y[i,j] * (u[i,j+1] - u[i,j]) / (dy) - dt * sigma_y_half[i,j] * vy[i,j]
    end

    return nothing
end

function update_auxiliary_2ndx!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx, Ny)
    
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    # center
    if i > 1 && i <= Nx-1 && j >= 1 && j <= Ny
        @inbounds wx[i,j] = wx[i,j] + dt / dx * sigma_y[i,j] * (vx[i,j] - vx[i-1,j])
    end
    if i >= 1 && i <= Nx && j > 1 && j <= Ny-1
        @inbounds wy[i,j] = wy[i,j] + dt / dy * sigma_x[i,j] * (vy[i,j] - vy[i,j-1])
    end

    return nothing
end

# ==============================
# PML 4TH ORDER
# ==============================

function update_pressure_pml_4th!(u, vx, vy, wx, wy, sigma_x, sigma_y, b, dx, dy, dt, Nx, Ny)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    # center
    if i > 2 && i < Nx-1 && j > 2 && j < Ny-1
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (vx[i-2,j] - 27*vx[i-1,j] + 27*vx[i,j] - vx[i+1,j]) / (24*dx) + 
            (vy[i,j-2] - 27*vy[i,j-1] + 27*vy[i,j] - vy[i,j+1]) / (24*dy) +
            wx[i,j] + wy[i,j]
        ) - dt * (sigma_x[i,j] * u[i,j] + sigma_y[i,j] * u[i,j])        
    end

    # top
    if i == 2 && j > 2 && j < Ny-1
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (-22*vx[i-1,j] + 17*vx[i,j] + 9*vx[i+1,j] - 5*vx[i+2,j] + vx[i+3,j]) / (24*dx) + 
            (vy[i,j-2] - 27*vy[i,j-1] + 27*vy[i,j] - vy[i,j+1]) / (24*dy) +
            wx[i,j] + wy[i,j]
        ) - dt * (sigma_x[i,j] * u[i,j] + sigma_y[i,j] * u[i,j])
    end

    # bottom
    if i == Nx-1 && j > 2 && j < Ny-1
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (-1*vx[i-4,j] + 5*vx[i-3,j] - 9*vx[i-2,j] - 17*vx[i-1,j] + 22*vx[i,j]) / (24*dx) +
            (vy[i,j-2] - 27*vy[i,j-1] + 27*vy[i,j] - vy[i,j+1]) / (24*dy) +
            wx[i,j] + wy[i,j]
        ) - dt * (sigma_x[i,j] * u[i,j] + sigma_y[i,j] * u[i,j])
    end

    # left
    if i > 1 && i < Nx && j == 2
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (vx[i-2,j] - 27*vx[i-1,j] + 27*vx[i,j] - vx[i+1,j]) / (24*dx) + 
            (-22*vy[i,j-1] + 17*vy[i,j] + 9*vy[i,j+1] - 5*vy[i,j+2] + vy[i,j+3]) / (24*dy) +
            wx[i,j] + wy[i,j]
        ) - dt * (sigma_x[i,j] * u[i,j] + sigma_y[i,j] * u[i,j])
    end

    # right
    if i > 1 && i < Nx && j == Ny-1
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (vx[i-2,j] - 27*vx[i-1,j] + 27*vx[i,j] - vx[i+1,j]) / (24*dx) + 
            (-1*vy[i,j-4] + 5*vy[i,j-3] - 9*vy[i,j-2] - 17*vy[i,j-1] + 22*vy[i,j]) / (24*dy) +
            wx[i,j] + wy[i,j]
        ) - dt * (sigma_x[i,j] * u[i,j] + sigma_y[i,j] * u[i,j])
    end

    # top-left
    if i == 2 && j == 2
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (-22*vx[i-1,j] + 17*vx[i,j] + 9*vx[i+1,j] - 5*vx[i+2,j] + vx[i+3,j]) / (24*dx) + 
            (-22*vy[i,j-1] + 17*vy[i,j] + 9*vy[i,j+1] - 5*vy[i,j+2] + vy[i,j+3]) / (24*dy) +
            wx[i,j] + wy[i,j]
        ) - dt * (sigma_x[i,j] * u[i,j] + sigma_y[i,j] * u[i,j])
    end

    # top-right
    if i == 2 && j == Ny-1
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (-22*vx[i-1,j] + 17*vx[i,j] + 9*vx[i+1,j] - 5*vx[i+2,j] + vx[i+3,j]) / (24*dx) + 
            (-1*vy[i,j-4] + 5*vy[i,j-3] - 9*vy[i,j-2] - 17*vy[i,j-1] + 22*vy[i,j]) / (24*dy) +
            wx[i,j] + wy[i,j]
        ) - dt * (sigma_x[i,j] * u[i,j] + sigma_y[i,j] * u[i,j])
    end

    # bottom-left
    if i == Nx-1 && j == 2
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (-1*vx[i-4,j] + 5*vx[i-3,j] - 9*vx[i-2,j] - 17*vx[i-1,j] + 22*vx[i,j]) / (24*dx) +
            (-22*vy[i,j-1] + 17*vy[i,j] + 9*vy[i,j+1] - 5*vy[i,j+2] + vy[i,j+3]) / (24*dy) +
            wx[i,j] + wy[i,j]
        ) - dt * (sigma_x[i,j] * u[i,j] + sigma_y[i,j] * u[i,j])
    end

    # bottom-right
    if i == Nx-1 && j == Ny-1
        @inbounds u[i,j] = u[i,j] + dt * b[i,j] * (
            (-1*vx[i-4,j] + 5*vx[i-3,j] - 9*vx[i-2,j] - 17*vx[i-1,j] + 22*vx[i,j]) / (24*dx) +
            (-1*vy[i,j-4] + 5*vy[i,j-3] - 9*vy[i,j-2] - 17*vy[i,j-1] + 22*vy[i,j]) / (24*dy) +
            wx[i,j] + wy[i,j]
        ) - dt * (sigma_x[i,j] * u[i,j] + sigma_y[i,j] * u[i,j])
    end

    return nothing
end

function update_velocity_pml_4th!(u, vx, vy, sigma_x_half, sigma_y_half, a_x, a_y, dx, dy, dt, Nx, Ny)

    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i >= 2 && i <= Nx-2 && j >= 1 && j <= Ny
        @inbounds vx[i,j] = vx[i,j] + dt * a_x[i,j] * (u[i-1,j] - 27*u[i,j] + 27*u[i+1,j] - u[i+2,j]) / (24*dx) - dt * sigma_x_half[i,j] * vx[i,j]
    end
    if i == 1 && j >= 1 && j <= Ny
        @inbounds vx[i,j] = vx[i,j] + dt * a_x[i,j] * (-22*u[i,j] + 17*u[i+1,j] + 9*u[i+2,j] - 5*u[i+3,j] + u[i+4,j]) / (24*dx) - dt * sigma_x_half[i,j] * vx[i,j]
    end
    if i == Nx-1 && j >= 1 && j <= Ny
        @inbounds vx[i,j] = vx[i,j] + dt * a_x[i,j] * (-1*u[i-3,j] + 5*u[i-2,j] - 9*u[i-1,j] - 17*u[i,j] + 22*u[i+1,j]) / (24*dx) - dt * sigma_x_half[i,j] * vx[i,j]
    end

    if i >= 1 && i <= Nx && j >= 2 && j <= Ny-2
        @inbounds vy[i,j] = vy[i,j] + dt * a_y[i,j] * (u[i,j-1] - 27*u[i,j] + 27*u[i,j+1] - u[i,j+2]) / (24*dy) - dt * sigma_y_half[i,j] * vy[i,j]
    end
    if i >= 1 && i <= Nx && j == 1
        @inbounds vy[i,j] = vy[i,j] + dt * a_y[i,j] * (-22*u[i,j] + 17*u[i,j+1] + 9*u[i,j+2] - 5*u[i,j+3] + u[i,j+4]) / (24*dy) - dt * sigma_y_half[i,j] * vy[i,j]
    end
    if i >= 1 && i <= Nx && j == Ny-1
        @inbounds vy[i,j] = vy[i,j] + dt * a_y[i,j] * (-1*u[i,j-3] + 5*u[i,j-2] - 9*u[i,j-1] - 17*u[i,j] + 22*u[i,j+1]) / (24*dy) - dt * sigma_y_half[i,j] * vy[i,j]
    end

    return nothing
end

"""
    update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx, Ny)

Update the auxiliary fields wx, wy using 4th-order finite differences and PML terms.

Performance suggestions and improvements:
- Use local variables for commonly used expressions and array accesses to reduce repeated memory reads, especially on the GPU.
- Use `@inbounds` around the whole function or at the block level to minimize bounds-check overhead (inside each valid branch).
- Reduce repeated computation of indices.
- Consider reducing register pressure by splitting into two kernel launches if register spilling is severe; here, we stick to a single kernel for clarity.
- Avoid repeated accesses of parameters like dt, dx, dy, Nx, Ny by passing them as constants or ensuring compile-time constness (not shown here—just as a general note).
"""

function update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx, Ny)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    @inbounds begin
        if i > 2 && i < Nx-1 && j > 2 && j < Ny-1
            # Center: all regular 4th-order stencils
            vxm2 = vx[i-2, j]; vxm1 = vx[i-1, j]; vxi = vx[i, j]; vxp1 = vx[i+1, j]
            wx[i,j] += dt * sigma_y[i,j] * (vxm2 - 27vxm1 + 27vxi - vxp1) / (24dx)

            vy_m2 = vy[i, j-2]; vy_m1 = vy[i, j-1]; vy_i = vy[i, j]; vy_p1 = vy[i, j+1]
            wy[i,j] += dt * sigma_x[i,j] * (vy_m2 - 27vy_m1 + 27vy_i - vy_p1) / (24dy)
        elseif i == 2 && j > 2 && j < Ny-1
            # Left edge (excluding corners)
            vxm1 = vx[i-1, j]; vxi = vx[i, j]; vxp1 = vx[i+1, j]; vxp2 = vx[i+2, j]; vxp3 = vx[i+3, j]
            wx[i,j] += dt * sigma_y[i,j] * (-22vxm1 + 17vxi + 9vxp1 - 5vxp2 + vxp3) / (24dx)

            vy_m2 = vy[i, j-2]; vy_m1 = vy[i, j-1]; vy_i = vy[i, j]; vy_p1 = vy[i, j+1]
            wy[i,j] += dt * sigma_x[i,j] * (vy_m2 - 27vy_m1 + 27vy_i - vy_p1) / (24dy)
        elseif i == Nx-1 && j > 2 && j < Ny-1
            # Right edge (excluding corners)
            vxm4 = vx[i-4, j]; vxm3 = vx[i-3, j]; vxm2 = vx[i-2, j]; vxm1 = vx[i-1, j]; vxi = vx[i, j]
            wx[i,j] += dt * sigma_y[i,j] * (-1vxm4 + 5vxm3 - 9vxm2 - 17vxm1 + 22vxi) / (24dx)

            vy_m2 = vy[i, j-2]; vy_m1 = vy[i, j-1]; vy_i = vy[i, j]; vy_p1 = vy[i, j+1]
            wy[i,j] += dt * sigma_x[i,j] * (vy_m2 - 27vy_m1 + 27vy_i - vy_p1) / (24dy)
        elseif i > 2 && i < Nx-1 && j == 2
            # Bottom edge (excluding corners)
            vxm2 = vx[i-2, j]; vxm1 = vx[i-1, j]; vxi = vx[i, j]; vxp1 = vx[i+1, j]
            wx[i,j] += dt * sigma_y[i,j] * (vxm2 - 27vxm1 + 27vxi - vxp1) / (24dx)

            vy_m1 = vy[i, j-1]; vy_i = vy[i, j]; vy_p1 = vy[i, j+1]; vy_p2 = vy[i, j+2]; vy_p3 = vy[i, j+3]
            wy[i,j] += dt * sigma_x[i,j] * (-22vy_m1 + 17vy_i + 9vy_p1 - 5vy_p2 + vy_p3) / (24dy)
        elseif i > 2 && i < Nx-1 && j == Ny-1
            # Top edge (excluding corners)
            vxm2 = vx[i-2, j]; vxm1 = vx[i-1, j]; vxi = vx[i, j]; vxp1 = vx[i+1, j]
            wx[i,j] += dt * sigma_y[i,j] * (vxm2 - 27vxm1 + 27vxi - vxp1) / (24dx)

            vy_m4 = vy[i, j-4]; vy_m3 = vy[i, j-3]; vy_m2 = vy[i, j-2]; vy_m1 = vy[i, j-1]; vy_i = vy[i, j]
            wy[i,j] += dt * sigma_x[i,j] * (-1vy_m4 + 5vy_m3 - 9vy_m2 - 17vy_m1 + 22vy_i) / (24dy)
        elseif i == 2 && j == 2
            # Lower-left corner
            vxm1 = vx[i-1, j]; vxi = vx[i, j]; vxp1 = vx[i+1, j]; vxp2 = vx[i+2, j]; vxp3 = vx[i+3, j]
            wx[i,j] += dt * sigma_y[i,j] * (-22vxm1 + 17vxi + 9vxp1 - 5vxp2 + vxp3) / (24dx)

            vy_m1 = vy[i, j-1]; vy_i = vy[i, j]; vy_p1 = vy[i, j+1]; vy_p2 = vy[i, j+2]; vy_p3 = vy[i, j+3]
            wy[i,j] += dt * sigma_x[i,j] * (-22vy_m1 + 17vy_i + 9vy_p1 - 5vy_p2 + vy_p3) / (24dy)
        elseif i == Nx-1 && j == 2
            # Lower-right corner
            vxm4 = vx[i-4, j]; vxm3 = vx[i-3, j]; vxm2 = vx[i-2, j]; vxm1 = vx[i-1, j]; vxi = vx[i, j]
            wx[i,j] += dt * sigma_y[i,j] * (-1vxm4 + 5vxm3 - 9vxm2 - 17vxm1 + 22vxi) / (24dx)

            vy_m1 = vy[i, j-1]; vy_i = vy[i, j]; vy_p1 = vy[i, j+1]; vy_p2 = vy[i, j+2]; vy_p3 = vy[i, j+3]
            wy[i,j] += dt * sigma_x[i,j] * (-22vy_m1 + 17vy_i + 9vy_p1 - 5vy_p2 + vy_p3) / (24dy)
        elseif i == 2 && j == Ny-1
            # Upper-left corner
            vxm1 = vx[i-1, j]; vxi = vx[i, j]; vxp1 = vx[i+1, j]; vxp2 = vx[i+2, j]; vxp3 = vx[i+3, j]
            wx[i,j] += dt * sigma_y[i,j] * (-22vxm1 + 17vxi + 9vxp1 - 5vxp2 + vxp3) / (24dx)

            vy_m4 = vy[i, j-4]; vy_m3 = vy[i, j-3]; vy_m2 = vy[i, j-2]; vy_m1 = vy[i, j-1]; vy_i = vy[i, j]
            wy[i,j] += dt * sigma_x[i,j] * (-1vy_m4 + 5vy_m3 - 9vy_m2 - 17vy_m1 + 22vy_i) / (24dy)
        elseif i == Nx-1 && j == Ny-1
            # Upper-right corner
            vxm4 = vx[i-4, j]; vxm3 = vx[i-3, j]; vxm2 = vx[i-2, j]; vxm1 = vx[i-1, j]; vxi = vx[i, j]
            wx[i,j] += dt * sigma_y[i,j] * (-1vxm4 + 5vxm3 - 9vxm2 - 17vxm1 + 22vxi) / (24dx)

            vy_m4 = vy[i, j-4]; vy_m3 = vy[i, j-3]; vy_m2 = vy[i, j-2]; vy_m1 = vy[i, j-1]; vy_i = vy[i, j]
            wy[i,j] += dt * sigma_x[i,j] * (-1vy_m4 + 5vy_m3 - 9vy_m2 - 17vy_m1 + 22vy_i) / (24dy)
        end
    end

    return nothing
end

function copy_slice_kernel!(dest, src, slice_idx)
    # CUDA kernel to copy src[:, :] into dest[:, :, slice_idx]
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    Nx, Ny = size(src)
    if i >= 1 && i <= Nx && j >= 1 && j <= Ny
        @inbounds dest[i, j, slice_idx] = src[i, j]
    end

    return nothing
end
