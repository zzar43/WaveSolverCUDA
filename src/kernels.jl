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
        end
    end

    return nothing
end

# ==============================
# RECEIVER
# ==============================

function record_wavefield!(u, receiver_position_x, receiver_position_y, receiver_vals, receiver_num, idx_time)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i <= receiver_num
        ix, iy = receiver_position_x[i], receiver_position_y[i]
        if ix >= 1 && ix <= size(u,1) && iy >= 1 && iy <= size(u,2)
            receiver_vals[idx_time, i] = u[ix, iy]
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

function update_auxiliary_pml_4th!(wx, wy, vx, vy, sigma_x, sigma_y, dx, dy, dt, Nx, Ny)
    
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    # center
    if i > 2 && i < Nx-1 && j > 2 && j < Ny-1
        @inbounds wx[i,j] = wx[i,j] + dt * sigma_y[i,j] * (vx[i-2,j] - 27*vx[i-1,j] + 27*vx[i,j] - vx[i+1,j]) / (24*dx)
        @inbounds wy[i,j] = wy[i,j] + dt * sigma_x[i,j] * (vy[i,j-2] - 27*vy[i,j-1] + 27*vy[i,j] - vy[i,j+1]) / (24*dy)
    end

    # Boundary points (one-sided differences)
    if i == 2 && j > 2 && j < Ny-1
        @inbounds wx[i,j] = wx[i,j] + dt * sigma_y[i,j] * (-22*vx[i-1,j] + 17*vx[i,j] + 9*vx[i+1,j] - 5*vx[i+2,j] + vx[i+3,j]) / (24*dx)
        @inbounds wy[i,j] = wy[i,j] + dt * sigma_x[i,j] * (vy[i,j-2] - 27*vy[i,j-1] + 27*vy[i,j] - vy[i,j+1]) / (24*dy)
    end

    if i == Nx-1 && j > 2 && j < Ny-1
        @inbounds wx[i,j] = wx[i,j] + dt * sigma_y[i,j] * (-1*vx[i-4,j] + 5*vx[i-3,j] - 9*vx[i-2,j] - 17*vx[i-1,j] + 22*vx[i,j]) / (24*dx)
        @inbounds wy[i,j] = wy[i,j] + dt * sigma_x[i,j] * (vy[i,j-2] - 27*vy[i,j-1] + 27*vy[i,j] - vy[i,j+1]) / (24*dy)
    end

    if i > 2 && i < Nx-1 && j == 2
        @inbounds wx[i,j] = wx[i,j] + dt * sigma_y[i,j] * (vx[i-2,j] - 27*vx[i-1,j] + 27*vx[i,j] - vx[i+1,j]) / (24*dx)
        @inbounds wy[i,j] = wy[i,j] + dt * sigma_x[i,j] * (-22*vy[i,j-1] + 17*vy[i,j] + 9*vy[i,j+1] - 5*vy[i,j+2] + vy[i,j+3]) / (24*dy)
    end

    if i > 2 && i < Nx-1 && j == Ny-1
        @inbounds wx[i,j] = wx[i,j] + dt * sigma_y[i,j] * (vx[i-2,j] - 27*vx[i-1,j] + 27*vx[i,j] - vx[i+1,j]) / (24*dx)
        @inbounds wy[i,j] = wy[i,j] + dt * sigma_x[i,j] * (-1*vy[i,j-4] + 5*vy[i,j-3] - 9*vy[i,j-2] - 17*vy[i,j-1] + 22*vy[i,j]) / (24*dy)
    end

    # four points
    if i == 2 && j == 2
        @inbounds wx[i,j] = wx[i,j] + dt * sigma_y[i,j] * (-22*vx[i-1,j] + 17*vx[i,j] + 9*vx[i+1,j] - 5*vx[i+2,j] + vx[i+3,j]) / (24*dx)
        @inbounds wy[i,j] = wy[i,j] + dt * sigma_x[i,j] * (-22*vy[i,j-1] + 17*vy[i,j] + 9*vy[i,j+1] - 5*vy[i,j+2] + vy[i,j+3]) / (24*dy)
    end

    if i == Nx-1 && j == 2
        @inbounds wx[i,j] = wx[i,j] + dt * sigma_y[i,j] * (-1*vx[i-4,j] + 5*vx[i-3,j] - 9*vx[i-2,j] - 17*vx[i-1,j] + 22*vx[i,j]) / (24*dx)
        @inbounds wy[i,j] = wy[i,j] + dt * sigma_x[i,j] * (-22*vy[i,j-1] + 17*vy[i,j] + 9*vy[i,j+1] - 5*vy[i,j+2] + vy[i,j+3]) / (24*dy)
    end

    if i == 2 && j == Ny-1
        @inbounds wx[i,j] = wx[i,j] + dt * sigma_y[i,j] * (-22*vx[i-1,j] + 17*vx[i,j] + 9*vx[i+1,j] - 5*vx[i+2,j] + vx[i+3,j]) / (24*dx)
        @inbounds wy[i,j] = wy[i,j] + dt * sigma_x[i,j] * (-1*vy[i,j-4] + 5*vy[i,j-3] - 9*vy[i,j-2] - 17*vy[i,j-1] + 22*vy[i,j]) / (24*dy)
    end

    if i == Nx-1 && j == Ny-1
        @inbounds wx[i,j] = wx[i,j] + dt * sigma_y[i,j] * (-1*vx[i-4,j] + 5*vx[i-3,j] - 9*vx[i-2,j] - 17*vx[i-1,j] + 22*vx[i,j]) / (24*dx)
        @inbounds wy[i,j] = wy[i,j] + dt * sigma_x[i,j] * (-1*vy[i,j-4] + 5*vy[i,j-3] - 9*vy[i,j-2] - 17*vy[i,j-1] + 22*vy[i,j]) / (24*dy)
    end

    return nothing 
end
