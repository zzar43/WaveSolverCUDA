using BenchmarkTools, JLD2, CairoMakie

include("src/adjoint.jl")
include("src/forward.jl")

demo = 2

try
    readdir("data/adjoint_demo/")
catch err
    mkdir("data/adjoint_demo")
end

if demo == 1
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

    val = maximum(data) * 0.1
    for idx = 1:source_num
        u = data[:,:,idx]'
        fig2 = Figure(size = (1200, 800))
        ax2 = Axis(fig2[1, 1], aspect = 1, yreversed=true)
        hm2 = heatmap!(u, colormap=:bwr, colorrange=(-val, val))
        Colorbar(fig2[1, 2], hm2)
        save("data/adjoint_demo/demo1_source$idx.png", fig2)
    end
    
    println("    Computing adjoint...")
    CUDA.@time gg = adjoint_c(data, c0, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16)
    println("    Done")
    
    img = Array{myReal}(gg)
    val = maximum(img)
    fig1 = Figure(size=(1200,800))
    ax1 = Axis(fig1[1,1], aspect=DataAspect(), yreversed=true)
    hm1 = heatmap!(img, colormap=:bwr, colorrange=(-val,val))
    Colorbar(fig1[1,2], hm1)
    save("data/adjoint_demo/demo1_adjoint.png", fig1)
else
    using ImageFiltering

    # DEMO 2: Marmousi
    @load "data/marmousi/marmousi20.jld2"
    c = 1000 .* c
    rho = 1000 .* rho
    c0 = imfilter(c, Kernel.gaussian(10));
    c0[1:23,:] .= 1500.

    Nx, Ny = size(c)
    dx, dy = 20, 20
    Nt = 4000
    Fs = 500
    dt = 1/Fs
    t = range(0, (Nt-1)*dt, Nt)
    pml_len = 100
    pml_coef = 200

    a = 1 ./ rho
    b = rho .* c .^ 2;

    source_num = 15
    source_position = zeros(2,source_num)
    for i = 1:source_num
        source_position[1,i] = 5
        source_position[2,i] = 51 + (i-1)*51
    end
    source_vals = zeros(Nt, source_num)
    for i = 1:source_num
        source_vals[:,i] = source_ricker_int(5,0.2,t) * 1e6
    end

    receiver_num = 425 * 2
    receiver_position = zeros(2, receiver_num)
    for i = 1:receiver_num
        receiver_position[1,i] = 1
        receiver_position[2,i] = (i-1)*1 + 1
    end

    println("    Computing data...")
    CUDA.@time data = forward_acoustic_c(c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, idx_source=0, recordWaveField=false, saveRatio=1)
    println("    Done.")

    val = maximum(data) * 0.05
    for idx = 1:source_num
        u = data[:,:,idx]'
        fig2 = Figure(size = (1200, 800))
        ax2 = Axis(fig2[1, 1], aspect = 1, yreversed=true)
        hm2 = heatmap!(u, colormap=:bwr, colorrange=(-val, val))
        Colorbar(fig2[1, 2], hm2)
        save("data/adjoint_demo/demo2_source$idx.png", fig2)
    end

    println("    Computing adjoint...")
    CUDA.@time gg = adjoint_c(data, c0, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16)
    println("    Done.")

    img = Array{myReal}(gg)'
    val = maximum(img)
    fig1 = Figure(size=(1200,800))
    ax1 = Axis(fig1[1,1], aspect=DataAspect(), yreversed=true)
    hm1 = heatmap!(img, colormap=:bwr, colorrange=(-val,val))
    Colorbar(fig1[1,2], hm1)
    save("data/adjoint_demo/demo2_adjoint.png", fig1)
end

