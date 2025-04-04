# This is a demo for forward modeling.

include("src/solver.jl")
include("src/forward.jl")

using BenchmarkTools, CairoMakie, JLD2

demo = 2

try
    readdir("data/forward_demo/")
catch err
    mkdir("data/forward_demo")
end

if demo == 1
    # DEMO 1: square domain
    Nx = 801
    Ny = 801
    dx = 5
    dy = 5
    Nt = 2000
    Fs = 500
    dt = 1/Fs
    t = range(0, (Nt-1)*dt, Nt)
    pml_len = 100
    pml_coef = 100

    rho = 1 .* ones(myReal, Nx, Ny)
    c = 1000 .* ones(myReal, Nx, Ny)
    c[1:301,1:301] .= 1200
    # a = 1 ./ rho
    # b = rho .* c .^ 2;

    source_num = 8
    source_position = zeros(2,source_num)
    for i = 1:source_num
        source_position[1,i] = 401
        source_position[2,i] = 1 + (i-1)*101
    end
    source_vals = zeros(Nt, source_num)
    for i = 1:source_num
        source_vals[:,i] = source_ricker_int(12,0.2,t) * 1e6
    end

    receiver_num = 401
    receiver_position = zeros(2, receiver_num)
    for i = 1:receiver_num
        receiver_position[1,i] = 51
        receiver_position[2,i] = (i-1)*2 + 1
    end

    println("    Computing wavefield...")
    @time data, U = forward_acoustic_c(c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, idx_source=5, recordWaveField=true)
    println("    Done.")

    # record movie
    val = 0.1 * maximum(U)
    fig1, ax1, hm1 = heatmap(Array(U[:,:,1]), colormap=:seismic, axis=(;aspect=DataAspect(),yreversed=true), colorrange=(-val, val), figure=(size=(1200,800),))
    CairoMakie.record(fig1, "data/forward_demo/demo1_wavefield.mp4", 1:10:Nt) do i
        hm1[1] = Array(U[:,:,i])
    end

    println("    Computing forward modeling...")
    @time data = forward_acoustic_c(c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, idx_source=0, recordWaveField=false)
    println("    Done.")

    # record data
    for idx = 1:source_num
        u = data[:,:,idx]'
        fig2 = Figure(size = (1200, 800))
        ax2 = Axis(fig2[1, 1], aspect = 1.5, yreversed=true)
        hm2 = heatmap!(u, colormap=:seismic, colorrange=(-val, val))
        Colorbar(fig2[1, 2], hm2)
        save("data/forward_demo/demo1_source$idx.png", fig2)
    end
end

if demo == 2
    # DEMO 2: Marmousi
    @load "data/marmousi/marmousi10.jld2"
    c = 1000 .* c
    rho = 1000 .* rho

    Nx, Ny = size(c)
    dx, dy = 10, 10
    Nt = 8000
    Fs = 1000
    dt = 1/Fs
    t = range(0, (Nt-1)*dt, Nt)
    pml_len = 100
    pml_coef = 100

    a = 1 ./ rho
    b = rho .* c .^ 2;

    source_num = 15
    source_position = zeros(2,source_num)
    for i = 1:source_num
        source_position[1,i] = 5
        source_position[2,i] = 101 + (i-1)*101
    end
    source_vals = zeros(Nt, source_num)
    for i = 1:source_num
        source_vals[:,i] = source_ricker_int(10,0.2,t) * 1e0
    end

    receiver_num = 425 * 2
    receiver_position = zeros(2, receiver_num)
    for i = 1:receiver_num
        receiver_position[1,i] = 1
        receiver_position[2,i] = (i-1)*2 + 1
    end
    println("    Computing wavefield...")
    @time data, U = forward_acoustic(a, b, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, idx_source=12, recordWaveField=true, saveRatio=5)
    println("    Done.")

    val = 0.02 * maximum(U)
    fig3, ax3, hm3 = heatmap(Array(U[:,:,1])', colormap=:seismic, axis=(;aspect=DataAspect(),yreversed=true), colorrange=(-val, val), figure=(size=(1200,800),))
    CairoMakie.record(fig3, "data/forward_demo/demo2_wavefield.mp4", 1:5:size(U,3)) do i
        hm3[1] = Array(U[:,:,i])'
    end

    println("    Computing forward modeling...")
    @time data = forward_acoustic(a, b, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, idx_source=0, recordWaveField=false)
    println("    Done.")

    for idx = 1:source_num
        u = data[:,:,idx]'
        fig4 = Figure(size = (1200, 800))
        ax4 = Axis(fig4[1, 1], aspect = 1.5, yreversed=true)
        hm4 = heatmap!(u, colormap=:seismic, colorrange=(-val, val))
        Colorbar(fig4[1, 2], hm4)
        save("data/forward_demo/demo2_source$idx.png", fig4)
    end
end
