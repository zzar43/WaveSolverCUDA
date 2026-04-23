using LinearAlgebra

include("solver.jl")

function eval_obj_fwi(data_true, c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16)

    c = reshape(c, Nx, Ny)

    data = forward_acoustic_c(c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=blockx, blocky=blocky)

    return 1/2 * norm(data - data_true)^2

end


function eval_grad_fwi(data_true, c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16, cutoff=0)

    c = reshape(c, Nx, Ny)

    data = forward_acoustic_c(c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=16, blocky=16)

    grad = adjoint_method_c(data_true, c, Nx, Ny, Nt, dx, dy, dt, source_num, source_position, source_vals, receiver_num, receiver_position, pml_len, pml_coef; blockx=blockx, blocky=blocky)

    if cutoff > 0
        grad[1:cutoff, :] .= 0
    end

    return 1/2 * norm(data - data_true)^2, grad[:]

end