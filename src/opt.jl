using Printf
using JLD2

myReal = Float32

# backtrack linesearch
function linesearch_back(u_bar, u0, f_u0, eval_fn, eta; alpha=1, maxSearch=5)
    u1 = similar(u0)
    for i = 1:maxSearch
        @printf "    Line search time: %1d\n" i
        u1 = u0 + alpha * (u_bar - u0)
        f_u1 = eval_fn(Array{myReal}(u1))
        @printf "    alpha = %1.3e,  f_u0 = %1.5e,  f_u1 = %1.5e\n" alpha f_u0 f_u1
        if f_u1 < f_u0 && (i < maxSearch)
            println("    Line search succeed.")
            break
        elseif f_u1 >= f_u0 && (i == maxSearch)
            u1 = copy(u0)
            println("    Line search failed.")
        else
            alpha = eta * alpha
        end
    end
    return u1
end

# Weak Wolfe-Powell linesearch:
# wwp_c1 = 1e-15
function WWP_linesearch(u_bar, u0, f_u0, grad_u0, eval_grad_handle, eta; wwp_c1=1e-4, wwp_c2=0.9, alpha=1, maxSearch=5)
    u1 = similar(u0)
    d0 = (u_bar - u0)
    for i = 1:maxSearch
        @printf "    Line search time: %1d\n" i
        u1 = u0 + alpha * d0
        f_u1, grad_u1 = eval_grad_handle(Array{myReal}(u1))
        grad_time_d0 = sum(grad_u0 .* d0)
        grad_time_d1 = sum(grad_u1 .* d0)
        @printf "    alpha = %1.5e,  f_u0 = %1.5e,  f_u1 = %1.5e\n" alpha f_u0 f_u1
        if (f_u1 <= f_u0 + wwp_c1 * alpha * grad_time_d0) && (grad_time_d1 >= wwp_c2 * grad_time_d0) && (i < maxSearch)
            println("    Line search succeed.")
            break
        elseif (i == maxSearch)
            u1 = copy(u0)
            println("    Line search failed.")
        else
            alpha = eta * alpha
        end
    end
    return u1
end

# l-BFGS matrix computation
function compute_H_ku(u, S, Y, R, D)
    u = u[:]
    s0 = S[:, end]
    y0 = Y[:, end]
    R_inv = inv(R)
    gamma = (y0'*s0)[1] / (y0'*y0)[1]
    A = [S'; gamma * Y'] * u
    A = [R_inv'*(D+gamma*Y'*Y)*R_inv -R_inv'; -R_inv 0*R] * A
    A = [S gamma * Y] * A
    A = gamma .* u + A
    return A
end

function compute_B_ku(u, S, Y, L, D)
    u = u[:]
    s0 = S[:, end]
    y0 = Y[:, end]
    sigma = (y0'*s0)[1] / (s0'*s0)[1]
    A = [sigma * S'; Y'] * u
    B = [sigma*S'*S L; L' -D]
    A = inv(B) * A
    A = [sigma * S Y] * A
    A = sigma .* u - A
    return A
end

function lbfgs(u0, eval_fn_handle, eval_grad_handle; m=5, iterMax=10, etaBack=0.1, etaWWP=0.9, stepSizeBack=1., stepSizeWWP=1., LinesearchMax=5, wwp_c1=1e-4, wwp_c2=0.9, saveStep=false)

    # Initialization: optimization
    u0 = u0[:]
    n = length(u0)
    @printf "Optimization initializing...\n"
    @printf "    Problem dimension: %d, l-BFGS temporary save step: %d\n" n m
    @printf "    Linesearch backtrack eta = %f, step size = %f, max search time = %d\n" etaBack stepSizeBack LinesearchMax
    @printf "    Linesearch WWP eta = %f, step size = %f, max search time = %d\n" etaWWP stepSizeWWP LinesearchMax
    @printf "    Total iteration time: %d\n" iterMax
    # save name
    save_file_name0 = pwd() * "/temp_data/"
    obj_fn = zeros(iterMax)

    @printf "Preparing optimization...\n"
    # initialization
    S = zeros(myReal, n, m)
    Y = zeros(myReal, n, m)
    R = zeros(myReal, m, m)
    D = zeros(myReal, m, m)
    L = zeros(myReal, m, m)

    @printf "\nOptimization start.\n"
    # iter 1
    iter = 1
    lbfgs_count = 1
    @printf "Iteration: %1d,    temporary save step: %1d\n" iter lbfgs_count
    fn_u0, grad0 = eval_grad_handle(Array{myReal}(u0))
    grad0 = grad0[:]
    u_bar = u0 - grad0
    u1 = linesearch_back(u_bar, u0, fn_u0, eval_fn_handle, etaBack; alpha=stepSizeBack, maxSearch=LinesearchMax)
    if u1 == u0
        println("Line search failed.")
    end
    fn_u1, grad1 = eval_grad_handle(Array{myReal}(u1))
    grad1 = grad1[:]
    # save
    obj_fn[iter] = fn_u0
    if saveStep == true
        save_file_name = save_file_name0 * string(iter) * ".jld2"
        @save save_file_name u1
    end

    for iter = 2:iterMax
        lbfgs_count = min(lbfgs_count, m)
        @printf "\nIteration: %1d,    temporary save step: %1d\n" iter lbfgs_count

        S[:, 1:m-1] = S[:, 2:m]
        S[:, m] = u1 - u0
        Y[:, 1:m-1] = Y[:, 2:m]
        Y[:, m] = grad1 - grad0
        if sum(S[:, m] .* Y[:, m]) < 0
            println("s^T y < 0, break.")
            break
        end
        D[1:m-1, 1:m-1] = D[2:m, 2:m]
        D[m, m] = sum(S[:, m] .* Y[:, m])
        R[1:m-1, 1:m-1] = R[2:m, 2:m]
        for i = 1:m
            R[i, m] = sum(S[:, i] .* Y[:, m])
        end
        L[1:m-1, 1:m-1] = L[2:m, 2:m]
        for i = 1:m-1
            L[m, i] = sum(S[:, m] .* Y[:, i])
        end
        u1_tilde = u1 - compute_H_ku(grad1, S[:, end-lbfgs_count+1:end], Y[:, end-lbfgs_count+1:end], R[end-lbfgs_count+1:end, end-lbfgs_count+1:end], D[end-lbfgs_count+1:end, end-lbfgs_count+1:end])
        u2 = WWP_linesearch(u1_tilde, u1, fn_u1, grad1, eval_grad_handle, etaWWP; wwp_c1=wwp_c1, wwp_c2=wwp_c2, alpha=stepSizeWWP, maxSearch=LinesearchMax)
        if u2 == u1
            @printf "    Linesearch failed. Update with gradient. Reset memory.\n"
            u1_tilde = u1 - grad1
            u2 = linesearch_back(u1_tilde, u1, fn_u1, eval_fn_handle, etaBack; alpha=stepSizeBack, maxSearch=LinesearchMax)
            S = zeros(myReal, n, m)
            Y = zeros(myReal, n, m)
            R = zeros(myReal, m, m)
            D = zeros(myReal, m, m)
            L = zeros(myReal, m, m)
            lbfgs_count = 0
            if u2 == u1
                @printf "Linesearch failed. Break.\n"
                break
            end
        end
        fn_u2, grad2 = eval_grad_handle(Array{myReal}(u2))
        grad2 = grad2[:]

        u0 = copy(u1)
        u1 = copy(u2)
        grad0 = copy(grad1)
        grad1 = copy(grad2)
        fn_u0 = copy(fn_u1)
        fn_u1 = copy(fn_u2)
        lbfgs_count += 1
        # save
        obj_fn[iter] = fn_u1
        if saveStep == true
            save_file_name = save_file_name0 * string(iter) * ".jld2"
            @save save_file_name u1 grad1
        end
    end
    return u1, obj_fn
end

function projBox!(A, lo, hi)
    A[A .< lo] .= lo
    A[A .> hi] .= hi
end


# This is not the actual lbfgsb, but it performs similar in this inverse problem
# using ImageFiltering

function lbfgsb(u0, eval_fn_handle, eval_grad_handle, lo, hi; m=5, iterMax=10, etaBack=0.1, etaWWP=0.9, stepSizeBack=1., stepSizeWWP=1., LinesearchMax=5, wwp_c1=1e-4, wwp_c2=0.9, saveStep=false)

    # gaussCoef = 3

    # Initialization: optimization
    u0 = u0[:]
    n = length(u0)
    @printf "Optimization initializing...\n"
    @printf "    Problem dimension: %d, l-BFGS temporary save step: %d\n" n m
    @printf "    Linesearch backtrack eta = %f, step size = %f, max search time = %d\n" etaBack stepSizeBack LinesearchMax
    @printf "    Linesearch WWP eta = %f, step size = %f, max search time = %d\n" etaWWP stepSizeWWP LinesearchMax
    @printf "    Total iteration time: %d\n" iterMax
    # save name
    save_file_name0 = pwd() * "/temp_data/"
    obj_fn = zeros(iterMax)

    @printf "Preparing optimization...\n"
    # initialization
    S = zeros(myReal, n, m)
    Y = zeros(myReal, n, m)
    R = zeros(myReal, m, m)
    D = zeros(myReal, m, m)
    L = zeros(myReal, m, m)

    @printf "\nOptimization start.\n"
    # iter 1
    iter = 1
    lbfgs_count = 1
    @printf "Iteration: %1d,    temporary save step: %1d\n" iter lbfgs_count
    fn_u0, grad0 = eval_grad_handle(Array{myReal}(u0))
    grad0 = grad0[:]
    # grad0 = imfilter(reshape(grad0, 201, 201), Kernel.gaussian(gaussCoef))[:]
    u_bar = u0 - grad0
    u1 = linesearch_back(u_bar, u0, fn_u0, eval_fn_handle, etaBack; alpha=stepSizeBack, maxSearch=LinesearchMax)
    projBox!(u1, lo, hi)
    fn_u1, grad1 = eval_grad_handle(Array{myReal}(u1))
    grad1 = grad1[:]
    # grad1 = imfilter(reshape(grad1, 201, 201), Kernel.gaussian(gaussCoef))[:]
    # save
    obj_fn[iter] = fn_u0
    if saveStep == true
        save_file_name = save_file_name0 * string(iter) * ".jld2"
        @save save_file_name u1
    end

    for iter = 2:iterMax
        lbfgs_count = min(lbfgs_count, m)
        @printf "\nIteration: %1d,    temporary save step: %1d\n" iter lbfgs_count

        S[:, 1:m-1] = S[:, 2:m]
        S[:, m] = u1 - u0
        Y[:, 1:m-1] = Y[:, 2:m]
        Y[:, m] = grad1 - grad0
        if sum(S[:, m] .* Y[:, m]) < 0
            println("s^T y < 0, break.")
            break
        end
        D[1:m-1, 1:m-1] = D[2:m, 2:m]
        D[m, m] = sum(S[:, m] .* Y[:, m])
        R[1:m-1, 1:m-1] = R[2:m, 2:m]
        for i = 1:m
            R[i, m] = sum(S[:, i] .* Y[:, m])
        end
        L[1:m-1, 1:m-1] = L[2:m, 2:m]
        for i = 1:m-1
            L[m, i] = sum(S[:, m] .* Y[:, i])
        end
        u1_tilde = u1 - compute_H_ku(grad1, S[:, end-lbfgs_count+1:end], Y[:, end-lbfgs_count+1:end], R[end-lbfgs_count+1:end, end-lbfgs_count+1:end], D[end-lbfgs_count+1:end, end-lbfgs_count+1:end])
        u2 = WWP_linesearch(u1_tilde, u1, fn_u1, grad1, eval_grad_handle, etaWWP; wwp_c1=wwp_c1, wwp_c2=wwp_c2, alpha=stepSizeWWP, maxSearch=LinesearchMax)
        projBox!(u2, lo, hi)
        if u2 == u1
            @printf "    Linesearch failed. Update with gradient. Reset memory.\n"
            u1_tilde = u1 - grad1
            u2 = linesearch_back(u1_tilde, u1, fn_u1, eval_fn_handle, etaBack; alpha=stepSizeBack, maxSearch=LinesearchMax)
            projBox!(u2, lo, hi)
            S = zeros(myReal, n, m)
            Y = zeros(myReal, n, m)
            R = zeros(myReal, m, m)
            D = zeros(myReal, m, m)
            L = zeros(myReal, m, m)
            lbfgs_count = 0
            if u2 == u1
                @printf "Linesearch failed. Break.\n"
                break
            end
        end
        fn_u2, grad2 = eval_grad_handle(Array{myReal}(u2))
        grad2 = grad2[:]
        # grad2 = imfilter(reshape(grad2, 201, 201), Kernel.gaussian(gaussCoef))[:]

        u0 = copy(u1)
        u1 = copy(u2)
        grad0 = copy(grad1)
        grad1 = copy(grad2)
        fn_u0 = copy(fn_u1)
        fn_u1 = copy(fn_u2)
        lbfgs_count += 1
        # save
        obj_fn[iter] = fn_u1
        if saveStep == true
            save_file_name = save_file_name0 * string(iter) * ".jld2"
            @save save_file_name u1 grad1
        end
    end
    return u1, obj_fn
end