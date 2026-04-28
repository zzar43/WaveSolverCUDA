function eval_theta(x, a, b, nu)

    n = length(x)
    m = length(a)
    res = 0
    count_zero = 0

    for i in 1:n
        res0 = []
        for j in 1:m
            if x[i] < a[j]
                push!(res0, 1/2 * (x[i] - a[j])^2 * nu)
            elseif x[i] > b[j]
                push!(res0, 1/2 * (x[i] - b[j])^2 * nu)
            else
                push!(res0, 0)
                count_zero += 1
            end
        end
        res += minimum(res0)
    end

    return res
end

function eval_prox_op_single_entry(v, a, b, nu, lamb)
    min_h = 0
    argmin_h = 0
   
    if v >= a && v <= b
        min_h = 0
        argmin_h = v
        return argmin_h, min_h
    elseif v < a
        min_h = nu / 2 * (v - a)^2 / (nu * lamb + 1)
        argmin_h = (nu * lamb * a + v) / (nu * lamb + 1)
    elseif v > b
        min_h = nu / 2 * (v - b)^2 / (nu * lamb + 1)
        argmin_h = (nu * lamb * b + v) / (nu * lamb + 1)
    end

    return argmin_h, min_h
end

function eval_prox_op(v::Real, a::Vector, b::Vector, nu, lamb)

    argmin_h = zeros(length(a))
    min_h = zeros(length(a))

    for i in 1:length(a)
        argmin_h[i], min_h[i] = eval_prox_op_single_entry(v, a[i], b[i], nu, lamb)
    end

    return argmin_h[argmin(min_h)]
end

function eval_prox_op(v::Vector, a::Vector, b::Vector, nu, lamb)

    res = zeros(length(v))

    for i in 1:length(v)
        res[i] = eval_prox_op(v[i], a, b, nu, lamb)
    end

    return res
end

using Printf
function prox_grad_method(eval_fn, eval_grad, eval_prox_op, x0, lamb; max_iter=10, beta=0.9, ls_time = 5, saveStep=false)

    save_file_name0 = pwd() * "/temp_data/"

    v = 0 .* x0
    x = copy(x0)
    lamb0 = copy(lamb)

    for iter in 1:max_iter

        fn0, grad = eval_grad(x)
        fn0 = eval_fn(x)
        @printf "\nIteration: %1d,    obj fn value: %f\n" iter fn0

        # turn off for lambda reuse
        # lamb = copy(lamb0)

        v = x - lamb * grad[:]
        x_prox = eval_prox_op(v, lamb)

        # lamb = beta * lamb

        # linesearch
        fn = eval_fn(x_prox)
        @printf "    Step size: %f, Obj fn after prox update: %f\n" lamb fn
        ls_status = false
        if fn >= fn0
            for iter_ls in 1:ls_time
                lamb = beta * lamb
                v = x - lamb * grad[:]
                x_prox = eval_prox_op(v, lamb)
                fn = eval_fn(x_prox)
                @printf "    New step size: %f, obj fn after prox update: %f\n" lamb fn
                if fn < fn0
                    ls_status = true
                    break
                end
            end
        else
            ls_status = true
            @printf "    Line search is not required.\n"
        end

        if ls_status == false
            @printf "    Line search failed. Break."
            break
        end

        x = copy(x_prox)

        if saveStep == true
            save_file_name = save_file_name0 * string(iter) * ".jld2"
            @save save_file_name x grad
        end

    end

    return x
end