
# check errors
function check_order!(order)
    if order != 2 && order != 4
        throw(ArgumentError("order has to be 2 or 4."))
    end
    return nothing
end

function check_source_position!(source_position, source_num)

    if source_num == 1
        if typeof(source_position) <: Vector
            source_position = reshape(source_position, 2, 1)
        end
    elseif source_num > 1
        if (source_num, 2) == size(source_position)
            source_position = source_position'
        end
    elseif source_num <= 0
        throw(ArgumentError("source_num has to be > 0"))
    end

end

function check_source_vals!(source_vals, Nt, source_num)

    if (source_num, Nt) == size(source_vals)
        source_vals = source_vals'
    end
    if (Nt, source_num) != size(source_vals)
        throw(ArgumentError("The size of source_vals has to be Nt times source_num"))
    end

end