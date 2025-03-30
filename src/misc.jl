# check errors
function check_order(order)
    if order != 2 && order != 4
        # throw(ArgumentError("order has to be 2 or 4."))
        error("order has to be 2 or 4.")
    end
    return nothing
end

function check_source_position(source_position, source_num)

    if size(source_position) == (2, source_num)
        return source_position
    elseif size(source_position) == (source_num, 2)
        return transpose(source_position)
    else
        error("source_position has to be in the shape of (2, source_num)")
    end

end

function check_receiver_position(receiver_position, receiver_num)

    if size(receiver_position) == (2, receiver_num)
        return receiver_position
    elseif size(receiver_position) == (receiver_num, 2)
        return transpose(receiver_position)
    else
        error("receiver_position has to be in the shape of (2, receiver_num)")
    end

end

function check_source_vals(source_vals, Nt, source_num)

    if size(source_vals) == (Nt, source_num)
        return source_vals
    elseif size(source_vals) == (source_num, Nt)
        return transpose(source_vals)
    else
        error("source_vals has to be in the shape of (Nt, source_num)")
    end

end
