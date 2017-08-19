"""
    funarray(fun, x, dim; axis)

Helper function to execute a function on each column (row if axis==1) and
return the results.
"""
function funarray( fun::Function, x::AbstractArray{Float},
                   dim::Int, axis::Int=2 )::Union{Float,Vector{Float}}
    res = nothing
    if length(size(x))==1
        res = fun(x)
    else
        @assert axis in [1,2] "unknown axis index"
        @assert size(x)[axis==2?1:2]==dim "dimensions don't match"
        if axis==1
            # read by rows
            res = [fun(x[i,:]) for i in 1:size(x)[1]]
        else
            # read by columns
            res = [fun(x[:,i]) for i in 1:size(x)[2]]
        end
    end
    res
end
