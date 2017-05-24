module ExpFamily

using Compat

# @compat abstract type EFam end

@compat abstract type EFamily end
@compat abstract type NatParam  <: EFamily end
@compat abstract type MeanParam <: EFamily end

const Float = Float64

export
    natparam,
    meanparam,
    suffstats,
    project

import Base.+, Base.-, Base.*, Base./, Base.-

include("models/gaussian.jl")
include("models/gaussian_ops.jl")

end # module
