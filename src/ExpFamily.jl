module ExpFamily

using Compat

# @compat abstract type EFam end

@compat abstract type EFamily end
@compat abstract type NatParam  <: EFamily end
@compat abstract type MeanParam <: EFamily end

@compat abstract type SuffStats end

const Float = Float64

export
    EFamily,
    natparam,
    meanparam,
    suffstats,
    project,
    loglik,
    gradloglik

import Base.+, Base.-, Base.*, Base./, Base.-

include("models/gaussian.jl")
include("models/gaussian_ops.jl")
include("models/gaussian_suffstats.jl")

end # module
