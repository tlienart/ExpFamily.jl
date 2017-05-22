module ExpFamily

using Compat

# @compat abstract type EFam end

@compat abstract type NatParam end
@compat abstract type MeanParam end

const Float = (Int==Int64) ? Float64 : Float32

export
    natparam,
    meanparam

import Base.+, Base.-

include("models/gaussian.jl")

end # module
