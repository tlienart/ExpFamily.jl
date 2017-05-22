module ExpFamily

using Compat

# @compat abstract type EFam end

@compat abstract type NatParam end
@compat abstract type MeanParam end

const Float = (Int==Int64) ? Float64 : Float32

export
    # conversion
    natparam,
    meanparam

import Base.+, Base.-, Base.*, Base./, Base.-

include("models/gaussian.jl")

end # module
