module ExpFamily

using Compat

# @compat abstract type EFam end

@compat abstract type NatParam end
@compat abstract type MeanParam end

const Int   = Int64
const Float = Float64

export
    natparam,
    meanparam

import Base.+, Base.-

include("models/gaussian.jl")

end # module
