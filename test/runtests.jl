using ExpFamily
using Base.Test

@testset "gaussian"     begin include("gaussian_test.jl")       end
@testset "diaggaussian" begin include("diaggaussian_test.jl")   end
