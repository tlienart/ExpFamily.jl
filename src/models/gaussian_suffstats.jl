export
    GaussianSuffStats,
    DiagGaussianSuffStats,
    suffstats

immutable GaussianSuffStats <: SuffStats
    ss::Tuple{Vector{Float}, Symmetric{Float, Matrix{Float}}}
    function GaussianSuffStats(s1::Vector{Float}, s2::Matrix{Float})
        @assert length(s1)==size(s2)[1]==size(s2)[2] "Inconsistent dimensions"
        new((s1, Symmetric(s2)))
    end
end

immutable DiagGaussianSuffStats <: SuffStats
    ss::Tuple{Vector{Float}, Vector{Float}}
    function DiagGaussianSuffStats(s1::Vector{Float}, s2::Vector{Float})
        @assert length(s1)==length(s2) "Inconsistent dimensions"
        new((s1, s2))
    end
end

const FGaussSS = GaussianSuffStats
const DGaussSS = DiagGaussianSuffStats
const GaussSS  = Union{FGaussSS, DGaussSS}

getindex(s::GaussSS, ind::Int) = s.ss[ind]

suffstats(::Type{GaussNP}, x::Vector{Float})=GaussianSuffStats(x, 0.5x*x')
suffstats(::Type{GaussMP}, x::Vector{Float})=GaussianSuffStats(x, 0.5x*x')
suffstats(::Type{DGaussNP},x::Vector{Float})=DiagGaussianSuffStats(x, 0.5x.^2)
suffstats(::Type{DGaussMP},x::Vector{Float})=DiagGaussianSuffStats(x, 0.5x.^2)

Base.vec(s::GaussSS)  = vcat(s[1], vec(s[2]))
Base.norm(s::GaussSS) = norm(vec(s))

# The operation that matters the most is the sum and the product for a weighted
# sum (monte carlo estimator)

+(sa::FGaussSS, sb::FGaussSS) = GaussianSuffStats(sa[1]+sb[1], sa[2]+sb[2])
+(sa::DGaussSS, sb::DGaussSS) = DiagGaussianSuffStats(sa[1]+sb[1], sa[2]+sb[2])

*(a::Real, s::FGaussSS) = GaussianSuffStats(a*s[1], a*s[2].data)
*(a::Real, s::DGaussSS) = DiagGaussianSuffStats(a*s[1], a*s[2])

*(s::GaussSS, a::Real) = a*s
/(s::GaussSS, a::Real) = (1.0/a)*s

# Minus may be useful for

-(sa::FGaussSS, sb::FGaussSS) = GaussianSuffStats(sa[1]-sb[1], sa[2]-sb[2])
-(sa::DGaussSS, sb::DGaussSS) = DiagGaussianSuffStats(sa[1]-sb[1], sa[2]-sb[2])
