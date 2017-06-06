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

suffstats(::Type{GaussNP}, x::Vector{Float})=GaussianSuffStats(x, 0.5x*x')
suffstats(::Type{GaussMP}, x::Vector{Float})=GaussianSuffStats(x, 0.5x*x')
suffstats(::Type{DGaussNP},x::Vector{Float})=DiagGaussianSuffStats(x, 0.5x.^2)
suffstats(::Type{DGaussMP},x::Vector{Float})=DiagGaussianSuffStats(x, 0.5x.^2)

+(sa::FGaussSS, sb::FGaussSS) = GaussianSuffStats(sa[1]+sb[1], sa[2]+sb[2])
+(sa::DGaussSS, sb::DGaussSS) = DiagGaussianSuffStats(sa[1]+sb[1], sa[2]+sb[2])
