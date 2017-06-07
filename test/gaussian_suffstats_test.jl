using ExpFamily
using Base.Test

x = randn(10)

ssNP  = suffstats(GaussianNatParam, x)
ssMP  = suffstats(GaussianMeanParam, x)
ssdNP = suffstats(DiagGaussianNatParam, x)
ssdMP = suffstats(DiagGaussianMeanParam, x)

ssNP2   = 2*ssNP
ssdNP2  = ssdNP*2
ssNP2b  = ssNP + ssNP
ssdNP2b = ssdNP + ssdNP

@test isapprox(ssNP[1],   x)
@test isapprox(ssMP[2],   (x*x')/2)
@test isapprox(ssMP[1],   x)
@test isapprox(ssMP[2],   (x*x')/2)
@test isapprox(ssdNP[1],  x)
@test isapprox(ssdNP[2],  (x.^2)/2)
@test isapprox(ssdMP[1],  x)
@test isapprox(ssdMP[2],  (x.^2)/2)
@test isapprox(ssNP2[1],  2x)
@test isapprox(ssNP2[2],  (x*x'))
@test isapprox(ssdNP2[1], 2x)
@test isapprox(ssdNP2[2], x.^2)
@test isapprox(ssNP2[1],  ssNP2b[1])
@test isapprox(ssNP2[2],  ssNP2b[2])
@test isapprox(ssdNP2[1], ssdNP2b[1])
@test isapprox(ssdNP2[2], ssdNP2b[2])

@test isapprox( vec(ssNP),              vcat(x,vec(x*x'/2)))
@test isapprox( norm(ssNP),             norm(vcat(x,vec(x*x'/2))))
@test isapprox( vec(2*ssNP-ssNP),       vec(ssNP))
@test isapprox( vec((2*ssNP)/2),        vec(ssNP))
@test isapprox( vec((5*ssdNP-ssdNP)/4), vec(ssdNP))
