using ExpFamily
using Base.Test

C  = [1.0 0.5; 0.5 1.0]
P  = 2/3 * [2 -1; -1 2]
m  = [1.0;-0.5]

gNP = GaussianNatParam(mu=m, cov=C)
gMP = GaussianMeanParam(mu=m, cov=C)

@test isapprox(gNP.theta1, P*m)
@test isapprox(gNP.theta2, -P)
@test isapprox(gMP.mu1, m)
@test isapprox(gMP.mu2, (m*m'+C)/2)

@test isvalid(gNP)
@test isvalid(gMP)

mpFromNP = meanparam(gNP)
npFromMP = natparam(gMP)

@test isapprox(mpFromNP.mu1, gMP.mu1)
@test isapprox(mpFromNP.mu2, gMP.mu2)

@test isapprox(mean(gNP), m)
@test isapprox(mean(gMP), m)
@test isapprox(cov(gNP), C)
@test isapprox(cov(gMP), C)

@test isapprox(var(gMP), diag(C))
@test isapprox(var(gNP), diag(C))

@test_throws DomainError GaussianNatParam(mu=m, cov=-C, check=true)
@test_throws DomainError GaussianMeanParam(mu=m, cov=-C, check=true)

gNP2 = gNP+gNP

@test isapprox(gNP2.theta1, gNP.theta1+gNP.theta1)
@test isapprox(gNP2.theta2, gNP.theta2+gNP.theta2)
