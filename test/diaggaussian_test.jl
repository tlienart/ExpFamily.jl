using ExpFamily
using Base.Test

srand(123)

Cd = [0.5, 1.0]
Pd = [2.0, 1.0]
m  = [1.0,-0.5]

dgNP = DiagGaussianNatParam(mean=m, cov=Cd)
dgMP = DiagGaussianMeanParam(mean=m, cov=Cd)

@test isapprox(dgNP.theta1, Pd.*m)
@test isapprox(dgNP.theta2, -Pd)
@test isapprox(dgMP.mu1, m)
@test isapprox(dgMP.mu2, (m.^2+Cd)/2)

gNP = full(dgNP)
gMP = full(dgMP)

@test isapprox(gNP.theta2, diagm(-Pd))
@test isapprox(gMP.mu2, diagm(m.^2+Cd)/2)

@test isvalid(dgNP)
@test isvalid(dgMP)

mpFromNP = meanparam(dgNP)
npFromMP = natparam(dgMP)

@test isapprox(mpFromNP.mu1, dgMP.mu1)
@test isapprox(mpFromNP.mu2, dgMP.mu2)
@test isapprox(npFromMP.theta1, dgNP.theta1)
@test isapprox(npFromMP.theta2, dgNP.theta2)

@test isapprox(mean(dgNP), m)
@test isapprox(mean(dgMP), m)
@test isapprox(cov(dgNP), Cd)
@test isapprox(cov(dgMP), Cd)
@test isapprox(var(dgMP), Cd)
@test isapprox(var(dgNP), Cd)
@test isapprox(std(dgMP), sqrt(Cd))
@test isapprox(std(dgNP), sqrt(Cd))

@test_throws DomainError DiagGaussianNatParam(mean=m,cov=-C,check=true)
@test_throws DomainError DiagGaussianMeanParam(mean=m,cov=-C,check=true)
