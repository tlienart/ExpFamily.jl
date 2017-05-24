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

mpFromNP  = meanparam(dgNP)
mpFromNP2 = meanparam(dgNP, .5)
npFromMP = natparam(dgMP)

@test isapprox(mpFromNP.mu1, dgMP.mu1)
@test isapprox(mpFromNP.mu2, dgMP.mu2)
@test isapprox(npFromMP.theta1, dgNP.theta1)
@test isapprox(npFromMP.theta2, dgNP.theta2)
@test isapprox(cov(mpFromNP2), 2*cov(mpFromNP))

@test isapprox(mean(dgNP), m)
@test isapprox(mean(dgMP), m)
@test isapprox(cov(dgNP), Cd)
@test isapprox(cov(dgMP), Cd)
@test isapprox(var(dgMP), Cd)
@test isapprox(var(dgNP), Cd)
@test isapprox(std(dgMP), sqrt(Cd))
@test isapprox(std(dgNP), sqrt(Cd))

@test_throws DomainError DiagGaussianNatParam(mean=m,cov=-Cd,check=true)
@test_throws DomainError DiagGaussianMeanParam(mean=m,cov=-Cd,check=true)

dim = 4
dgNPones = ones(DiagGaussianNatParam, dim)
dgMPones = ones(DiagGaussianMeanParam, dim)

@test isapprox(mean(dgNPones), zeros(dim))
@test isapprox(cov(dgNPones),  ones(dim))
@test isapprox(mean(dgMPones), zeros(dim))
@test isapprox(cov(dgMPones),  ones(dim))

@test length(dgNPones) == dim
@test length(dgMPones) == dim

rNP = rand(dgNPones,5000)
rMP = rand(dgMPones,5000)

@test isapprox(mean(rNP,2), zeros(dim), atol=1e-1)
@test isapprox(mean(rMP,2), zeros(dim), atol=1e-1)
@test isapprox(cov(rNP,2),  eye(dim),  atol=1e-1)
@test isapprox(cov(rMP,2),  eye(dim),  atol=1e-1)

dgNP2 = dgNP+dgNP
dgMP2 = dgMP+dgMP

@test isapprox(dgNP2.theta1, 2dgNP.theta1)
@test isapprox(dgNP2.theta2, 2dgNP.theta2)
@test isapprox(dgMP2.mu1, 2dgMP.mu1)
@test isapprox(dgMP2.mu2, 2dgMP.mu2)

m1  = randn(dim)
m2  = randn(dim)
C1  = randn(dim,dim)
C1 *= C1'
C2  = randn(dim,dim)
C2 *= C2'

dgNPa = DiagGaussianNatParam(mean=m1, cov=diag(C1))
dgNPb = DiagGaussianNatParam(mean=m2, cov=diag(C2))
dgMPa = DiagGaussianMeanParam(mean=m1, cov=diag(C1))
dgMPb = DiagGaussianMeanParam(mean=m2, cov=diag(C2))

dgNPtest = dgNPa + dgNPb - natparam(dgMPb)
dgMPtest = dgMPa + dgMPb - meanparam(dgNPb)

@test isapprox(dgNPtest.theta1, dgNPa.theta1)
@test isapprox(dgNPtest.theta2, dgNPa.theta2)
@test isapprox(dgMPtest.mu1, dgMPa.mu1)
@test isapprox(dgMPtest.mu2, dgMPa.mu2)

@test isapprox(dgMPtest, dgMPa)

@test isapprox(DiagGaussianNatParam(vec(dgNPa),dim),dgNPa)
@test isapprox(DiagGaussianMeanParam(vec(dgMPa),dim),dgMPa)

@test isapprox(2dgNPa + dgNPa*2 - 3dgNPa, dgNPa)
@test isapprox(2dgMPa + dgMPa*2 - 3dgMPa, dgMPa)
@test isapprox(dgNPa/2 + dgNPa/2, dgNPa)
@test isapprox(dgMPa/2 + dgMPa/2, dgMPa)

x = randn(10)

gMP  = suffstats(GaussianNatParam, x)
gMP2 = suffstats(GaussianMeanParam,x)
dgMP = suffstats(DiagGaussianNatParam, x)
dgMP2= suffstats(DiagGaussianMeanParam, x)

@test isapprox(gMP.mu1, x)
@test isapprox(gMP.mu2, (x*x')/2)
@test isapprox(dgMP.mu1, x)
@test isapprox(dgMP.mu2, x.^2/2)
@test isapprox(gMP2, gMP)
@test isapprox(dgMP2, dgMP)

nP = -[1.5;1e7]
m  = [1.0;-0.5]
lm = 1e-6

dgNPproj = project(DiagGaussianNatParam(-nP.*m, nP), minvar=lm)
dgMPproj = project(DiagGaussianMeanParam(mean=m, cov=-1.0./nP), minvar=lm)

nPthresh = -[1.5; 1./lm]

@test isapprox(dgNPproj.theta1, -nPthresh.*m)
@test isapprox(dgNPproj.theta2, -[1.5; 1./lm])
@test isapprox(dgMPproj.mu1, m)
@test isapprox(cov(dgMPproj), [1./1.5; lm])
