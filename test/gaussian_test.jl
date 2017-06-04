using ExpFamily
using Base.Test

srand(123)

C  = [1.0 0.5; 0.5 1.0]
P  = 2/3 * [2 -1; -1 2]
m  = [1.0;-0.5]

gNP = GaussianNatParam(mean=m, cov=C)
gMP = GaussianMeanParam(mean=m, cov=C)

@test isapprox(gNP.theta1, P*m)
@test isapprox(gNP.theta2, -P)
@test isapprox(gMP.mu1, m)
@test isapprox(gMP.mu2, (m*m'+C)/2)

@test isvalid(gNP)
@test isvalid(gMP)

mpFromNP  = meanparam(gNP)
mpFromNP2 = meanparam(gNP, 2.0)
npFromMP  = natparam(gMP)

@test isapprox(mpFromNP.mu1, gMP.mu1)
@test isapprox(mpFromNP.mu2, gMP.mu2)
@test isapprox(npFromMP.theta1, gNP.theta1)
@test isapprox(npFromMP.theta2, gNP.theta2)
@test isapprox(cov(mpFromNP2), 2cov(mpFromNP))

@test isapprox(mean(gNP), m)
@test isapprox(mean(gMP), m)
@test isapprox(cov(gNP), C)
@test isapprox(cov(gMP), C)
@test isapprox(var(gMP), diag(C))
@test isapprox(var(gNP), diag(C))
@test isapprox(std(gMP), sqrt.(diag(C)))
@test isapprox(std(gNP), sqrt.(diag(C)))

@test_throws DomainError GaussianNatParam(mean=m,cov=-C,check=true)
@test_throws DomainError GaussianMeanParam(mean=m,cov=-C,check=true)

dim = 4
gNPones = ones(GaussianNatParam, dim)
gMPones = ones(GaussianMeanParam, dim)

@test isapprox(mean(gNPones), zeros(dim))
@test isapprox(cov(gNPones),  eye(dim))
@test isapprox(mean(gMPones), zeros(dim))
@test isapprox(cov(gMPones),  eye(dim))

@test length(gNPones) == dim
@test length(gMPones) == dim

rNP = rand(gNPones,5000)
rMP = rand(gMPones,5000)

@test isapprox(mean(rNP,2), zeros(dim), atol=1e-1)
@test isapprox(mean(rMP,2), zeros(dim), atol=1e-1)
@test isapprox(cov(rNP,2),  eye(dim),   atol=1e-1)
@test isapprox(cov(rMP,2),  eye(dim),   atol=1e-1)

gNP2 = gNP+gNP
gMP2 = gMP+gMP

@test isapprox(gNP2.theta1, 2gNP.theta1)
@test isapprox(gNP2.theta2.data, 2gNP.theta2.data)
@test isapprox(gMP2.mu1, 2gMP.mu1)
@test isapprox(gMP2.mu2.data, 2gMP.mu2.data)

m1  = randn(dim)
m2  = randn(dim)
C1  = randn(dim,dim)
C1 *= C1'
C2  = randn(dim,dim)
C2 *= C2'

gNPa = GaussianNatParam(mean=m1, cov=C1)
gNPb = GaussianNatParam(mean=m2, cov=C2)
gMPa = GaussianMeanParam(mean=m1, cov=C1)
gMPb = GaussianMeanParam(mean=m2, cov=C2)

gNPtest = gNPa + gNPb - natparam(gMPb)
gMPtest = gMPa + gMPb - meanparam(gNPb)

@test isapprox(gNPtest.theta1, gNPa.theta1)
@test isapprox(gNPtest.theta2, gNPa.theta2)
@test isapprox(gMPtest.mu1, gMPa.mu1)
@test isapprox(gMPtest.mu2, gMPa.mu2)

@test isapprox(gMPtest, gMPa)

@test isapprox(GaussianNatParam(vec(gNPa),dim),gNPa)
@test isapprox(GaussianMeanParam(vec(gMPa),dim),gMPa)

@test isapprox(2gNPa + gNPa*2 - 3gNPa, gNPa)
@test isapprox(2gMPa + gMPa*2 - 3gMPa, gMPa)
@test isapprox(gNPa/2 + gNPa/2, gNPa)
@test isapprox(gMPa/2 + gMPa/2, gMPa)

gNP = GaussianNatParam(mean=m, cov=C)
@test isapprox(project(gNP), gNP)

nP = -diagm([1.5, 1e7])
m  = [1.0;-0.5]
lm = 1e-6

gNPproj = project(GaussianNatParam(-nP*m, nP), minvar=lm)
gMPproj = project(GaussianMeanParam(mean=m, cov=inv(-nP)), minvar=lm)

nPthresh = -diagm([1.5, 1./lm])

@test isapprox(gNPproj.theta1, -nPthresh*m)
@test isapprox(gNPproj.theta2, -diagm([1.5, 1./lm]))
@test isapprox(gMPproj.mu1, m)
@test isapprox(cov(gMPproj), diagm([1./1.5, lm]))

x = randn(dim)

Ca = cov(gNPa)
ma = mean(gNPa)
l  = exp( -dot(x-ma, Ca\(x-ma)) / 2 )/sqrt( (2pi)^dim * det(Ca) )
ll = log(l)

@test isapprox( loglikelihood(gNPa, x), ll )
@test isapprox( loglikelihood(gNPa, x),
                loglikelihood(gMPa, x) )
