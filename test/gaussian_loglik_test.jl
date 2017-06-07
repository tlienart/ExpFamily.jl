using ExpFamily
using Base.Test

dim = 4
srand(123)

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

x   = randn(dim)
ns  = 5
xb  = randn(dim,ns)

Ca = cov(gNPa)
ma = mean(gNPa)
l  = exp( -dot(x-ma, Ca\(x-ma)) / 2 )/sqrt( (2pi)^dim * det(Ca) )
ll = log(l)

lb  = [exp( -dot(xb[:,i]-ma, Ca\(xb[:,i]-ma))/2) for i in 1:ns]
lb /= sqrt( (2pi)^dim * det(Ca) )
llb = log.(lb)

ul  = -0.5(dot(x, Ca\x)-2dot(x, Ca\ma))
ulb = [-0.5(dot(xb[:,i], Ca\xb[:,i])-2dot(xb[:,i], Ca\ma))
        for i in 1:ns]

Cda   = diag(Ca)
gdNPa = DiagGaussianNatParam(mean=ma, cov=Cda)
gdMPa = DiagGaussianMeanParam(mean=ma, cov=Cda)

ldb  = [exp( -dot(xb[:,i]-ma, (xb[:,i]-ma)./Cda)/2) for i in 1:ns]
ldb /= sqrt( (2pi)^dim * prod(Cda) )
lldb = log.(ldb)

uldb  = [-0.5(dot(xb[:,i], xb[:,i]./Cda)-2dot(xb[:,i], ma./Cda))
            for i in 1:ns]


llNPa   = loglik(gNPa,  x)
llNPab  = loglik(gNPa,  xb)
lldNPab = loglik(gdNPa, xb)
llMPab  = loglik(gMPa, xb)

@test isapprox( llNPa,   ll  )
@test isapprox( llNPab,  llb )
@test isapprox( llNPab,  loglik(gNPa, xb', axis=1))
@test isapprox( llNPab,  llMPab )
@test isapprox( llMPab,  loglik(gMPa, xb', axis=1))
@test isapprox( lldNPab, lldb )
@test isapprox( lldNPab, loglik(gdNPa, xb', axis=1) )
@test isapprox( loglik(gdMPa, xb), lldNPab)

@test isapprox( uloglik(gNPa, x), ul)
@test isapprox( uloglik(gNPa, xb), ulb )
@test isapprox( uloglik(gNPa, xb', axis=1), ulb)
@test isapprox( uloglik(gMPa, xb), ulb)
@test isapprox( uloglik(gMPa, xb', axis=1), ulb)

@test isapprox( uloglik(gdNPa, xb), uldb)
@test isapprox( uloglik(gdNPa, xb', axis=1), uldb)
@test isapprox( uloglik(gdMPa, xb), uldb)
@test isapprox( uloglik(gdMPa, xb', axis=1), uldb)

@test isapprox( gradloglik(gNPa, x), Ca\(ma-x) )
@test isapprox( gradloglik(gMPa, x), gradloglik(gNPa, x) )
