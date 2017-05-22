export
    GaussianNatParam,
    GaussianMeanParam,
    DiagGaussianNatParam,
    DiagGaussianMeanParam

###############
## Containers #
###############

immutable GaussianNatParam <: NatParam
    theta1::Vector{Float} # prec * mu
    theta2::Symmetric{Float, Matrix{Float}} # -prec
end

# immutable DiagGaussianNatParam <: NatParam
#     theta1::Vector{Float} # prec*mu
#     theta2::Vector{Float} # diag(-prec)
# end

immutable GaussianMeanParam <: MeanParam
    mu1::Vector{Float} # mu
    mu2::Symmetric{Float, Matrix{Float}} # (mu*mu' + cov)/2
end

# immutable DiagGaussianMeanParam <: MeanParam
#     mu1::Vector{Float}
#     mu2::Vector{Float}
# end

#################
## Constructors #
#################

function GaussianNatParam( t1::Vector{Float}, t2::Matrix{Float}, check=false )
    (check && !isposdef(-t2)) ? throw(DomainError) : nothing
    GaussianNatParam(t1, Symmetric(t2))
end
function GaussianNatParam(;mu::Vector{Float}=[0.0], cov::Matrix{Float}=[1.0],
                           check = false)
    P = inv(cov)
    GaussianNatParam(P*mu, -P, check)
end

function GaussianMeanParam(m1::Vector{Float}, m2::Matrix{Float})
    GaussianMeanParam(m1, Symmetric(m2))
end
function GaussianMeanParam(;mu::Vector{Float}=[0.0], cov::Matrix{Float}=[1.0])
    GaussianMeanParam(mu, 0.5(mu*mu'+cov))
end

###############
## Conversion #
###############

function natparam(g::GaussianMeanParam)::GaussianNatParam
    P = inv(Symmetric(2.0g.mu2.data-g.mu1*g.mu1'))
    GaussianNatParam(P*g.mu1, -P)
end
function meanparam(g::GaussianNatParam)::GaussianMeanParam
    cov = -inv(g.theta2)
    mu1 = cov * g.theta1
    GaussianMeanParam(mu1, 0.5(mu1*mu1' + cov))
end

##############################
## Operations extending base #
##############################

Base.mean(g::GaussianNatParam)     = -g.theta2\g.theta1
Base.mean(g::GaussianMeanParam)    = g.mu1
Base.cov(g::GaussianNatParam)      = -inv(g.theta2)
Base.cov(g::GaussianMeanParam)     = 2.0g.mu2.data-g.mu1*g.mu1'
Base.isvalid(g::GaussianNatParam)  = isposdef(-g.theta2.data)
Base.isvalid(g::GaussianMeanParam) = isposdef(2.0g.mu2.data-g.mu1*g.mu1')

Base.ones(::GaussianNatParam, d::Int) =
    GaussianNatParam(mu=zeros(d), cov=eye(d))
Base.ones(::GaussianMeanParam, d::Int) =
    GaussianMeanParam(mu=zeros(d), cov=eye(d))
# zeros does not really make sense

###################
## Safe operators #
###################

+{T<:GaussianNatParam}(g1::T, g2::T) =
    GaussianNatParam(g1.theta1+g2.theta1, g1.theta2+g2.theta2)

####################
## Unafe operators #
####################

#-{T<:GaussianNatParam}(g1::GaussianNatParam)
