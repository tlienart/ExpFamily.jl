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

# Short types to simplify internal code
const GaussNP = GaussianNatParam
const GaussMP = GaussianMeanParam
const Gauss   = Union{GaussNP, GaussMP}

#################
## Constructors #
#################

### Gaussian Natural Parameter

function GaussianNatParam(t1::Vector{Float}, t2::Matrix{Float},
                          check=false)::GaussNP
    #
    (check && !isposdef(-t2)) ? throw(DomainError()) : nothing
    GaussianNatParam(t1, Symmetric(t2))
end
function GaussianNatParam(;mean::Vector{Float}=[0.0],
                           cov::Matrix{Float}=[1.0],
                           check=false)::GaussNP
    #
    P = inv(cov)
    GaussianNatParam(P*mean, -P, check)
end

### Gaussian Mean Parameter

function GaussianMeanParam(m1::Vector{Float}, m2::Matrix{Float},
                           check=false)::GaussMP
    #
    (check && !isposdef(2m2-m1*m1')) ? throw(DomainError()) : nothing
    GaussianMeanParam(m1, Symmetric(m2))
end
function GaussianMeanParam(;mean::Vector{Float}=[0.0],
                            cov::Matrix{Float}=[1.0],
                            check=false)::GaussMP
    #
    GaussianMeanParam(mean, 0.5(mean*mean'+cov), check)
end

###############
## Conversion #
###############

function natparam(g::GaussMP)::GaussNP
    P = inv(Symmetric(2.0g.mu2.data-g.mu1*g.mu1'))
    GaussianNatParam(P*g.mu1, -P)
end
function meanparam(g::GaussNP)::GaussMP
    cov = -inv(g.theta2)
    mu1 = cov * g.theta1
    GaussianMeanParam(mu1, 0.5(mu1*mu1' + cov))
end

##############################
## Operations extending base #
##############################

Base.mean(g::GaussNP) = -g.theta2\g.theta1
Base.mean(g::GaussMP) = g.mu1
Base.cov(g::GaussNP)  = -inv(g.theta2)
Base.cov(g::GaussMP)  = 2.0g.mu2.data-g.mu1*g.mu1'
Base.var(g::Gauss)    = diag(cov(g))
Base.std(g::Gauss)    = sqrt(var(g))

Base.isvalid(g::GaussNP)  = isposdef(-g.theta2.data)
Base.isvalid(g::GaussMP) = isposdef(2.0g.mu2.data-g.mu1*g.mu1')

Base.ones(::Type{GaussNP}, d::Int) =
    GaussianNatParam(mean=zeros(d), cov=eye(d))
Base.ones(::Type{GaussMP}, d::Int) =
    GaussianMeanParam(mean=zeros(d), cov=eye(d))
# zeros does not really make sense

Base.length(g::GaussNP) = length(g.theta1)
Base.length(g::GaussMP) = length(g.mu1)

Base.rand(g::GaussNP, n::Int=1) =
    repmat(mean(g),1,n)+chol(-g.theta2)\randn(length(g), n)
Base.rand(g::GaussMP, n::Int=1) =
    repmat(mean(g),1,n)+chol(cov(g))*randn(length(g), n)

###################
## Safe operators #
###################

+(g1::GaussNP, g2::GaussNP) =
    GaussianNatParam(g1.theta1+g2.theta1, g1.theta2+g2.theta2)
+(g1::GaussMP, g2::GaussMP) =
    GaussianMeanParam(g1.mu1+g2.mu1, g1.mu2+g2.mu2)

*(a::Float, g::GaussNP) = GaussianNatParam(a*g1.theta1, a*g1.theta2)
*(a::Float, g::GaussMP) = GaussianMeanParam(a*g1.mu1,   a*g1.mu2)
*(g::Gauss, a::Float)   = *(a,g)
/(a::Float, g::Gauss)   = *(1.0/a,g)
/(g::Gauss, a::Float)   = /(a,g)

#####################
## Unsafe operators # (result is not necessarily valid Gaussian)
#####################

-(g1::GaussNP, g2::GaussNP) =
    GaussianNatParam(g1.theta1-g2.theta1, g1.theta2-g2.theta2)
-(g1::GaussMP, g2::GaussMP) =
    GaussianMeanParam(g1.mu1-g2.mu1, g1.mu2-g2.mu2)

#-{T<:GaussianNatParam}(g1::GaussNP)
