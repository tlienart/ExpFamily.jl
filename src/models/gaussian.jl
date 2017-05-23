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

immutable DiagGaussianNatParam <: NatParam
    theta1::Vector{Float} # prec*mu
    theta2::Vector{Float} # diag(-prec)
end

immutable GaussianMeanParam <: MeanParam
    mu1::Vector{Float} # mu
    mu2::Symmetric{Float, Matrix{Float}} # (mu*mu' + cov)/2
end

immutable DiagGaussianMeanParam <: MeanParam
    mu1::Vector{Float}
    mu2::Vector{Float}
end

# Short types to simplify internal code
const GaussNP  = GaussianNatParam
const DGaussNP = DiagGaussianNatParam
const GaussMP  = GaussianMeanParam
const DGaussMP = DiagGaussianMeanParam
const Gauss    = Union{GaussNP, DGaussNP, GaussMP, DGaussMP}

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

# Constructor for GaussianNatParam assuming theta=[theta1;theta2[:]].
function GaussianNatParam(theta::Vector{Float}, d::Int)
    @assert length(theta)==d+d^2 "inconsistent dimensions"
    GaussianNatParam(theta[1:d], reshape(theta[d+1:end],d,d))
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
# Constructor for GaussianNatParam assuming theta=[theta1;theta2[:]].
function GaussianMeanParam(mu::Vector{Float}, d::Int)
    @assert length(mu)==d+d^2 "inconsistent dimensions"
    GaussianMeanParam(mu[1:d], reshape(mu[d+1:end],d,d))
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
# NOTE zeros does not really make sense

Base.length(g::GaussNP) = length(g.theta1)
Base.length(g::GaussMP) = length(g.mu1)

Base.rand(g::GaussNP, n::Int=1) =
    repmat(mean(g),1,n)+chol(-g.theta2)\randn(length(g), n)
Base.rand(g::GaussMP, n::Int=1) =
    repmat(mean(g),1,n)+chol(cov(g))*randn(length(g), n)

Base.vec(g::GaussNP) = [g.theta1;g.theta2[:]]
Base.vec(g::GaussMP) = [g.mu1;g.mu2[:]]

###################
## Safe operators #
###################

+(g1::GaussNP, g2::GaussNP) =
    GaussianNatParam(vec(g1)+vec(g2), length(g1))
+(g1::GaussMP, g2::GaussMP) =
    GaussianMeanParam(vec(g1)+vec(g2), length(g1))

*(a::Real,  g::GaussNP) = GaussianNatParam(a*vec(g), length(g))
*(a::Real,  g::GaussMP) = GaussianMeanParam(a*vec(g), length(g))
*(g::Gauss, a::Real)   = *(a,g)
/(a::Real,  g::Gauss)   = *(1.0/a,g)
/(g::Gauss, a::Real)   = /(a,g)

#####################
## Unsafe operators # (result is not necessarily valid Gaussian)
#####################

-(g1::GaussNP, g2::GaussNP) =
    GaussianNatParam(vec(g1)-vec(g2), length(g1))
-(g1::GaussMP, g2::GaussMP) =
    GaussianMeanParam(vec(g1)-vec(g2), length(g1))

#########################
## Comparison operators #
#########################

Base.norm(g::Gauss) = norm(vec(g))

function Base.isapprox(g1::Gauss, g2::Gauss;
                       rtol::Real=sqrt(eps()), atol::Real=0)
    norm(g1-g2) <= atol + rtol*max(norm(g1), norm(g2))
end
