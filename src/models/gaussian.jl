export
    GaussianNatParam,
    GaussianMeanParam,
    DiagGaussianNatParam,
    DiagGaussianMeanParam

################################
### GAUSSIAN NATURAL PARAMETER
################################

immutable GaussianNatParam <: NatParam
    theta1::Vector{Float} # prec * mu
    theta2::Symmetric{Float, Matrix{Float}} # -prec
end

#####################################
### DIAG GAUSSIAN NATURAL PARAMETER
#####################################

immutable DiagGaussianNatParam <: NatParam
    theta1::Vector{Float} # prec*mu
    theta2::Vector{Float} # diag(-prec)
    function DiagGaussianNatParam(t1::Vector{Float}, t2::Vector{Float},
                                  check::Bool=false)
        (check && any(t2.>0.0)) ? throw(DomainError()) : nothing
        new(t1, t2)
    end
end

#############################
### GAUSSIAN MEAN PARAMETER
#############################

immutable GaussianMeanParam <: MeanParam
    mu1::Vector{Float} # mu
    mu2::Symmetric{Float, Matrix{Float}} # (mu*mu' + cov)/2
end

##################################
### DIAG GAUSSIAN MEAN PARAMETER
##################################

immutable DiagGaussianMeanParam <: MeanParam
    mu1::Vector{Float}
    mu2::Vector{Float}
    function DiagGaussianMeanParam(m1::Vector{Float}, m2::Vector{Float},
                                  check::Bool=false)
        (check && any(2m2-m1.^2 .< 0.0)) ? throw(DomainError()) : nothing
        new(m1, m2)
    end
end

# Short types to simplify internal code
const GaussNP  = GaussianNatParam
const DGaussNP = DiagGaussianNatParam
const GaussMP  = GaussianMeanParam
const DGaussMP = DiagGaussianMeanParam
const Gauss    = Union{GaussNP, DGaussNP, GaussMP, DGaussMP}
const FGauss   = Union{GaussNP, GaussMP}
const DGauss   = Union{DGaussNP, DGaussMP}

######################
# Extra constructors
######################

### Gaussian Natural Parameter

function GaussianNatParam(t1::Vector{Float}, t2::Matrix{Float},
                          check::Bool=false)::GaussNP
    (check && !isposdef(-t2)) ? throw(DomainError()) : nothing
    GaussianNatParam(t1, Symmetric(t2))
end
function GaussianNatParam(;mean::Vector{Float}=[0.0],
                           cov::Matrix{Float}=[1.0],
                           check::Bool=false)::GaussNP
    P = inv(cov)
    GaussianNatParam(P*mean, -P, check)
end
function GaussianNatParam(theta::Vector{Float}, d::Int,
                          check::Bool=false)::GaussNP
    @assert length(theta)==d+d^2 "inconsistent dimensions"
    GaussianNatParam(theta[1:d], reshape(theta[d+1:end],d,d), check)
end

### Diagonal Gaussian Natural Parameter

function DiagGaussianNatParam(;mean::Vector{Float}=[0.0],
                               cov::Vector{Float}=[1.0], check::Bool=false)::DGaussNP
    P = 1./cov
    DiagGaussianNatParam(P.*mean, -P, check)
end
function DiagGaussianNatParam(theta::Vector{Float}, d::Int,
                              check::Bool=false)::DGaussNP
    @assert length(theta)==2d "inconsistent dimensions"
    DiagGaussianNatParam(theta[1:d], theta[d+1:end], check)
end

### Gaussian Mean Parameter

function GaussianMeanParam(m1::Vector{Float}, m2::Matrix{Float},
                           check::Bool=false)::GaussMP
    (check && !isposdef(2m2-m1*m1')) ? throw(DomainError()) : nothing
    GaussianMeanParam(m1, Symmetric(m2))
end
function GaussianMeanParam(;mean::Vector{Float}=[0.0],
                            cov::Matrix{Float}=[1.0], check::Bool=false)::GaussMP
    GaussianMeanParam(mean, 0.5(mean*mean'+cov), check)
end
function GaussianMeanParam(mu::Vector{Float}, d::Int)::GaussMP
    @assert length(mu)==d+d^2 "inconsistent dimensions"
    GaussianMeanParam(mu[1:d], reshape(mu[d+1:end],d,d))
end

### Gaussian Natural Parameter

function DiagGaussianMeanParam(;mean::Vector{Float}=[0.0],
                                cov::Vector{Float}=[1.0],
                                check::Bool=false)::DGaussMP
    DiagGaussianMeanParam(mean, 0.5(mean.^2+cov), check)
end
function DiagGaussianMeanParam(mu::Vector{Float}, d::Int,
                               check::Bool=false)::DGaussMP
    @assert length(mu)==2d "inconsistent dimensions"
    DiagGaussianMeanParam(mu[1:d], mu[d+1:end], check)
end
