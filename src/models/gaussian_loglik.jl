const neghalflog2pi = -.5log(2pi)

#######################################
### loglik == Natural Parameter space
#######################################

"""
    loglik( g::GaussNP, x )

Exact log likelihood of a Gaussian expressed in the Natural Parameter space.
By default x is assumed to be of size D x N but if axis=1 then it will consider
it is given as N x D (where N=number of points, D=dimension).
"""
function loglik( g::GaussNP, x::AbstractArray{Float};
                 axis::Int=2 )::Union{Float,Vector{Float}}
    precmu   = g.theta1
    sqrtprec = chol(-g.theta2)
    tmp      = sqrtprec'\precmu
    fun(x)   = ( dot(x,g.theta2*x) + 2dot(x,precmu) )/2
    result   = sum(neghalflog2pi + log.(diag(sqrtprec))) - norm(tmp)^2/2
    result  += funarray(fun, x, length(precmu), axis)
end

"""
    loglik( g::DGaussNP, x )

Exact log likelihood of a Diagonal Gaussian expressed in the Natural Parameter
space. By default x is assumed to be of size D x N but if axis=1 then it will
consider it is given as N x D (where N=number of points, D=dimension).
"""
function loglik( g::DGaussNP, x::AbstractArray{Float};
                 axis::Int=2 )::Union{Float,Vector{Float}}
    precmu   = g.theta1
    sqrtprec = sqrt.(-g.theta2)
    tmp      = precmu./sqrtprec
    fun(x)   = ( dot(x, g.theta2 .* x) + 2dot(x, precmu) )/2
    result   = sum(neghalflog2pi + log.(sqrtprec)) - norm(tmp)^2/2
    result  += funarray(fun, x, length(precmu), axis)
end

#####################################
### loglik == Mean Parameter space
#####################################

"""
    loglik( g::GaussMP, x )

Exact log likelihood of a Gaussian expressed in the Mean Parameter space.
By default x is assumed to be of size D x N but if axis=1 then it will consider
it is given as N x D (where N=number of points, D=dimension).
"""
function loglik( g::GaussMP, x::AbstractArray{Float};
                 axis::Int=2 )::Union{Float,Vector{Float}}
    sqrtcov = chol(cov(g))
    fun(x)  = -norm(sqrtcov'\(x-g.mu1))^2/2
    result  = sum(neghalflog2pi - log.(diag(sqrtcov)))
    result += funarray(fun, x, length(g.mu1), axis)
end

"""
    loglik( g::DGaussMP, x )

Exact log likelihood of a Diagonal Gaussian expressed in the Mean Parameter
space. By default x is assumed to be of size D x N but if axis=1 then it will
consider it is given as N x D (where N=number of points, D=dimension).
"""
function loglik( g::DGaussMP, x::AbstractArray{Float};
                 axis::Int=2 )::Union{Float,Vector{Float}}
    stds    = std(g)
    fun(x)  = -norm((x-g.mu1)./stds)^2/2
    result  = sum(neghalflog2pi - log.(stds))
    result += funarray(fun, x, length(g.mu1), axis)
end

#######################################
### uloglik == Natural Parameter space
#######################################

"""
    uloglik( g::GaussNP, x )

Loglikelihood of a GaussNP without the constants (numerically more stable as
it does not require computing determinants and inversions). This is useful
in a sampler where the normalising constant can be neglected.
It corresponds to -0.5(<x, Px> -2<x,Pmu>).
By default x is assumed to be of size D x N but if axis=1 then it will consider
it is given as N x D (where N=number of points, D=dimension).
"""
function uloglik( g::GaussNP, x::AbstractArray{Float};
                  axis::Int=2 )::Union{Float,Vector{Float}}
    fun(x) = dot(x, g.theta1+g.theta2*x/2)
    funarray(fun, x, length(g), axis)
end

"""
    uloglik( g::DGaussNP, x )

Loglikelihood of a DGaussNP without the constants (numerically more stable as
it does not require computing determinants and inversions). This is useful
in a sampler where the normalising constant can be neglected.
It corresponds to -0.5(<x, Px> -2<x,Pmu>).
By default x is assumed to be of size D x N but if axis=1 then it will consider
it is given as N x D (where N=number of points, D=dimension).
"""
function uloglik( g::DGaussNP, x::AbstractArray{Float};
                  axis::Int=2 )::Union{Float,Vector{Float}}
    fun(x) = dot(x, g.theta1+g.theta2.*x/2)
    funarray(fun, x, length(g), axis)
end

#####################################
### uloglik == Mean Parameter space
#####################################

"""
    uloglik( g::GaussMP, x )

Loglikelihood of a GaussMP without the constants (numerically more stable as
it does not require computing determinants and inversions). This is useful
in a sampler where the normalising constant can be neglected.
It corresponds to -0.5(<x, Px> -2<x,Pmu>).
"""
function uloglik( g::GaussMP, x::AbstractArray{Float};
                  axis::Int=2 )::Union{Float,Vector{Float}}
    S = cov(g)
    fun(x) = dot(x, S\(g.mu1-x/2))
    funarray(fun, x, length(g), axis)
end

"""
    uloglik( g::DGaussMP, x )

Loglikelihood of a DGaussMP without the constants (numerically more stable as
it does not require computing determinants and inversions). This is useful
in a sampler where the normalising constant can be neglected.
It corresponds to -0.5(<x, Px> -2<x,Pmu>).
"""
function uloglik( g::DGaussMP, x::AbstractArray{Float};
                  axis::Int=2 )::Union{Float,Vector{Float}}
    S = cov(g)
    fun(x) = dot(x, (g.mu1-x/2)./S)
    funarray(fun, x, length(g), axis)
end

################
### gradloglik
################

# NOTE at the moment only handles point inputs

gradloglik(g::GaussNP,  x::Vector{Float}) = g.theta1 + g.theta2*x
gradloglik(g::DGaussNP, x::Vector{Float}) = g.theta1 + g.theta2 .* x
gradloglik(g::GaussMP,  x::Vector{Float}) = cov(g)\(g.mu1 - x)
gradloglik(g::DGaussMP, x::Vector{Float}) = (g.mu1 - x)./cov(g)
