#######################################
## Conversion (assumes input isvalid)
#######################################

function natparam(g::GaussMP)::GaussNP
    P = pinv(2g.mu2.data-g.mu1*g.mu1')
    GaussianNatParam(P*g.mu1, -P)
end
function natparam(g::DGaussMP)::DGaussNP
    P = 1./(2g.mu2-g.mu1.^2)
    DiagGaussianNatParam(P.*g.mu1, -P)
end
function meanparam(g::GaussNP, correction::Float=1.0)::GaussMP
    # NOTE correction -> may want to use (N-P-2)/(N-1) if g.theta2 is noisy,
    # NOTE see litt for estimation of precision matrix.
    cov = -pinv(g.theta2.data)*correction
    mu1 = cov * g.theta1
    GaussianMeanParam(mu1, (mu1*mu1' + cov)/2)
end
function meanparam(g::DGaussNP)::DGaussMP
    cov = -1./g.theta2
    mu1 = cov.*g.theta1
    DiagGaussianMeanParam(mu1, (mu1.^2+cov)/2)
end

Base.full(g::DGaussNP) = GaussianNatParam(g.theta1, diagm(g.theta2))
Base.full(g::DGaussMP) = GaussianMeanParam(g.mu1, diagm(g.mu2))

##############################
## Operations extending base
##############################

Base.mean(g::GaussNP)  = -g.theta2\g.theta1
Base.mean(g::DGaussNP) = -g.theta1./g.theta2
Base.mean(g::GaussMP)  = g.mu1
Base.mean(g::DGaussMP) = g.mu1
Base.cov(g ::GaussNP)  = -pinv(g.theta2.data)
Base.cov(g ::DGaussNP) = -1./g.theta2
Base.cov(g ::GaussMP)  = 2g.mu2.data-g.mu1*g.mu1'
Base.cov(g ::DGaussMP) = 2g.mu2-g.mu1.^2
Base.var(g ::FGauss)   = diag(cov(g))
Base.var(g ::DGauss)   = cov(g)
Base.std(g ::Gauss)    = sqrt.(var(g))

Base.isvalid(g::GaussNP)  = isposdef(-g.theta2.data)
Base.isvalid(g::DGaussNP) = all(-g.theta2 .> 0)
Base.isvalid(g::GaussMP)  = isposdef(2g.mu2.data-g.mu1*g.mu1')
Base.isvalid(g::DGaussMP) = all(2g.mu2-g.mu1.^2 .> 0)

Base.ones(::Type{GaussNP}, d::Int) =
    GaussianNatParam(mean=zeros(d), cov=eye(d))
Base.ones(::Type{DGaussNP}, d::Int) =
    DiagGaussianNatParam(mean=zeros(d), cov=ones(d))
Base.ones(::Type{GaussMP}, d::Int) =
    GaussianMeanParam(mean=zeros(d), cov=eye(d))
Base.ones(::Type{DGaussMP}, d::Int) =
    DiagGaussianMeanParam(mean=zeros(d), cov=ones(d))
# NOTE zeros does not really make sense, just use a one with variance
# to infinity or to zero based on use case

Base.length(g::Union{GaussNP, DGaussNP}) = length(g.theta1)
Base.length(g::Union{GaussMP, DGaussMP}) = length(g.mu1)

Base.rand(g::GaussNP, n::Int=1) =
    broadcast(+, mean(g), chol(-g.theta2)\randn(length(g), n))
Base.rand(g::GaussMP, n::Int=1) =
    broadcast(+, mean(g), chol(cov(g))*randn(length(g), n))
Base.rand(g::Union{DGaussNP, DGaussMP}, n::Int=1) =
    broadcast(+, mean(g), broadcast(*, std(g), randn(length(g),n)))

Base.vec(g::GaussNP)  = [g.theta1;g.theta2[:]]
Base.vec(g::DGaussNP) = [g.theta1;g.theta2]
Base.vec(g::GaussMP)  = [g.mu1;g.mu2[:]]
Base.vec(g::DGaussMP) = [g.mu1;g.mu2]

###################
## Safe operators
###################

+(g1::GaussNP, g2::GaussNP) =
    GaussianNatParam(vec(g1)+vec(g2), length(g1))
+(g1::GaussMP, g2::GaussMP) =
    GaussianMeanParam(vec(g1)+vec(g2), length(g1))
+(g1::DGaussNP, g2::DGaussNP) =
    DiagGaussianNatParam(vec(g1)+vec(g2), length(g1))
+(g1::DGaussMP, g2::DGaussMP) =
    DiagGaussianMeanParam(vec(g1)+vec(g2), length(g1))

*(a::Real,  g::GaussNP)  = GaussianNatParam(a*vec(g), length(g))
*(a::Real,  g::DGaussNP) = DiagGaussianNatParam(a*vec(g), length(g))
*(a::Real,  g::GaussMP)  = GaussianMeanParam(a*vec(g), length(g))
*(a::Real,  g::DGaussMP) = DiagGaussianMeanParam(a*vec(g), length(g))

*(g::Gauss, a::Real)   = *(a,g)
/(a::Real,  g::Gauss)   = *(1.0/a,g)
/(g::Gauss, a::Real)   = /(a,g)

#####################
## Unsafe operators (NOTE result is not necessarily valid Gaussian)
#####################

-(g1::GaussNP, g2::GaussNP) =
    GaussianNatParam(vec(g1)-vec(g2), length(g1))
-(g1::GaussMP, g2::GaussMP) =
    GaussianMeanParam(vec(g1)-vec(g2), length(g1))
-(g1::DGaussNP, g2::DGaussNP) =
    DiagGaussianNatParam(vec(g1)-vec(g2), length(g1))
-(g1::DGaussMP, g2::DGaussMP) =
    DiagGaussianMeanParam(vec(g1)-vec(g2), length(g1))

#########################
## Comparison operators
#########################

Base.norm(g::Gauss) = norm(vec(g))

function Base.isapprox(g1::Gauss, g2::Gauss;
                       rtol::Real=sqrt(eps()), atol::Real=0)
    norm(g1-g2) <= atol + rtol*max(norm(g1), norm(g2))
end

########################
## Projection operator
########################

function project(g::GaussNP; minmu::Float=-Inf, maxmu::Float=Inf,
                 minvar::Float=0., maxvar::Float=Inf)::GaussNP
    (D, V)  = eig(-g.theta2)
    D, V    = real(D), real(V)
    Dthresh = max.(1.0/maxvar, min.(1.0/minvar, D))
    mthresh = max.(minmu, min.(maxmu, mean(g)))
    Pthresh = V*Diagonal(Dthresh)*V'
    GaussianNatParam(Pthresh*mthresh, -Pthresh)
end
function project(g::GaussMP; minmu::Float=-Inf, maxmu::Float=Inf,
                 minvar::Float=0., maxvar::Float=Inf)::GaussMP
    (D,V)   = eig(cov(g))
    D, V    = real(D), real(V)
    Dthresh = max.(minvar, min.(maxvar, D))
    mthresh = max.(minmu, min.(maxmu, mean(g)))
    GaussianMeanParam(mean=mthresh, cov=V*Diagonal(Dthresh)*V')
end
function project(g::DGaussNP; minmu=-Inf, maxmu=Inf,
                 minvar=0, maxvar=Inf)::DGaussNP
    mthresh = max.(minmu,  min.(maxmu,  mean(g)))
    pthresh = max.(1.0 / maxvar, min.(1.0 / minvar, -g.theta2))
    DiagGaussianNatParam(pthresh.*mthresh, -pthresh)
end
function project(g::DGaussMP; minmu=-Inf, maxmu=Inf,
                 minvar=0, maxvar=Inf)::DGaussMP
    mthresh = max.(minmu,  min.(maxmu,  mean(g)))
    vthresh = max.(minvar, min.(maxvar, var(g)))
    DiagGaussianMeanParam(mean=mthresh, cov=vthresh)
end
