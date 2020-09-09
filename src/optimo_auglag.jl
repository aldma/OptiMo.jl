# OPTIMO_AUGLAG
#
# PART OF OPTIMO

export AugLagOptiModel, AugLagUpdate!

"""
Let an AbstractOptiModel `prob` represent the problem
   optimize     f(x) + g(x)
   subject to   c(x) in S

Given an AbstractVector `mu` of positive penalty parameters and
an AbstractVector `y` of Lagrange multipliers,

   AugLagOptiModel(prob, mu, y)

returns an AugLagOptiModel ´alprob´ which represents the problem

   optimize     L(X) + g(X)

where
  L(x) = f(x) + 0.5 * sum_i [(c_i(x) - p_i(x))^2 / mu_i] - 0.5 * sum_i [mu_i * y_i^2]
  p(x) = proj_S( c(x) )

Note that both ´mu´ and `y` are vectors of dimension `prob.meta.ncon`.
"""
mutable struct AugLagOptiModel{R <: Real} <: AbstractOptiModel
    meta::OptiModelMeta
    base::AbstractOptiModel             # original problem
    mu::AbstractVector{R}               # penalty parameter
    y::AbstractVector{R}                # dual variable estimate
    # additional storage
    _muy::AbstractVector{R}              # mu .* y
    _musqy::R                            # 0.5 * sum( mu .* y.^2 )
    _cx::AbstractVector{R}               # c(x)
    _wz::AbstractVector{R}               # w - z
    _yu::AbstractVector{R}               # yu = ( w - z )./mu

    function AugLagOptiModel{R}( meta::OptiModelMeta,
                                 prob::TP,
                                 mu::Tx,
                                 y::Tx,
                        ) where {R <: Real, TP <: AbstractOptiModel, Tx <: AbstractVector{R}}
        ny = prob.meta.ncon
        muy = mu .* y
        musqy = 0.5 * sum( muy .* y )
        new( meta, prob, mu, y, muy, musqy, zeros(R, ny), zeros(R, ny), zeros(R, ny) )
    end
end

function AugLagOptiModel{R}( prob::TP,
                             mu::Tx,
                             y::Tx;
                             name=prob.meta.name * "-al",
                        ) where {R <: Real, TP <: AbstractOptiModel, Tx <: AbstractVector{R}}
    @assert prob.meta.ncon > 0
    @lencheck prob.meta.ncon mu y
    @assert all( mu .> 0 )
    meta = OptiModelMeta( prob.meta.nvar, 0, x0=prob.meta.x0, minimize=prob.meta.minimize, name=name )
    al = AugLagOptiModel{R}( meta, prob, mu, y )
    finalizer( prob -> finalize(prob.base), al )
    return al
end

AugLagOptiModel( prob, mu, y, ::Type{R}; kwargs...) where {R} =
    AugLagOptiModel{R}( prob, mu, y; kwargs...)
AugLagOptiModel( prob, mu, y; kwargs...) =
    AugLagOptiModel( prob, mu, y, Float64; kwargs...)

"""
    AugLagUpdate!( al, mu, y )
"""
function AugLagUpdate!( al::TA, mu::Tx, y::Tx,
            ) where {R <: Real, Tx <: AbstractVector{R}, TA <: AugLagOptiModel{R}}
    @lencheck al.base.meta.ncon mu y
    @assert all( mu .> 0 )
    al.mu = mu
    al.y  = y
    al._muy = al.mu .* al.y
    al._musqy = 0.5 * sum( al._muy .* al.y )
    return nothing
end

"""
    lx = obj( al, x )
"""
function obj( al::AugLagOptiModel, x::AbstractVector )
    @lencheck al.meta.nvar x
    lx = objcons!( al.base, x, al._cx )     # fx, cx
    al._wz .= al._cx .+ al._muy             # w = cx + mu .* y
    proj!( al.base, al._wz, al._yu )        # z = proj( cx + mu .* y )
    al._wz .-= al._yu                       # w - z
    al._yu .= al._wz ./ al.mu               # yu = (w - z) ./ mu
    lx += 0.5 * sum( al._wz .* al._yu )     # fx + 1/(2 mu) dist[S]^2( w )
    lx -= al._musqy
    return lx
end

"""
    grad!( al, x, dlx )
"""
function grad!( al::AugLagOptiModel, x::AbstractVector, dlx::AbstractVector )
    @lencheck al.meta.nvar x dlx
    cons!( al.base, x, al._cx )             # cx
    al._wz .= al._cx .+ al._muy             # w = cx + mu .* y
    proj!( al.base, al._wz, al._yu )        # z = proj( cx + mu .* y )
    al._wz .-= al._yu                       # w - z
    al._yu .= al._wz ./ al.mu               # yu = (w - z) ./ mu
    grad!( al.base, x, dlx )                # dfx
    dlx .+= jtprod( al.base, x, al._yu )    # dfx + dcx' * yu
    return nothing
end

"""
    lx = objgrad!( al, x, dlx )
"""
function objgrad!( al::AugLagOptiModel, x::AbstractVector, dlx::AbstractVector )
    @lencheck al.meta.nvar x dlx
    cons!( al.base, x, al._cx )             # cx
    al._wz .= al._cx .+ al._muy             # w = cx + mu .* y
    proj!( al.base, al._wz, al._yu )        # z = proj( cx + mu .* y )
    al._wz .-= al._yu                       # w - z
    al._yu .= al._wz ./ al.mu               # yu = (w - z) ./ mu
    lx = objgrad!( al.base, x, dlx )        # fx, dfx
    lx += 0.5 * sum( al._wz .* al._yu )     # fx + 1/(2 mu) dist[S]^2( w )
    lx -= al._musqy
    dlx .+= jtprod( al.base, x, al._yu )    # dfx + dcx' * yu
    return lx
end

"""
    prox!( al, x, a, z )
"""
function prox!(al::AugLagOptiModel, x::AbstractVector, a::Real, z::AbstractVector)
    @lencheck al.meta.nvar x z
    prox!( al.base, x, a, z )
    return nothing
end

"""
    gz = objprox!( al, x, a, z )
"""
function objprox!(al::AugLagOptiModel, x::AbstractVector, a::Real, z::AbstractVector)
    @lencheck al.meta.nvar x z
    gz = objprox!( al.base, x, a, z )
    return gz
end
