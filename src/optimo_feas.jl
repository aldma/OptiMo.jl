# OPTIMO_FEAS
#
# PART OF OPTIMO

"""
    FeasOptiModel( prob, [x0, with_indicator, xp, dp, wp, name] )
    represents the following structured, proximal, feasibility problem:
    ```
    minimize  Φ(x) + with_indicator ind_g( x )
    ```
    with respect to `x`, starting from `x0`, where
    ```
    Φ(x)  :=  m( c(x) - proj_S( c(x) ) ) + wprox/2 * || dprox .* (x - xprox) ||^2
    ```
    and `m()` is a loss function, e.g., L1-norm, L2-norm, and Huber loss.

    Optional arguments:
    * `x0::Vector` (default: `xprox`) initial guess
    * `with_indicator::Bool` (default: `false`) include proximable term as indicator?
    * `xprox::Vector` (default: `prob.meta.x0`) proximal point
    * `dprox::Vector` (default: `ones(R,nvar)`) vector for nonnegative, diagonal scaling
    * `wprox::Real` (default: `0`) nonnegative scalar value for proximal regularization
    * `name::String` (default: `prob.meta.name * "-feas"`) model name
"""

using LinearAlgebra: norm

export FeasOptiModel
export infeasibility, infeasibilitygrad!
export cviolation, proxdistance, unsproxdistance

mutable struct FeasOptiModel <: AbstractOptiModel
    meta::OptiModelMeta
    base::AbstractOptiModel
    xprox::AbstractVector
    dprox::AbstractVector
    wprox::Real
    with_prox::Bool
    with_indicator::Bool
    loss::Symbol
    huber_rho::Real
    huber_mu::Real
    # additional allocation
    _cx::AbstractVector
    _px::AbstractVector

    function FeasOptiModel( meta::OptiModelMeta,
                            base::TP,
                            xprox::Tx,
                            dprox::Tx,
                            wprox::Real,
                            with_prox::Bool,
                            with_indicator::Bool,
                            loss::Symbol,
                            huber_rho::Real,
                            huber_mu::Real,
    ) where {TP <: AbstractOptiModel, Tx <: AbstractVector}
        R = eltype(base.meta.x0)
        ncon = base.meta.ncon
        new( meta, base, xprox, dprox, wprox, with_prox, with_indicator, loss,
             huber_rho, huber_mu, zeros(R,ncon), zeros(R,ncon) )
    end
end

function FeasOptiModel( prob::TP;
                        xprox::Tx=prob.meta.x0,
                        dprox::Tx=ones(eltype(prob.meta.x0),prob.meta.nvar),
                        wprox::Real=zero(eltype(prob.meta.x0)),
                        x0::Tx=xprox,
                        with_indicator::Bool=false,
                        loss::Symbol=:l2sq,
                        huber_rho::Real=1.0,
                        huber_mu::Real=1.0,
                        name::String=prob.meta.name * "-feas",
        ) where {TP <: AbstractOptiModel, Tx <: AbstractVector}
    @assert length(x0) == prob.meta.nvar
    @assert length(xprox) == prob.meta.nvar
    @assert length(dprox) == prob.meta.nvar
    @assert 0 <= wprox
    @assert all( 0 .<= dprox )
    @assert loss ∈ [:l1, :l2, :l2sq, :huber]
    @assert 0 < huber_rho
    @assert 0 < huber_mu
    meta = OptiModelMeta( prob.meta.nvar, 0, x0=x0, minimize=true, name=name )
    with_prox = wprox > 0 && all( dprox .> 0 )
    fprob = FeasOptiModel( meta, prob, xprox, dprox, wprox, with_prox,
                           with_indicator, loss, huber_rho, huber_mu )
    finalizer( p -> finalize(p.base), fprob )
    return fprob
end

##########################################################
# Methods
##########################################################
# obj, grad!, prox!, objprox!
"""
    obj( fprob, x )
evaluates Φ at x.
"""
function obj(prob::FeasOptiModel, x::AbstractVector)
    Φ = infeasibility( prob, x )
    if prob.with_prox
        Φ += prob.wprox * proxdistance( prob, x )
    end
    return Φ
end

"""
    grad!( fprob, x, dfx )
computes ∇Φ at x, in place.
"""
function grad!(prob::FeasOptiModel, x::AbstractVector, dfx::AbstractVector)
    @lencheck prob.meta.nvar dfx
    infeasibilitygrad!( prob, x, dfx )
    if prob.with_prox
        dfx .+= prob.wprox .* prob.dprox .* (x .- prob.xprox)
    end
    return nothing
end

"""
    prox!( fprob, x, a, z )
computes z := prox_{a g}(x), in place.
"""
function prox!(prob::FeasOptiModel, x::AbstractVector, a::Real, z::AbstractVector)
    @lencheck prob.meta.nvar x z
    if prob.with_indicator
        prox!(prob.base, x, a, z )
    else
        copyto!(z, x)
    end
    return nothing
end

"""
    objprox!( fprob, x, a, z )
computes z := prox_{a g}(x) and g(z), in place, assuming g is an indicator.
"""
function objprox!(prob::FeasOptiModel, x::AbstractVector, a::Real, z::AbstractVector)
    @lencheck prob.meta.nvar x z
    if prob.with_indicator
        prox!(prob.base, x, a, z )
    else
        copyto!(z, x)
    end
    return 0.0
end


##########################################################
"""
    infeasibility( fprob, x )
computes the infeasibility measure Φ at x, without proximal regularization (wprox = 0).
"""
function infeasibility( prob::FeasOptiModel, x::AbstractVector )
    evalinfeasvec( prob, x )
    if prob.loss == :l1
        return norm( prob._px, 1 )
    elseif prob.loss == :l2
        return norm( prob._px, 2 )
    elseif prob.loss == :l2sq
        return 0.5 * norm( prob._px, 2 )^2
    elseif prob.loss == :huber
        return huberloss( prob._px, prob.huber_rho, prob.huber_mu )
    end
end

"""
    infeasibility_grad!( fprob, x, dfx )
computes the gradient ∇Φ of the infeasibility measure Φ at x, without proximal
regularization (wprox = 0).
"""
function infeasibilitygrad!(prob::FeasOptiModel, x::AbstractVector, dfx::AbstractVector)
    evalinfeasvec( prob, x )
    if prob.loss == :l1
        prob._px .= sign.( prob._px )
        jtprod!( prob.base, x, prob._px, dfx )
    elseif prob.loss == :l2
        normx = norm( prob._px, 2 )
        if normx == 0
            dfx .= 0
        else
            prob._px ./= normx
            jtprod!( prob.base, x, prob._px, dfx )
        end
    elseif prob.loss == :l2sq
        jtprod!( prob.base, x, prob._px, dfx )
    elseif prob.loss == :huber
        huberlossgrad!( prob._px, prob.huber_rho, prob.huber_mu, prob._px )
        jtprod!( prob.base, x, prob._px, dfx )
    end
    return nothing
end

"""
    cviolation( fprob, x )
computes the constraint violation.
"""
function cviolation( prob::FeasOptiModel, x::AbstractVector )
    evalinfeasvec( prob, x )
    return norm( prob._px, Inf )
end

"""
    proxdistance( fprob, x )
computes the (scaled) distance to the proximal point.
"""
proxdistance, unsproxdistance
function proxdistance( prob::FeasOptiModel, x::AbstractVector )
    @lencheck prob.meta.nvar x
    return 0.5 * sum( prob.dprox .* (x .- prob.xprox).^2 )
end

"""
    unsproxdistance( fprob, x )
computes the unscaled distance to the proximal point.
"""
function unsproxdistance( prob::FeasOptiModel, x::AbstractVector )
    @lencheck prob.meta.nvar x
    return 0.5 * norm( x .- prob.xprox, 2)^2
end

##########################################################
# INTERNAL FUNCTIONS
##########################################################
function evalinfeasvec( prob::FeasOptiModel, x::AbstractVector )
    @lencheck prob.meta.nvar x
    cons!( prob.base, x, prob._cx )
    proj!( prob.base, prob._cx, prob._px )
    prob._px .= prob._cx .- prob._px
    return nothing
end

"""
    Huber loss, with parameter ρ > 0
        f(x) = { 1/2 ||x||^2           if ||x|| ⩽ ρ
               { ρ ( ||x|| - ρ/2 )     otherwise
"""
function huberloss( x::AbstractVector, rho::Real, mu::Real )
    normx = norm( x, 2 )
    if normx <= rho
        return 0.5 * mu * normx^2
    else
        return rho * mu * ( normx - 0.5 * rho )
    end
end

function huberlossgrad!( x::AbstractVector, rho::Real, mu::Real, dfx::AbstractVector )
    normx = norm( x, 2 )
    if normx <= rho
        dfx .= mu .* x
    else
        dfx .= ( rho * mu / normx ) .* x
    end
    return nothing
end
