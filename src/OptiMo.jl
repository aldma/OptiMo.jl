module OptiMo

import LinearAlgebra.norm

export AbstractOptiModelMeta, OptiModelMeta, AbstractOptiModel

export obj, grad!, cons!, jprod!, jtprod!, proj!, prox!, objprox!

export grad, objgrad, objgrad!, cons, objcons, objcons!,
       jprod, jtprod, prox, objprox, proj, dist

const Maybe{T} = Union{T,Nothing}

include("optimo_utils.jl")
include("optimo_types.jl")

##########################################################
# Methods to be necessarily overridden in other packages.
##########################################################
"""
    fx = obj(prob, x)
"""
function obj end

"""
    grad!(prob, x, dfx)
"""
function grad! end

"""
    cons!(prob, x, cx)
"""
function cons! end

"""
    jprod!(prob, x, v, Jv)
"""
function jprod! end

"""
    jtprod!(prob, x, v, Jtv)
"""
function jtprod! end

"""
    proj!(prob, v, p)
"""
function proj! end

"""
    prox!(prob, x, a, z)
"""
function prox! end

"""
    gz = objprox!(prob, x, a, z)
"""
function objprox! end

##########################################################
# Methods with basic default implementation
##########################################################
"""
    dfx = grad(prob, x)
"""
function grad(prob::AbstractOptiModel, x::AbstractArray)
  @lencheck prob.meta.nvar x
  dfx = similar(x)
  grad!(prob, x, dfx)
  return dfx
end

"""
    fx, dfx = objgrad(prob, x)
"""
function objgrad(prob::AbstractOptiModel, x::AbstractArray)
  @lencheck prob.meta.nvar x
  fx = obj(prob, x)
  dfx = grad(prob, x)
  return fx, dfx
end

"""
    fx = objgrad!(prob, x, dfx)
"""
function objgrad!(prob::AbstractOptiModel, x::AbstractArray, dfx::AbstractArray)
  @lencheck prob.meta.nvar x dfx
  fx = obj(prob, x)
  grad!(prob, x, dfx)
  return fx
end

"""
    cx = cons(prob, x)
"""
function cons(prob::AbstractOptiModel, x::AbstractArray)
  @lencheck prob.meta.nvar x
  cx = similar(x, prob.meta.ncon)
  cons!(prob, x, cx)
  return cx
end

"""
    fx, cx = objcons(prob, x)
"""
function objcons(prob::AbstractOptiModel, x::AbstractArray)
  @lencheck prob.meta.nvar x
  fx = obj(prob, x)
  cx = prob.meta.ncon > 0 ? cons(prob, x) : eltype(x)[]
  return fx, cx
end

"""
    fx = objcons!(prob, x, cx)
"""
function objcons!(prob::AbstractOptiModel, x::AbstractArray, cx::AbstractArray)
  @lencheck prob.meta.nvar x
  @lencheck prob.meta.ncon cx
  fx = obj(prob, x)
  prob.meta.ncon > 0 && cons!(prob, x, cx)
  return fx
end

"""
    Jv = jprod(prob, x, v)
"""
function jprod(prob::AbstractOptiModel, x::AbstractArray, v::AbstractArray)
  @lencheck prob.meta.nvar x v
  Jv = similar(v, prob.meta.ncon)
  jprod!(prob, x, v, Jv)
  return Jv
end

"""
    Jtv = jtprod(prob, x, v)
"""
function jtprod(prob::AbstractOptiModel, x::AbstractArray, v::AbstractArray)
  @lencheck prob.meta.nvar x
  @lencheck prob.meta.ncon v
  Jtv = similar(x)
  jtprod!(prob, x, v, Jtv)
  return Jtv
end

"""
    z = prox(prob, x, a)
"""
function prox(prob::AbstractOptiModel, x::AbstractArray, a::Real)
  @lencheck prob.meta.nvar x
  z = similar(x)
  prox!(prob, x, a, z)
  return z
end

"""
    gz, z = objprox(prob, x, a)
"""
function objprox(prob::AbstractOptiModel, x::AbstractArray, a::Real)
  @lencheck prob.meta.nvar x
  z = similar(x)
  gz = objprox!(prob, x, a, z)
  return gz, z
end

"""
    p = proj(prob, v)
"""
function proj(prob::AbstractOptiModel, v::AbstractArray)
  @lencheck prob.meta.ncon v
  p = similar(v)
  proj!(prob, v, p)
  return p
end

"""
    d = dist(prob, v)
"""
function dist(prob::AbstractOptiModel, v::AbstractArray)
  @lencheck prob.meta.ncon v
  p = proj(prob, v)
  return norm(p - v)
end

##########################################################
# Model subtypes
##########################################################
include("optimo_autodiff.jl")
include("optimo_slack.jl")
include("optimo_auglag.jl")
include("optimo_prox.jl")
include("optimo_feas.jl")
include("optimo_nlp.jl")

##########################################################
# Output types
##########################################################
include("optimo_output.jl")

end # module
