
import NLPModels

export NLPOptiModel

#export obj, grad!, cons!, jprod!, jtprod!, proj!, prox!, objprox!

#
#  Given an AbstractNLPModel ´nlp´ representing the problem
#
#    optimize   f(x)
#    s.t.       xl <=   x  <= xu
#               cl <= c(x) <= cu
#
#  the call
#
#    prob = NLPOptiModel( nlp )
#
#  returns an NLPOptiModel <: AbstractOptiModel ´prob´ representing the same problem
#  in the form
#
#    optimize   f(x) + g(x)
#    s.t.       c(x) in S
#
#  where
#
#    g = indBox( xl, xu )
#    S = Box( cl, cu )
#
#  In addition to keeping `meta` as any OptiModel, an NLPOptiModel
#  also stores ´base´, the original problem ´nlp´.

# Type
mutable struct NLPOptiModel{M <: NLPModels.AbstractNLPModel} <: AbstractOptiModel
  meta::OptiModelMeta
  base::M
end

# Constructor
function NLPOptiModel( nlp::M; name=nlp.meta.name * "-nlpopti" ) where {M <: NLPModels.AbstractNLPModel}

  meta = OptiModelMeta(nlp.meta.nvar, nlp.meta.ncon, x0=nlp.meta.x0, y0=nlp.meta.y0, minimize=nlp.meta.minimize, name=name)

  return NLPOptiModel(meta, nlp)
end

##########################################################
# Necessary methods
##########################################################
# obj       : unchanged
# grad!     : unchanged
# cons!     : unchanged
# jprod!    : unchanged
# jtprod!   : unchanged
# proj!     : projection onto [cl, cu]
# prox!     : projection onto [xl, xu]
# objprox!  : projection onto [xl, xu], zero cost

function obj(prob::NLPOptiModel, x::AbstractVector)
  return NLPModels.obj(prob.base, x)
end

function grad!(prob::NLPOptiModel, x::AbstractVector, dfx::AbstractVector)
  return NLPModels.grad!(prob.base, x, dfx)
end

function cons!(prob::NLPOptiModel, x::AbstractVector, cx::AbstractVector)
  return NLPModels.cons!(prob.base, x, cx)
end

function jprod!(prob::NLPOptiModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  return NLPModels.jprod!(prob.base, x, v, Jv)
end

function jtprod!(prob::NLPOptiModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  return NLPModels.jtprod!(prob.base, x, v, Jtv)
end

function proj!(prob::NLPOptiModel, cx::AbstractVector, px::AbstractVector)
  @lencheck prob.meta.ncon cx px
  px .= clamp.( cx, prob.base.meta.lcon, prob.base.meta.ucon)
  return nothing
end

function prox!(prob::NLPOptiModel, x::AbstractVector, a::Real, z::AbstractVector)
  @lencheck prob.meta.nvar x z
  z .= clamp.( x, prob.base.meta.lvar, prob.base.meta.uvar)
  return nothing
end

function objprox!(prob::NLPOptiModel, x::AbstractVector, a::Real, z::AbstractVector)
  @lencheck prob.meta.nvar x z
  z .= clamp.( x, prob.base.meta.lvar, prob.base.meta.uvar)
  return 0.0
end

#  for meth in (:obj, :grad!, :cons!, :jprod!, :jtprod!)
#    @eval NLPOptiMo.$meth(prob::NLPOptiModel, ) = NLPModels.$meth(prob.base, )
#  end
