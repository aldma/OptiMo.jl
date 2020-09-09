
#
# G(x) := g(x) + ind[S](c(x))
#
# prox_{a G}(x0) = argmin { G(x) + 1/(2a) ||x-x0||^2 }
#                    x
#                = argmin { g(x) + 1/(2a) ||x-x0||^2 | c(x) in S }
#                    x
#
# minimize         1/(2a) ||x-x0||^2 + g(x)
# subject to       c(x) in S
#
# ProxOptiModel( prob, x0, a, [y0] )
#   f(x) = 1/(2a) ||x-x0||^2
#   g(x) = g(x)
#   c(x) = c(x)
#   S    = S

import LinearAlgebra.norm

export ProxOptiModel

# Type
mutable struct ProxOptiModel <: AbstractOptiModel
  meta::OptiModelMeta
  base::AbstractOptiModel
  a::Real
  x0::AbstractVector

  function ProxOptiModel( meta::OptiModelMeta, prob::TP, a::Real, x0::Tx ) where {TP <: AbstractOptiModel, Tx <: AbstractVector}
    new( meta, prob, a, x0 )
  end
end

# Constructor
function ProxOptiModel( prob::TP, a::Real; x0::Tx=prob.meta.x0, y0::Tx=prob.meta.y0, name=prob.meta.name * "-prox" ) where {TP <: AbstractOptiModel, Tx <: AbstractVector}
  @assert a > 0
  @lencheck prob.meta.nvar x0
  @lencheck prob.meta.ncon y0
  meta = OptiModelMeta( prob.meta.nvar, prob.meta.ncon, x0=x0, y0=y0, minimize=true, name=name )
  pr = ProxOptiModel(meta, prob, a, x0)
  finalizer(prob -> finalize(prob.base), pr)
  return pr
end

##########################################################
# Methods
##########################################################
# obj, grad!, cons!, jprod!, jtprod!, proj!, prox!, objprox!

function obj(pr::ProxOptiModel, x::AbstractVector)
  @lencheck pr.meta.nvar x
  return (0.5 / pr.a) * norm( x - pr.x0, 2 )^2
end

function grad!(pr::ProxOptiModel, x::AbstractVector, dfx::AbstractVector)
  @lencheck pr.meta.nvar x dfx
  dfx = (x - pr.x0) ./ pr.a
  return nothing
end

function cons!(pr::ProxOptiModel, x::AbstractVector, cx::AbstractVector)
  @lencheck pr.meta.nvar x
  @lencheck pr.meta.ncon cx
  cons!(pr.base, x, cx)
  return nothing
end

function jprod!(pr::ProxOptiModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck prob.meta.nvar x v
  @lencheck pr.meta.ncon Jv
  jprod!(pr.base, x, v, Jv)
  return nothing
end

function jtprod!(pr::ProxOptiModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck pr.meta.nvar x Jtv
  @lencheck pr.meta.ncon v
  jtprod!(pr.base, x, v, Jtv)
  return nothing
end

function proj!(pr::ProxOptiModel, cx::AbstractVector, px::AbstractVector)
  @lencheck pr.meta.ncon cx px
  proj!(pr.base, cx, px )
  return nothing
end

function prox!(pr::ProxOptiModel, x::AbstractVector, a::Real, z::AbstractVector)
  @lencheck pr.meta.nvar x z
  prox!(pr.base, x, a, z )
  return nothing
end

function objprox!(pr::ProxOptiModel, x::AbstractVector, a::Real, z::AbstractVector)
  @lencheck pr.meta.nvar x z
  gz = objprox!( pr.base, x, a, z )
  return gz
end


