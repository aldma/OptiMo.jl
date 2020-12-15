
import LinearAlgebra.norm

export SlackOptiModel

  # Let a model `prob` represent the problem
  #
  #  optimize     f(x) + g(x)
  #  subject to   c(x) in S
  #
  # A model `sprob` of this type, given `prob`, introduces a slack variable `s`
  # and transforms the problem into
  #
  #  optimize     F(X) + G(X)
  #  subject to   C(X) = 0
  #
  # where
  #  X = [x, s]
  #  F(X) = f(x)
  #  G(X) = g(x) + indicator[S](s)
  #  C(X) = c(x) - s

# Type
mutable struct SlackOptiModel <: AbstractOptiModel
  meta::OptiModelMeta
  base::AbstractOptiModel
end

function slack_meta(meta::OptiModelMeta; name=meta.name * "-slack")

  T = eltype(meta.x0)

  return OptiModelMeta( meta.nvar + meta.ncon, meta.ncon,
                        x0=[meta.x0 ; zeros(T, meta.ncon)],
                        y0=meta.y0,
                        minimize=meta.minimize, name=name )
end

# Constructor
function SlackOptiModel(prob::AbstractOptiModel; name=prob.meta.name * "-slack")
  meta = slack_meta(prob.meta, name=name)
  sprob = SlackOptiModel(meta, prob)
  finalizer(prob -> finalize(prob.base), sprob)
  return sprob
end

###########################################################################
# Necessary methods
###########################################################################
function obj(prob::SlackOptiModel, x::AbstractVector)
  @lencheck prob.meta.nvar x
  return obj(prob.base, @view x[1:prob.base.meta.nvar])
end

function grad!(prob::SlackOptiModel, x::AbstractVector, dfx::AbstractVector)
  @lencheck prob.meta.nvar x dfx
  n = prob.base.meta.nvar
  ns = prob.meta.nvar - n
  @views begin
    grad!(prob.base, x[1:n], dfx[1:n])
    dfx[n+1:n+ns] .= 0
  end
  return nothing
end

function cons!(prob::SlackOptiModel, x::AbstractVector, cx::AbstractVector)
  @lencheck prob.meta.nvar x
  @lencheck prob.meta.ncon cx
  n = prob.base.meta.nvar
  ns = prob.meta.nvar - n
  @views begin
    cons!(prob.base, x[1:n], cx)
    cx -= x[n+1:n+ns]
  end
  return nothing
end

function jprod!(prob::SlackOptiModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck prob.meta.nvar x v
  @lencheck prob.meta.ncon Jv
  n = prob.base.meta.nvar
  ns = prob.meta.nvar - n
  @views begin
    jprod!(prob.base, x[1:n], v[1:n], Jv)
    Jv -= v[n+1:n+ns]
  end
  return nothing
end

function jtprod!(prob::SlackOptiModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck prob.meta.nvar x Jtv
  @lencheck prob.meta.ncon v
  n = prob.base.meta.nvar
  ns = prob.meta.nvar - n
  @views begin
    jtprod!(prob.base, x[1:n], v, Jtv[1:n])
    Jtv[n+1:n+ns] = -v
  end
  return nothing
end

function proj!(prob::SlackOptiModel, cx::AbstractVector, px::AbstractVector)
  @lencheck prob.meta.ncon cx px
  fill!(px, 0)
  return nothing
end

function prox!(prob::SlackOptiModel, x::AbstractVector, a::Real, z::AbstractVector)
  @lencheck prob.meta.nvar x z
  n = prob.base.meta.nvar
  ns = prob.meta.nvar - n
  @views begin
    prox!(prob.base, x[1:n], a, z[1:n])
    proj!(prob.base, x[n+1:n+ns], z[n+1:n+ns])
  end
  return nothing
end

function objprox!(prob::SlackOptiModel, x::AbstractVector, a::Real, z::AbstractVector)
  @lencheck prob.meta.nvar x z
  n = prob.base.meta.nvar
  ns = prob.meta.nvar - n
  @views begin
    gz = objprox!(prob.base, x[1:n], a, z[1:n])
    proj!(prob.base, x[n+1:n+ns], z[n+1:n+ns])
  end
  return gz
end

###########################################################################
# Optional methods
###########################################################################
function objgrad!(prob::SlackOptiModel, x::AbstractVector, dfx::AbstractVector)
  @lencheck prob.meta.nvar x dfx
  n = prob.base.meta.nvar
  ns = prob.meta.nvar - n
  @views begin
    fx = objgrad!(prob.base, x[1:n], dfx[1:n])
    dfx[n+1:n+ns] .= 0
  end
  return fx
end

function proj(prob::SlackOptiModel, v::AbstractVector)
  @lencheck prob.meta.ncon v
  fill!(v, 0.0)
  return v
end

function dist(prob::SlackOptiModel, v::AbstractVector)
  @lencheck prob.meta.ncon v
  return norm( v )
end
