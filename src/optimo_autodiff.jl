using ForwardDiff

export AutodiffOptiModel

mutable struct AutodiffOptiModel <: AbstractOptiModel
  meta::OptiModelMeta
  f
  c
end

#  An AutodiffOptiModel instance is an AbstractOptiModel using ForwardDiff
#  to compute the derivatives. The represented problem is defined as
#
#    optimize    f(x) + g(x)
#    subject to  c(x) in S
#
#  Syntax:
#    AutodiffOptiModel(f, x0)
#    AutodiffOptiModel(f, x0, c, y0)

#=========================================================================#
function AutodiffOptiModel end

"
      AutodiffOptiModel(f, x0)
"
function AutodiffOptiModel(f, x0::AbstractVector{T};
                     name::String = "Generic", minimize::Bool=true) where T
  nvar = length(x0)
  @lencheck nvar x0
  meta = OptiModelMeta(nvar, 0, x0=x0, minimize=minimize, name=name)
  return AutodiffOptiModel(meta, f, x->T[])
end

"
      AutodiffOptiModel(f, x0, c, y0)
"
function AutodiffOptiModel(f, x0::AbstractVector{T}, c, y0::AbstractVector{T};
                     name::String = "Generic", minimize::Bool=true) where T
  nvar = length(x0)
  ncon = length(y0)
  @lencheck nvar x0
  @lencheck ncon y0
  meta = OptiModelMeta(nvar, ncon, x0=x0, y0=y0, minimize=minimize, name=name)
  return AutodiffOptiModel(meta, f, c)
end

#=========================================================================#
function obj(prob::AutodiffOptiModel, x::AbstractVector)
  @lencheck prob.meta.nvar x
  return prob.f(x)
end

function grad!(prob::AutodiffOptiModel, x::AbstractVector, dfx::AbstractVector)
  @lencheck prob.meta.nvar x dfx
  ForwardDiff.gradient!(dfx, prob.f, x)
  return dfx
end

#function prox!(prob::AutodiffOptiModel, x::AbstractVector, a::Real, z::AbstractVector)
#  @lencheck prob.meta.nvar x z
#  prox!(z, prob.g, x, a)
#  return z
#end

function cons!(prob::AutodiffOptiModel, x::AbstractVector, c::AbstractVector)
  @lencheck prob.meta.nvar x
  @lencheck prob.meta.ncon c
  c .= prob.c(x)
  return c
end

#function jac(prob::AutodiffOptiModel, x::AbstractVector)
#  @lencheck prob.meta.nvar x
#  return ForwardDiff.jacobian(prob.c, x)
#end

function jprod!(prob::AutodiffOptiModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck prob.meta.nvar x v
  @lencheck prob.meta.ncon Jv
  Jv .= ForwardDiff.derivative(t -> prob.c(x + t * v), 0)
  return Jv
end

function jtprod!(prob::AutodiffOptiModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck prob.meta.nvar x Jtv
  @lencheck prob.meta.ncon v
  Jtv .= ForwardDiff.gradient(x -> dot(prob.c(x), v), x)
  return Jtv
end

#function objprox!(prob::AutodiffOptiModel, x::AbstractVector, a, z::AbstractVector)
#  @lencheck prob.meta.nvar x z
#  gz = prox!(z, prob.g, x, a)
#  return gz, z
#end
#
#function proj!(prob::AutodiffOptiModel, v::AbstractVector, p::AbstractVector)
#  @lencheck prob.meta.ncon v p
#  proj!(p, prob.S, v)
#  return p
#end


