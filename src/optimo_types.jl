
export is_unconstrained

# Base type for an optimization model.
abstract type AbstractOptiModel end

# Base type for metadata related to an optimization model.
abstract type AbstractOptiModelMeta end

struct OptiModelMeta <: AbstractOptiModelMeta

  # A composite type that represents the main features of
  # the optimization problem
  #
  #  optimize     f(x) + g(x)
  #  subject to   c(x) in S
  #
  # where x         is an nvar-dimensional vector,
  #       f         is a differentiable real-valued objective function,
  #       g         is a proximable extended-real-valued objective function,
  #       c         is a vector-valued differentiable constraint function,
  #       S         is an ncon-dimensional closed, projectable set,
  #       optimize  is either "minimize" or "maximize".

  nvar     ::Int          # number of variables
  ncon     ::Int          # number of constraints
  x0       ::Vector       # initial primal guess
  y0       ::Vector       # initial dual guess
  minimize ::Bool         # true if optimize == minimize
  name     ::String       # problem name

  function OptiModelMeta( nvar, ncon; x0=zeros(nvar,), y0=zeros(ncon,), minimize=true, name="Generic" )
    if (nvar < 1) || (ncon < 0)
      error("Nonsensical dimensions")
    end

    @lencheck nvar x0
    @lencheck ncon y0

    new( nvar, ncon, x0, y0, minimize, name )
  end
end

"""
    is_unconstrained(prob)
    is_unconstrained(meta)
"""
is_unconstrained(meta::OptiModelMeta) = meta.ncon == 0
is_unconstrained(prob::AbstractOptiModel) = is_unconstrained( prob.meta )

