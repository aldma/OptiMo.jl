
using Printf

export AbstractOptiOutput, OptiOutput

###########################################################################
# Status
###########################################################################
const STATUSES = Dict(
        :exception      => "unhandled exception",
        :first_order    => "first-order stationary",
        :acceptable     => "solved to within acceptable tolerances",
        :infeasible     => "problem may be infeasible",
        :max_eval       => "maximum number of function evaluations",
        :max_iter       => "maximum iteration",
        :max_time       => "maximum elapsed time",
        :neg_pred       => "negative predicted reduction",
        :not_desc       => "not a descent direction",
        :small_residual => "small residual",
        :small_step     => "step too small",
        :stalled        => "stalled",
        :unbounded      => "objective function may be unbounded from below",
        :unknown        => "unknown",
        :ieee_nan_inf   => "NaN or Inf occurred",
        :user           => "user-requested stop",
       )

###########################################################################
# Output
###########################################################################

abstract type AbstractOptiOutput end

"""
    OptiOutput(status; ...)
An OptiOutput is a struct for storing output information of solvers.
It contains the following fields:
- `status::Symbol`: Indicates the output of the solver. Use `show_statuses()` for the full list;
- `x::Vector`: The final approximation returned by the solver (default: `nothing`);
- `y::Vector`: The Lagrange multiplers wrt to the constraints (default: `nothing`);
- `objective::Real`: The objective value at `x` (default: `nothing`);
- `optimality::Real`: The dual feasibility norm at `x` (and `y`) (default: `nothing`);
- `cviolation::Real`: The primal feasibility norm at `x` (default: `nothing`);
- `iterations::Int`: The number of iterations used by the solver (default: `nothing`);
- `time::Real`: The time elapsed while running the solver, in seconds [s] (default: `nothing`);
- `solver_name::String`: The solver denomination (default: `nothing`);
- `solver::Dict{Symbol,Any}`: A solver specific dictionary.

The `status` is mandatory on construction. All other variables can be input as keyword arguments.

Notice that `OptiOutput` does not compute anything, it simply stores.
"""

mutable struct OptiOutput <: AbstractOptiOutput
    status::Symbol             # solver final status
    x::Maybe{Vector}           # primal solution
    y::Maybe{Vector}           # dual solution
    objective::Maybe{Real}     # value of objective function
    optimality::Maybe{Real}    # value of dual residual at (x,y) : optimality
    cviolation::Maybe{Real}    # value of primal residual at x : constraint violation
    iterations::Maybe{Int}     # iterations
    time::Maybe{Real}          # elapsed time
    solver_name::Maybe{String} # solver name 
    solver::Dict{Symbol,Any}   # solver info
end

function OptiOutput( status::Symbol;
                     x::Maybe{Vector}=nothing,
                     y::Maybe{Vector}=nothing,
                     objective::Maybe{Real}=nothing,
                     optimality::Maybe{Real}=nothing,
                     cviolation::Maybe{Real}=nothing,
                     iterations::Maybe{Int}=nothing,
                     time::Maybe{Real}=nothing,
                     solver_name::Maybe{String}=nothing,
                     solver::Dict{Symbol,T}=Dict{Symbol,Any}() ) where {T}
    if !(status in keys(STATUSES))
        @error "Invalid status $status. Use one of the following: " join(keys(STATUSES), ", ")
        throw(KeyError(status))
    end
    return OptiOutput( status, x, y, objective, optimality, cviolation, iterations, time, solver_name, solver )
end

###########################################################################
# Printing
###########################################################################
import Base.show, Base.print, Base.println

function show(io::IO, out::AbstractOptiOutput)
    show(io, "Output: $(getStatus(out))")
end

function disp_vector(io::IO, x::Vector)
    if length(x) == 0
        print(io, "∅")
    elseif length(x) <= 5
        Base.show_delim_array(io, x, "[", " ", "]", false)
    else
        Base.show_delim_array(io, x[1:4], "[", " ", "", false)
        print(io, " ⋯ $(x[end])]")
    end
end

function print(io::IO, out::OptiOutput; showvec::Function=disp_vector)
    println(io, "Output")
    println(io, "  status: " * getStatus(out))
    if out.objective === nothing
        println(io, "  objective value: n/a ")
    else
        println(io, "  objective value: ", out.objective)
    end
    if out.cviolation === nothing
        println(io, "  constr. violation: n/a ")
    else
        println(io, "  constr. violation: ", out.cviolation)
    end
    if out.optimality === nothing
        println(io, "  optimality: n/a ")
    else
        println(io, "  optimality: ", out.optimality)
    end
    if out.x === nothing
        println(io, "  primal solution: n/a ")
    else
        print(io, "  primal solution: "); showvec(io, out.x); println(io, "")
    end
    if out.y === nothing
        println(io, "  dual solution: n/a ")
    else
        print(io, "  dual solution: "); showvec(io, out.y); println(io, "")
    end
    if out.iterations === nothing
        println(io, "  iterations: n/a ")
    else
        println(io, "  iterations: ", out.iterations)
    end
    if out.time === nothing
        println(io, "  elapsed time: n/a ")
    else
        println(io, "  elapsed time: ", out.time, " s")
    end
    if out.solver_name === nothing
        println(io, "  solver name: n/a ")
    else
        println(io, "  solver name: ", out.solver_name)
    end
    if length( out.solver ) > 0
        println(io, "  solver info:")
        for (k,v) in out.solver
            @printf(io, "    %s: ", k)
            if v isa Vector
                showvec(io, v)
            else
                show(io, v)
            end
            println(io, "")
        end
    end
end

print(out::OptiOutput; showvec::Function=disp_vector) = print(Base.stdout, out, showvec=showvec)

println(out::OptiOutput; showvec::Function=disp_vector) = print(Base.stdout, out, showvec=showvec)

println(io::IO, out::OptiOutput; showvec::Function=disp_vector) = print(io, out, showvec=showvec)

function getStatus(out::AbstractOptiOutput)
    return STATUSES[out.status]
end
