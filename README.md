# OptiMo.jl

OptiMo is a [Julia](https://julialang.org/) module providing a modeling tool for mathematical optimization.

An ```AbstractOptiModel``` represents problems in the form
```
    optimize     f(x) + g(x)
     x ∈ Rⁿ
   subject to    c(x) ∈ S
```
where ```f(x): Rⁿ --> R``` is a smooth objective function, ```g(x): Rⁿ --> R ∪ ∞``` is a proximable objective function, ```c(x): Rⁿ --> Rᵐ``` are the constraint functions, and ```S ⊆ Rᵐ``` is a closed set.


See also [JuMP](https://jump.dev/) and 
[NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).
