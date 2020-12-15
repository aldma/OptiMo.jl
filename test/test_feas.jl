using OptiMo
using CUTEst

# problempus
problems = CUTEst.select( min_var=1, max_var=100, min_con=1, max_con=100 );
problem = problems[4]
nlp = CUTEstModel( problem )
prob = NLPOptiModel( nlp );
x0 = copy( prob.meta.x0 )

xk = x0 .+ 0.1
xp = x0 .+ 0.2
dp = min.( 1.0, 1.0 ./ abs.(xp) )
wp = 1.0

# pure feasibility
fprob_1 = FeasOptiModel( prob )

# feasibility with diff starting point
fprob_2 = FeasOptiModel( prob, x0=xk )

# feasibility with indicator
fprob_3 = FeasOptiModel( prob, with_indicator=true )

# feasibility with proximal regularization
fprob_4 = FeasOptiModel( prob, x0=xk, xprox=xp, wprox=wp )
fprob_5 = FeasOptiModel( prob, x0=xk, xprox=xp, wprox=wp, dprox=dp )

# feasibility with indicator and proximal regularization
fprob_5 = FeasOptiModel( prob, x0=xk, xprox=xp, wprox=wp, dprox=dp, with_indicator=true )

# feasibility with Huber loss
fprob_6 = FeasOptiModel( prob, x0=xk, xprox=xp, wprox=wp, with_indicator=true, loss=:huber )
fprob_7 = FeasOptiModel( prob, x0=xk, xprox=xp, wprox=wp, dprox=dp, loss=:huber, huber_rho=1e-4 )

# feasibility problem with name
fprob_8 = FeasOptiModel( prob, with_indicator=true, loss=:huber, name="myfprob" )

infea = infeasibility( fprob_8, x0 )
println( infea )

finalize( nlp )
