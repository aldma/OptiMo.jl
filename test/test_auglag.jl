push!(LOAD_PATH,"/home/albertodm/Documents/optimo/src");

using OptiMo

using CUTEst, Random

# problem
problems = CUTEst.select( min_var=1, max_var=10, min_con=1, max_con=10 );
problem = problems[1]
nlp = CUTEstModel( problem )

prob = NLPOptiModel( nlp );
x = copy( prob.meta.x0 )
nvar = prob.meta.nvar
ncon = prob.meta.ncon
R = eltype( x )

y = randn(R,ncon)
mu = rand(R,ncon)

alprob = AugLagOptiModel( prob, mu, y )

mu = rand(R,ncon)
update!( alprob, mu, y )

y = randn(R,ncon)
update!( alprob, mu, y )

lx = obj( alprob, x )
lx, dlx = objgrad( alprob, x )
gz, z = objprox( alprob, x, 0.1 )

finalize( nlp )
