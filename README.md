# the_terpret_problem

This reproduce the Parity Chain problem in the TerpreT paper and figure out a way to solve it.

TerpreT: A Probabilistic Programming Language for Program Induction, Gaunt et al. 2016.

I also write a short description [here](https://luxxxlucy.github.io/projects/2020_terpret/terpret.html)


## options

1. type == 0: continuous surrogate.
2. type == 1: using gumbel-softmax trick.
3. type == 2: using straight-through estimator (STE)
4. type == 3: using Binary Optimizer (Bop)

##

run


    python run.py –type=3 –k=128 –v=128


And you can see the Bop converges to the right solution in about 100 epochs. For k=1024, Bop converge to the right solution in about 900 epoch.


You can also comment line 15 in `run.py` and uncomment line 16 so that every time it runs with different random seed. But anyway I tested with many random seed, Bop can always reach the good solutions.
