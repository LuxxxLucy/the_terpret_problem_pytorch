# the_terpret_problem

This reproduce the Parity Chain problem in the TerpreT paper and figure out a way to solve it.

## options

1. type == 0: continuous surrogate.
2. type == 1: using gumbel-softmax trick.
3. type == 2: using straight-through estimator (STE)
4. type == 3: using Binary Optimizer (Bop)

##

run


    python run.py –type=3 –k=128 –v=128


And you can see the Bop converges to the right solution in about 100 epochs. For k=1024, Bop converge to the right solution in about 900 epoch.
