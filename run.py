import argparse
import sys
import torch
import torch.optim as optim
from torch.autograd import Variable
from options import parse_options
# from util import SharedLogDirichletInitializer
import random
from terpret_problem import TerpretProblem,TerpretProblem_ConcreteDistribution,TerpretProblem_STE,TerpretProblem_Bop
from terpret_problem import Bop

if __name__ == "__main__":
    opts = parse_options()

    manualSeed = 42
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


    if opts.type == 0:
        ''' type 0: continuous surrogate '''
        tp = TerpretProblem(opts)
        # optimizer = optim.Adam(tp.parameters(), lr=0.001)
        optimizer = optim.Adam(tp.parameters(), lr=0.001)

    elif opts.type == 1:
        ''' type 1: using a local reparameterization of gumble-softmax trick '''
        opts.M=1
        tp = TerpretProblem_ConcreteDistribution(opts)
        optimizer = optim.Adam(tp.parameters(), lr=0.001)
    elif opts.type == 2:
        ''' type 2: using Straight-Through Estimator (STE) '''
        tp = TerpretProblem_STE(opts)
        optimizer = optim.Adam(tp.parameters(), lr=0.001)
    elif opts.type == 3:
        ''' type 3: using Binary Optimizer (Bop) '''
        tp = TerpretProblem_Bop(opts)
        # optimizer = Bop(binary_params=tp.parameters(), ar=0.00001, threshold=0.000001, continuous_optimizer=None)
        optimizer = Bop(binary_params=tp.parameters(), ar=0.00001, threshold=0.01, continuous_optimizer=None)
    else:
        print("not defined type")
        quit()


    x = torch.tensor([[1.0,0]],dtype=torch.float,requires_grad=False)

    for epoch in range(opts.n_epochs):


        tp.zero_grad()
        output = tp(x)
        loss = output
        loss.backward()
        # print(tp.ms.grad)
        optimizer.step()


        # sys.stdout.write( str(tp.ms.data) )
        mus = tp.return_0_prob(x).detach().numpy()
        # sys.stdout.write(str(mus))
        for i in range(opts.v):
            if mus[i,0] == 1:
                sys.stdout.write("\u2588" )
            elif mus[i,0] == 0:
                # sys.stdout.write("\u25A1" )
                sys.stdout.write(" " )
            elif mus[i,0] <= 0.25:
                sys.stdout.write("\u2581" )
            elif mus[i,0] < 0.5:
                sys.stdout.write("\u2583" )
            elif mus[i,0] == 0.5 :
                sys.stdout.write("\u2584" )
            elif mus[i,0] < 0.75:
                sys.stdout.write("\u2585" )
            elif mus[i,0] < 1:
                sys.stdout.write("\u2587" )
            else:
                sys.stdout.write("\u25CC" )
        sys.stdout.write(" epoch: [%d], loss : %.8f " % (epoch, loss.data))
        sys.stdout.write("\n")
