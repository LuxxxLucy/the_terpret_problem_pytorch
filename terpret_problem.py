import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.optim.optimizer import Optimizer

from typing import TypeVar, Union, Tuple, Optional, Callable

import numpy as np
import sys


class TerpretProblem(nn.Module):
    def __init__(self,opts):
        super(TerpretProblem, self).__init__()
        self.opts=opts
        self.ms = nn.Parameter(torch.rand(opts.k-1, 2,dtype=torch.float), requires_grad=True)

    def forward(self, x):
        opts=self.opts
        self.mus = torch.cat([x, torch.nn.functional.softmax(self.ms,dim=1)], 0)
        def soft_xor_p0(i):
            j = i+1
            j = j % opts.k
            return self.mus[i, 0] * self.mus[j, 0] + self.mus[i, 1] * self.mus[j, 1]

        self.ys_eq_0 = torch.stack([ soft_xor_p0(temp) for temp in range(opts.k)],0)

        self.result = torch.log(self.ys_eq_0).sum()
        return - self.result

    def return_0_prob(self,x):
        return self.mus


class TerpretProblem_ConcreteDistribution(TerpretProblem):
    def __init__(self,opts):
        super(TerpretProblem_ConcreteDistribution, self).__init__(opts)
        self.initial_temp = 2.5
         # innitial gumbel-softmax temperature
        self.temperature = self.initial_temp
        # annealation rate of softmax temperature

        self.anneal_rate = 0.9999
        self.step=0
        self.M=opts.M # the number of samples

    def forward(self, x):
        self.step+=1
        self.temperature = self.temperature * self.anneal_rate
        def gumbel_softmax_sample(logits, temperature):
            m = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
                torch.tensor([temperature]), logits=logits
                )
            # return m.rsample(),-m.log_prob(m.sample())
            return m.rsample() # note that `rsample` is reparameterization.

        opts=self.opts

        self.result=0
        for i in range(self.M):
            self.ms_sampled = gumbel_softmax_sample(self.ms,self.temperature)

            self.mus = torch.cat([x, self.ms_sampled], 0)
            def soft_xor_p0(i):
                j = i+1
                j = j % opts.k
                return self.mus[i, 0] * self.mus[j, 0] + self.mus[i, 1] * self.mus[j, 1]

            self.ys_eq_0 = torch.stack([ soft_xor_p0(temp) for temp in range(opts.k)],0)
            self.result += torch.log(self.ys_eq_0).sum()

        return - self.result / self.M

    def return_0_prob(self,x):
        self.mus = torch.cat([x, torch.nn.functional.softmax(self.ms,dim=1)], 0)
        return self.mus

class Binarize(Function):
    '''used where binarization of real-valued parameter into binary values are needed.'''
    clip_value = 0.5

    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)

        # output = inp.sign()
        output = torch.where(inp > Binarize.clip_value, torch.ones_like(inp).type_as(inp), torch.zeros_like(inp).type_as(inp))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp: Tensor = ctx.saved_tensors[0]

        # clipped = inp.abs() <= Binarize.clip_value
        clipped = (inp-0.5).abs() <= 0.5

        output = torch.zeros(inp.size()).to(grad_output.device)
        output[clipped] = 1
        output[~clipped] = 0

        return output * grad_output

binarize = Binarize.apply

class TerpretProblem_STE(TerpretProblem):
    def __init__(self,opts,adaptive_noise=False):
        super(TerpretProblem_STE, self).__init__(opts)
        self.opts=opts
        self.parameter = nn.Parameter(torch.rand(opts.k-1, 1,dtype=torch.float)*0.1+0.45, requires_grad=True)


    def forward(self, x):
        opts=self.opts
        temp = binarize(self.parameter)
        ms = torch.cat((1-temp, temp), dim=1)
        self.mus = torch.cat([x, ms], 0)

        def soft_xor_p0(i):
            j = i+1
            j = j % opts.k
            return self.mus[i, 0] * self.mus[j, 0] + self.mus[i, 1] * self.mus[j, 1]

        self.ys_eq_0 = torch.stack([ soft_xor_p0(temp) for temp in range(opts.k)],0)

        eps=1e-7
        self.result = torch.log(self.ys_eq_0+eps).sum()
        return - self.result

class Bop(Optimizer):
    def __init__(
        self,
        binary_params,
        ar = 0.0001,
        threshold = 0.00001,
        continuous_optimizer = None
    ):
        if not 0 < ar < 1:
            raise ValueError(
                "given adaptivity rate {} is invalid; should be in (0, 1) (excluding endpoints)".format(
                    ar
                )
            )
        if threshold < 0:
            raise ValueError(
                "given threshold {} is invalid; should be > 0".format(threshold)
            )

        self.total_weights = {}
        if continuous_optimizer is None:
            self.use_non_binary=False
        else:
            self.use_non_binary=True
            self._adam = continuous_optimizer


        defaults = dict(adaptivity_rate=ar, threshold=threshold)
        super(Bop, self).__init__(
            binary_params, defaults
        )

    def step(self, closure: Optional[Callable[[], float]] = ..., ar=None):
        if self.use_non_binary is True:
            self._adam.step()

        flips = {None}

        for group in self.param_groups:
            params = group["params"]

            y = group["adaptivity_rate"]
            t = group["threshold"]
            flips = {}

            if ar is not None:
                y = ar

            for param_idx, p in enumerate(params):
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state["moving_average"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # state["moving_average"] = torch.zeros_like(torch.rand(p.data.size()), memory_format=torch.preserve_format)

                m = state['moving_average']
                m.mul_((1 - y))
                m.add_(grad.mul(y))
                # print(m)
                m_=m

                p_temp = torch.where(p>0.5, torch.ones_like(p), - torch.ones_like(p))

                mask = (m_.abs() >= t ) * (m_.sign() == p_temp.sign())
                mask = mask.float() * 1
                flips[param_idx] = mask

                p.data.logical_xor_(mask).type(torch.float)

            # print('# of flip',flips)
        return flips,grad

    def zero_grad(self) -> None:
        super().zero_grad()
        if self.use_non_binary is True:
            self._adam.zero_grad()

class TerpretProblem_Bop(TerpretProblem):
    def __init__(self,opts,adaptive_noise=False):
        super(TerpretProblem_Bop, self).__init__(opts)
        self.opts=opts

        random_noise = torch.rand(opts.k-1, 1,dtype=torch.float)
        init_value = torch.where(random_noise>0.5,torch.ones_like(random_noise),torch.zeros_like(random_noise))
        self.parameter = nn.Parameter(init_value, requires_grad=True)

    def parameters(self):
        return [self.parameter]

    def forward(self, x):
        opts=self.opts
        temp = self.parameter
        ms = torch.cat((1-temp, temp), dim=1)
        self.mus = torch.cat([x, ms], 0)

        def soft_xor_p0(i):
            j = i+1
            j = j % opts.k
            return self.mus[i, 0] * self.mus[j, 0] + self.mus[i, 1] * self.mus[j, 1]


        self.ys_eq_0 = torch.stack([ soft_xor_p0(temp) for temp in range(opts.k)],0)

        eps=1e-7
        self.result = torch.log(self.ys_eq_0+eps).sum()
        return - self.result
