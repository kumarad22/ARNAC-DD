from collections import deque
import numpy as np
import torch


# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self, shape, device=None):
        self._n = 0
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._M = torch.zeros(shape, device=self.device, dtype=torch.float32)
        self._S = torch.zeros(shape, device=self.device, dtype=torch.float32)

    def push(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        else:
            x = x.to(self.device, dtype=torch.float32)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.clone()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = n

    @property
    def mean(self):
        return self._M

    @mean.setter
    def mean(self, M):
        self._M = M.to(self.device)

    @property
    def sum_square(self):
        return self._S

    @sum_square.setter
    def sum_square(self, S):
        self._S = S.to(self.device)

    @property  
    def var(self):
        if self._n == 0:
            return torch.zeros_like(self._M)
        elif self._n == 1:
            return torch.zeros_like(self._M)
        else:
            return self._S / (self._n - 1)

    @property
    def std(self):
        return torch.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    def __init__(self, shape, demean=True, destd=True, clip=10.0, device=None):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rs = RunningStat(shape, device=self.device)

    def __call__(self, x, update=True):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        else:
            x = x.to(self.device, dtype=torch.float32)

        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = torch.clamp(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape