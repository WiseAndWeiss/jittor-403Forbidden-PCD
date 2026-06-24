"""PD-LTS monotone residual flow block (Jittor port of models/layers/iMonotoneBlock.py).

The block is the resolvent of a monotone operator built from a spectrally-normalized
net g (Lipschitz < 1):

    forward(x):  solve  w = sqrt(2)*x - g(w)   (Banach fixed-point),  y = sqrt(2)*w - x
    inverse(y):  solve  w = sqrt(2)*y + g(w),                          x = sqrt(2)*w - y

These are exact inverses (verified algebraically). PD-LTS also estimates the
log-det-Jacobian here, but for FBM denoising that term is unused (logp is hard-coded
to 0 in deflow.py), so we drop the whole stochastic log-det machinery and keep only
the invertible map.

Gradient: backprop through an unrolled Banach solve is too deep (Lip ~0.7 -> needs
many iters). We use Jacobian-Free Backpropagation (JFB, Fung et al. 2022): solve the
fixed point under no_grad, then take ONE grad-carrying application of g. This gives a
cheap, memory-light, well-behaved gradient that trains implicit/DEQ-style nets.
"""

import math
import jittor as jt
from jittor import nn

from .spectral import SpectralLinear
from ..pdf.flow import ActNorm

SQRT2 = math.sqrt(2.0)


class FCNet(nn.Module):
    """Spectrally-normalized MLP g (Lipschitz < 1). PD-LTS light: nhidden=2, idim=64,
    ELU activations, optional pre-activation."""

    def __init__(self, channel, idim=64, nhidden=2, coeff=0.9, preact=False, n_iter=1):
        super().__init__()
        layers = []
        if preact:
            layers.append(nn.ELU())
        d = channel
        for _ in range(nhidden):
            layers.append(SpectralLinear(d, idim, coeff=coeff, n_iterations=n_iter))
            layers.append(nn.ELU())
            d = idim
        layers.append(SpectralLinear(d, channel, coeff=coeff, n_iterations=n_iter))
        self.net = nn.Sequential(*layers)
        self._sn = [m for m in layers if isinstance(m, SpectralLinear)]

    def set_freeze(self, flag):
        for m in self._sn:
            m.freeze_sn = flag

    def refresh_sn(self, iters=10):
        for m in self._sn:
            m.refresh(iters)

    def execute(self, x):
        return self.net(x)


class MonotoneBlock(nn.Module):
    def __init__(self, channel, idim=64, nhidden=2, coeff=0.9, preact=False,
                 n_fixed_iter=30):
        super().__init__()
        self.g = FCNet(channel, idim, nhidden, coeff, preact)
        self.n_fixed_iter = n_fixed_iter

    def _solve(self, a, sign):
        """Solve w = a + sign*g(w) by Banach iteration (no grad), then one
        grad-carrying application (JFB). Spectral norm is frozen during the solve
        (the weight is constant across iterations -> updating u,v every iter is
        pure waste); it updates once on the final grad-carrying call."""
        if self.is_training():
            self.g.refresh_sn(10)          # re-converge u,v to current weight, once
        self.g.set_freeze(True)
        with jt.no_grad():
            w = a
            for _ in range(self.n_fixed_iter):
                w = a + sign * self.g(w)
        self.g.set_freeze(False)
        # JFB: re-evaluate g once at the converged point with grad flowing to params
        return a + sign * self.g(w.detach())

    def execute(self, x):                      # forward
        a = SQRT2 * x
        w = self._solve(a, -1.0)
        return SQRT2 * w - x

    def inverse(self, y):
        a = SQRT2 * y
        w = self._solve(a, +1.0)
        return SQRT2 * w - y


class FlowAssembly(nn.Module):
    """One PD-LTS flow module: [MonotoneBlock, ActNorm, MonotoneBlock(preact), ActNorm]."""

    def __init__(self, channel, idim=64, nhidden=2, coeff=0.9, n_fixed_iter=30):
        super().__init__()
        self.chain = nn.ModuleList([
            MonotoneBlock(channel, idim, nhidden, coeff, preact=False, n_fixed_iter=n_fixed_iter),
            ActNorm(channel),
            MonotoneBlock(channel, idim, nhidden, coeff, preact=True, n_fixed_iter=n_fixed_iter),
            ActNorm(channel),
        ])

    def execute(self, x):
        for m in self.chain:
            x = m(x)
        return x

    def inverse(self, z):
        for m in self.chain[::-1]:
            z = m.inverse(z)
        return z
