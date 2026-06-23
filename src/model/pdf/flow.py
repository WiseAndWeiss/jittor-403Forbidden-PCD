"""Invertible flow blocks (Jittor) for the PD-Flow denoiser.

Simplified, robust normalizing-flow stack (ActNorm + affine coupling, RealNVP /
Glow style) standing in for PD-LTS's implicit monotone residual flow. Each block
is exactly invertible in closed form (no fixed-point solver / spectral norm),
which ports cleanly to Jittor while preserving the method's core idea: an
invertible map to a latent space where noise is disentangled into a channel
subspace and zeroed out.
"""

import jittor as jt
from jittor import nn


class ActNorm(nn.Module):
    """Per-channel affine with data-dependent initialization (Glow ActNorm)."""

    def __init__(self, channel):
        super().__init__()
        self.logs = jt.zeros((1, 1, channel))   # trainable log-scale
        self.bias = jt.zeros((1, 1, channel))    # trainable bias
        self.inited = False
        self.eps = 1e-6

    def _init(self, x):
        with jt.no_grad():
            mean = x.mean(dims=[0, 1], keepdims=True)
            var = ((x - mean) ** 2).mean(dims=[0, 1], keepdims=True)
            std = jt.sqrt(var) + self.eps
            self.bias.assign(-mean / std)
            self.logs.assign(-jt.log(std))
        self.inited = True

    def execute(self, x):
        if not self.inited:
            self._init(x)
        return x * jt.exp(self.logs) + self.bias

    def inverse(self, z):
        return (z - self.bias) * jt.exp(-self.logs)


class AffineCoupling(nn.Module):
    """Affine coupling layer. Splits channels in half; one half conditions an
    MLP that scales+shifts the other. ``swap`` alternates the roles between
    blocks so every channel is transformed. Zero-initialized so it starts as the
    identity (stable training)."""

    def __init__(self, channel, hidden=64, swap=False):
        super().__init__()
        assert channel % 2 == 0, "AffineCoupling needs an even channel count"
        self.swap = swap
        self.half = channel // 2
        self.net = nn.Sequential(
            nn.Linear(self.half, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.half * 2),
        )
        nn.init.constant_(self.net[-1].weight, 0.0)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def _params(self, a):
        st = self.net(a)
        s, t = st[..., :self.half], st[..., self.half:]
        s = jt.tanh(s)                # bounded log-scale for stability
        return s, t

    def execute(self, x):
        a, b = x[..., :self.half], x[..., self.half:]
        if self.swap:
            a, b = b, a
        s, t = self._params(a)
        b = b * jt.exp(s) + t
        out = (b, a) if self.swap else (a, b)
        return jt.concat(out, dim=-1)

    def inverse(self, z):
        if self.swap:
            b, a = z[..., :self.half], z[..., self.half:]
        else:
            a, b = z[..., :self.half], z[..., self.half:]
        s, t = self._params(a)
        b = (b - t) * jt.exp(-s)
        out = (b, a) if self.swap else (a, b)
        return jt.concat(out, dim=-1)


class FlowBlock(nn.Module):
    """One flow step: ActNorm followed by an affine coupling."""

    def __init__(self, channel, hidden=64, swap=False):
        super().__init__()
        self.actnorm = ActNorm(channel)
        self.coupling = AffineCoupling(channel, hidden=hidden, swap=swap)

    def execute(self, x):
        return self.coupling(self.actnorm(x))

    def inverse(self, z):
        return self.actnorm.inverse(self.coupling.inverse(z))
