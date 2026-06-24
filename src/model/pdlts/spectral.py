"""Spectral-normalized linear layer (Jittor) for PD-LTS's monotone residual flow.

Faithful port of PD-LTS `models/layers/base/lipschitz.py::SpectralNormLinear`:
the weight matrix is soft-normalized so its spectral norm (largest singular
value) stays <= `coeff`, estimated by power iteration on the (u, v) singular
vectors. This Lipschitz bound (< 1) is what makes the residual map a contraction,
so the monotone block's forward/inverse fixed-point iterations converge (Banach).
"""

import jittor as jt
from jittor import nn


def _l2n(v, eps=1e-12):
    return v / (jt.sqrt((v * v).sum()) + eps)


class SpectralLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, coeff=0.9, n_iterations=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.n_iterations = n_iterations

        self.weight = jt.init.invariant_uniform((out_features, in_features))
        self.bias = jt.zeros((out_features,)) if bias else None
        self.freeze_sn = False             # when True, skip power-iter update (reuse u,v)

        # power-iteration singular vectors (buffers, never optimized)
        self.u = _l2n(jt.randn((out_features,))).stop_grad()
        self.v = _l2n(jt.randn((in_features,))).stop_grad()
        self._power(200)                                   # warm-up to convergence

    def _power(self, iters):
        W = self.weight.detach()
        u, v = self.u, self.v
        for _ in range(iters):
            v = _l2n(jt.matmul(W.transpose(1, 0), u))      # v <- W^T u / |.|
            u = _l2n(jt.matmul(W, v))                      # u <- W v / |.|
        self.u = u.stop_grad()
        self.v = v.stop_grad()

    def refresh(self, iters=10):
        """Re-converge u,v to the CURRENT weight. Called once per forward before a
        frozen fixed-point solve, so the (frozen) normalization is correct even
        after the optimizer changed the weight."""
        with jt.no_grad():
            self._power(iters)

    def compute_weight(self, update):
        if update:
            self._power(self.n_iterations)
        sigma = (self.u * jt.matmul(self.weight, self.v)).sum()   # u^T W v  ~ ||W||_2
        # soft normalization: only scale down when sigma exceeds coeff
        factor = jt.maximum(jt.ones(1), sigma / self.coeff)
        return self.weight / factor

    def execute(self, x):
        W = self.compute_weight(update=self.is_training() and not self.freeze_sn)
        y = jt.matmul(x, W.transpose(1, 0))
        if self.bias is not None:
            y = y + self.bias
        return y
