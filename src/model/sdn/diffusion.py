"""Score-based diffusion schedule for point cloud denoising.

This file implements the *scaling-elimination* formulation from
"Adaptive and Iterative Point Cloud Denoising with Score-Based Diffusion Model"
(Wang et al., 2025), Section 3.1.

The key idea (paper Eq. 1) is that, unlike the generative DDPM which scales the
clean signal by ``sqrt(alpha_bar_t)``, we keep the *mean* of the point cloud
fixed and only inject noise::

    q(x_t | x_0) = N(x_t ; x_0, ((1 - alpha_bar_t) / alpha_bar_t) I)
    =>  x_t = x_0 + sigma_bar_t * z,   z ~ N(0, I)

with the total noise variance (paper Eq. 3)::

    sigma_bar_t^2 = (1 - alpha_bar_t) / alpha_bar_t,
    alpha_bar_t   = prod_{s=1..t} (1 - beta_s).

Following the paper we use a linear schedule ``beta_t`` with ``beta_0 = 0`` and
``beta_T = 2e-6`` over ``T = 1000`` steps, which yields ``sigma_bar_t`` in
``(0, 0.03]`` -- i.e. the network learns to denoise up to ~3% noise.
"""

from typing import List, Tuple

import numpy as np


class DiffusionSchedule:
    """Pre-computed diffusion schedule (all arrays indexed by timestep ``t``).

    Index 0 is the clean point cloud (``sigma_bar_0 = 0``); index ``T`` is the
    most-noisy point cloud (``sigma_bar_T ~= 0.03``).
    """

    def __init__(self, num_steps: int = 1000, beta_T: float = 2e-6):
        self.T = int(num_steps)
        self.beta_T = float(beta_T)

        # Linear beta schedule with beta_0 = 0 (paper Sec. 4.2). Computed in
        # float64 because sigma_bar relies on a long cumulative product of
        # numbers extremely close to 1.
        t = np.arange(self.T + 1, dtype=np.float64)          # 0..T
        betas = self.beta_T * t / self.T                     # beta_0 = 0
        alphas = 1.0 - betas
        alpha_bar = np.cumprod(alphas)
        alpha_bar[0] = 1.0                                   # guard for t=0

        # Total injected noise std up to step t (paper Eq. 3).
        sigma_bar2 = (1.0 - alpha_bar) / alpha_bar
        sigma_bar = np.sqrt(np.clip(sigma_bar2, 0.0, None))

        self.betas = betas
        self.alpha_bar = alpha_bar
        self.sigma_bar = sigma_bar          # sigma_bar[t]
        self.sigma_bar2 = sigma_bar2

    # ------------------------------------------------------------------ #
    # Scalar look-ups (return python floats / numpy arrays)
    # ------------------------------------------------------------------ #
    def sigma_of(self, t):
        """sigma_bar_t for an int or int-array of timesteps."""
        return self.sigma_bar[np.asarray(t, dtype=np.int64)]

    def update_coef(self, t, s):
        """Deterministic (eta=0) reverse-step coefficient, paper Eq. 7.

        ``x_s = x_t + coef * score``, where ``s = t - Delta < t``::

            coef = 1 - sqrt( (1 - alpha_bar_s) * alpha_bar_t
                             / ((1 - alpha_bar_t) * alpha_bar_s) )

        ``t``/``s`` may be scalars or numpy arrays of equal shape. For ``s = 0``
        (``alpha_bar_0 = 1``) this evaluates to ``1`` -> jump straight onto the
        estimated surface.
        """
        t = np.asarray(t, dtype=np.int64)
        s = np.asarray(s, dtype=np.int64)
        ab_t = self.alpha_bar[t]
        ab_s = self.alpha_bar[s]
        num = (1.0 - ab_s) * ab_t
        den = (1.0 - ab_t) * ab_s
        den = np.clip(den, 1e-20, None)
        coef = 1.0 - np.sqrt(np.clip(num / den, 0.0, None))
        return coef

    # ------------------------------------------------------------------ #
    # Adaptive schedule arrangement (paper Sec. 4.3 / Algorithm 2)
    # ------------------------------------------------------------------ #
    def find_tau(self, sigma2_est: float) -> int:
        """Linear search for tau_hat whose sigma_bar^2 best matches the
        estimated noise variance (paper Eq. 16)::

            tau_hat = argmin_t | sigma_bar_t^2 - sigma2_est |
        """
        diff = np.abs(self.sigma_bar2 - float(sigma2_est))
        tau = int(np.argmin(diff))
        return max(tau, 1)

    def adaptive_timesteps(self, tau: int, L: int) -> List[Tuple[int, int]]:
        """Build the ``L``-step adaptive denoising schedule (Algorithm 2).

        Returns a list of ``(t, t_next)`` pairs from ``tau`` down to ``0``.
        Each element is snapped to an integer timestep on the training grid
        (``Round(l * tau / L)``) so that inference and training schedules align
        (paper "replace each element with the closest one in the training
        schedule").
        """
        L = max(int(L), 1)
        delta = tau / L
        ts = [int(round(l * delta)) for l in range(L + 1)]   # [0, .., tau]
        ts[0] = 0
        ts[-1] = tau
        pairs = []
        for l in range(L, 0, -1):       # l = L .. 1   ->  t = ts[l], next = ts[l-1]
            t_cur, t_next = ts[l], ts[l - 1]
            if t_cur <= 0:
                continue
            pairs.append((t_cur, t_next))
        return pairs
