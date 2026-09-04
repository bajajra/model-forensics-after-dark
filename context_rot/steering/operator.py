"""Coordinate-space read/write on the validated R-lens (causal control experiment).

The experiment intervenes on a *coordinate* in the span of a few declared normative lens
vectors, not on a single vector. That distinction matters: the lens rows for semantically close
tokens are correlated (cosines up to ~0.5 were measured in E2's T48 audit), so naively adding
``alpha * (v1 + v2 + v3)`` displaces each coordinate by an amount that depends on how correlated
the three happen to be. Reading with the pseudo-inverse and writing back through ``V`` gives a
displacement that is exactly the one requested::

    c  = V^+ h            (local coordinates)
    h' = h + V @ dc       (write)
    V^+ h' = c + dc       (exactly, when V has full column rank)

so ``dc`` is delivered verbatim and the component of ``h`` orthogonal to ``span(V)`` is
untouched by construction. Both properties are asserted numerically in
``tests/unit/test_steering.py`` rather than taken on faith.

Scaling is in empirical standard deviations of each coordinate, estimated on an
outcome-independent corpus, so ``alpha`` means the same thing across concepts with different
natural scales.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class CoordinateOperator:
    """Read/write operator for a fixed set of lens directions at one capture site."""

    #: [d, m] — lens vectors as COLUMNS, unit-norm rows of the fitted lens.
    V: torch.Tensor
    #: [m, d] — Moore-Penrose pseudo-inverse of V, precomputed.
    V_pinv: torch.Tensor
    #: [m] — empirical SD of each coordinate on the reference corpus.
    sigma: torch.Tensor
    #: [m] — the declared unit direction in coordinate space.
    u: torch.Tensor
    tokens: tuple[str, ...]
    layer: int

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        sigma: np.ndarray,
        tokens: tuple[str, ...],
        layer: int,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
        u: np.ndarray | None = None,
    ) -> CoordinateOperator:
        """``vectors`` is [m, d] as stored in the lens npz; it is transposed to columns here."""
        V = torch.tensor(np.asarray(vectors).T, device=device, dtype=dtype)  # [d, m]
        m = V.shape[1]
        rank = int(torch.linalg.matrix_rank(V.double()).item())
        if rank < m:
            raise ValueError(
                f"the {m} selected lens vectors are rank {rank}: they are linearly dependent, so "
                "the requested coordinate displacement cannot be delivered exactly. Choose a "
                "less redundant concept set."
            )
        if u is None:
            # Declared default: raise every selected normative coordinate equally. Fixed in the
            # prespecification so that no search over concept weightings can occur.
            u_vec = np.ones(m) / np.sqrt(m)
        else:
            u_vec = np.asarray(u, dtype=float)
            u_vec = u_vec / np.linalg.norm(u_vec)
        return cls(
            V=V,
            V_pinv=torch.linalg.pinv(V),
            sigma=torch.tensor(np.asarray(sigma), device=device, dtype=dtype),
            u=torch.tensor(u_vec, device=device, dtype=dtype),
            tokens=tuple(tokens),
            layer=layer,
        )

    def coordinates(self, h: torch.Tensor) -> torch.Tensor:
        """Local coordinates of a [..., d] residual block, as [..., m]."""
        return h.to(self.V_pinv.dtype) @ self.V_pinv.T

    def displacement(self, alpha: float) -> torch.Tensor:
        """``dc = alpha * sigma * u`` — the preregistered coordinate displacement."""
        return alpha * self.sigma * self.u

    def apply(self, h: torch.Tensor, alpha: float) -> torch.Tensor:
        """Return ``h + V dc``, in ``h``'s own dtype."""
        if alpha == 0.0:
            return h
        delta = (self.V @ self.displacement(alpha)).to(h.dtype)
        return h + delta

    def delta_vector(self, alpha: float) -> torch.Tensor:
        return (self.V @ self.displacement(alpha))


def matched_random_operator(
    op: CoordinateOperator,
    *,
    seed: int,
    covariance_sqrt: torch.Tensor | None = None,
) -> CoordinateOperator:
    """A random-direction control matched to ``op`` on everything that is not its meaning.

    Matched on: capture layer, number of coordinates, coordinate-space direction ``u``, the
    per-coordinate sigma, and — the one that actually matters — the **residual-stream norm of
    the delivered displacement** at every alpha, since the write is linear in alpha.

    ``covariance_sqrt`` optionally whitens the draw so the random directions have the same
    second-order structure as the activations rather than being isotropic. Isotropic noise in a
    6,656-dimensional space is nearly orthogonal to every real activation direction, which would
    make the control trivially inert and the specificity test vacuous.

    The seed is frozen before any outcome is generated.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    d, m = op.V.shape
    draw = torch.randn(d, m, generator=generator, dtype=torch.float32)
    if covariance_sqrt is not None:
        draw = covariance_sqrt.cpu().float() @ draw
    # Orthonormalise, then rescale so the DELIVERED DISPLACEMENT has matched magnitude.
    #
    # Matching column norms is not enough and getting this wrong quietly breaks the specificity
    # test. The real lens columns are correlated — semantically close tokens had cosines up to
    # ~0.5 in E2's T48 audit — so `V @ dc` is shorter than an orthonormal draw of the same
    # column norms would give. Column-matching therefore hands the random control a *larger*
    # push than the real one (measured 3.42 vs 3.06, 12% too big), which would make a null
    # specificity result look like the concept direction losing to noise.
    #
    # The write is linear in alpha, so matching ||V dc|| at alpha=1 matches it at every alpha.
    q, _ = torch.linalg.qr(draw)
    q = q.to(op.V.device, op.V.dtype)
    reference = (op.V @ op.displacement(1.0)).norm()
    candidate = (q @ op.displacement(1.0)).norm().clamp_min(1e-12)
    V_rand = q * (reference / candidate)
    return CoordinateOperator(
        V=V_rand,
        V_pinv=torch.linalg.pinv(V_rand),
        sigma=op.sigma,
        u=op.u,
        tokens=tuple(f"<random:{seed}:{i}>" for i in range(m)),
        layer=op.layer,
    )
