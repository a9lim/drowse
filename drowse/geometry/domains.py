"""Manifold domains — the intrinsic geometry a manifold's nodes live on.

A domain is an embedding of an ``n``-dimensional intrinsic manifold into
``R^m`` plus a distance function.  The RBF interpolant only ever needs
pairwise distances between embedded points and the embedding map, so any
topology that can be embedded is expressible without touching it:

* :class:`BoxDomain` — a product of open intervals and circles (box/disk,
  cylinder, n-torus);
* :class:`SphereDomain` — ``S^n`` with the chordal metric;
* :class:`KleinBottleDomain` / :class:`ProjectivePlaneDomain` — the flat Klein
  quotient and ``RP^2``, with quotient-aware foot translation;
* :class:`CustomDomain` — an explicit immersion (authoring coords *are* the
  embedding coords); the identity carrier for discover coordinates and
  synthesized affine subspaces.

The steering kernel consumes :meth:`ManifoldDomain.translate_foot` (the
``along`` step), :meth:`~ManifoldDomain.retract_position` and
:meth:`~ManifoldDomain.tangent_jacobian` (the foot solve).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import torch


class ManifoldDomain(ABC):
    """Embedding of an n-D intrinsic manifold into R^m, plus a metric.

    Subclasses define :meth:`embed` (authoring coords -> embedded
    coords), :meth:`embed_jacobian` and :meth:`clamp_position`.
    :meth:`distance` defaults to the chordal (Euclidean-in-embedding)
    metric and is rarely overridden -- a periodic axis embedded as a
    circle already wraps correctly under the chordal metric.
    """

    @property
    @abstractmethod
    def intrinsic_dim(self) -> int:
        """Dimension ``n`` of the intrinsic manifold (number of authoring axes)."""

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Dimension ``m`` of the embedding space."""

    @abstractmethod
    def embed(self, coords: torch.Tensor) -> torch.Tensor:
        """Map authoring coords ``(.., n)`` to embedded coords ``(.., m)``."""

    @abstractmethod
    def embed_jacobian(self, coords: torch.Tensor) -> torch.Tensor:
        """Jacobian ``d embed / d coords``: ``(.., n) -> (.., m, n)``.

        Batch-generic over leading dims; a bare ``(n,)`` point returns
        ``(m, n)``.
        """

    @abstractmethod
    def clamp_position(self, coords: torch.Tensor) -> torch.Tensor:
        """Clamp open axes to range, wrap periodic axes; ``(.., n) -> (.., n)``."""

    @abstractmethod
    def to_spec(self) -> dict[str, Any]:
        """Serialize to the ``manifold.json`` tagged-union ``domain`` object."""

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Chordal distance between embedded points ``(.., m)`` -> ``(..,)``."""
        return torch.linalg.vector_norm(a - b, dim=-1)

    def tangent_jacobian(self, coords: torch.Tensor) -> torch.Tensor:
        """Embedding derivative in the local frame used by the foot solver."""
        return self.embed_jacobian(coords)

    def retract_position(self, coords: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Apply a local-frame step, returning authoring coordinates."""
        return self.clamp_position(coords + step)

    def _tangent(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Tangent at ``a`` pointing toward ``b`` in authoring coords.

        Default is the linear ``b - a`` (correct for any flat, non-wrapping
        domain — :class:`CustomDomain` and the open-axis part of a
        :class:`BoxDomain`).  :class:`BoxDomain` overrides it for periodic axes
        (wrap-aware minimal arc).  Consumed by :meth:`translate_foot`.
        """
        return b - a

    def translate_foot(
        self,
        p: torch.Tensor,
        origin: torch.Tensor,
        target: torch.Tensor,
        frac: float | torch.Tensor,
    ) -> torch.Tensor:
        """Translate the foot ``p`` ``(.., n)`` by the fixed offset toward target.

        The ``along`` step.  Shift *every* token's foot by the same displacement
        — the neutral→target tangent ``target − origin``, parallel-transported to
        ``p``, scaled by ``frac`` — rather than sliding each foot onto the
        absolute ``target``.  The fixed offset **preserves the per-token
        in-subspace spread**, which the kernel ablation showed is what keeps
        strong steer coherent (collapsing onto one target erases that spread and
        degenerates into looping).

        ``p_new = clamp(p + frac · transport_{origin→p}(target − origin))``.

        Flat default — transport is the identity, so the offset is the plain
        ``_tangent(origin, target)``.  :class:`SphereDomain` overrides for
        curvature; :class:`BoxDomain` inherits this but supplies a wrap-aware
        ``_tangent``.  For the affine frame (``origin = 0``) this is
        ``p + frac · target``.
        """
        return self.clamp_position(p + frac * self._tangent(origin, target))


@dataclass(frozen=True)
class BoxAxis:
    """One axis of a :class:`BoxDomain` -- open (an interval) or periodic."""

    name: str
    periodic: bool
    period: float = 1.0
    lo: float = 0.0
    hi: float = 1.0


class BoxDomain(ManifoldDomain):
    """A product of intervals and circles: box/disk, cylinder, n-torus.

    Each axis is open (contributes its raw coordinate) or periodic (a
    circle of the given ``period``, embedded as ``(cos, sin)``).  All
    axes open -> a Euclidean box; all periodic -> the n-torus; a mix ->
    a cylinder.  ``n == 1`` all-open reproduces the natural cubic spline;
    ``n == 1`` periodic is the closed-loop manifold.
    """

    def __init__(self, axes: Sequence[BoxAxis]):
        self._axes = tuple(axes)
        if not self._axes:
            raise ValueError("BoxDomain needs at least one axis")
        m = 0
        for ax in self._axes:
            m += 2 if ax.periodic else 1
        self._embed_dim = m

    @property
    def axes(self) -> tuple[BoxAxis, ...]:
        return self._axes

    @property
    def intrinsic_dim(self) -> int:
        return len(self._axes)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embed(self, coords: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for i, ax in enumerate(self._axes):
            ci = coords[..., i]
            if ax.periodic:
                w = 2.0 * math.pi / ax.period
                parts.append(torch.cos(w * ci))
                parts.append(torch.sin(w * ci))
            else:
                parts.append(ci)
        return torch.stack(parts, dim=-1)

    def embed_jacobian(self, coords: torch.Tensor) -> torch.Tensor:
        n = self.intrinsic_dim
        m = self.embed_dim
        batch = coords.shape[:-1]
        J = torch.zeros(*batch, m, n, dtype=coords.dtype, device=coords.device)
        out = 0
        for i, ax in enumerate(self._axes):
            if ax.periodic:
                w = 2.0 * math.pi / ax.period
                ci = coords[..., i]
                J[..., out, i] = -w * torch.sin(w * ci)
                J[..., out + 1, i] = w * torch.cos(w * ci)
                out += 2
            else:
                J[..., out, i] = 1.0
                out += 1
        return J

    def clamp_position(self, coords: torch.Tensor) -> torch.Tensor:
        out = coords.clone()
        for i, ax in enumerate(self._axes):
            if ax.periodic:
                out[..., i] = torch.remainder(coords[..., i], ax.period)
            else:
                out[..., i] = coords[..., i].clamp(min=ax.lo, max=ax.hi)
        return out

    def _tangent(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Linear delta, but periodic axes take the wrap-aware minimal arc.

        The translate connection on a torus/cylinder is flat (a fixed
        authoring-coord offset is parallel along every axis), so
        :meth:`ManifoldDomain.translate_foot` is inherited unchanged — it only
        needs this wrap-aware tangent for the periodic axes.
        """
        delta = b - a
        for i, ax in enumerate(self._axes):
            if ax.periodic:
                half = ax.period / 2.0
                delta[..., i] = (
                    torch.remainder(delta[..., i] + half, ax.period) - half
                )
        return delta

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": "box",
            "axes": [
                {
                    "name": ax.name,
                    "periodic": ax.periodic,
                    "period": ax.period,
                    "lo": ax.lo,
                    "hi": ax.hi,
                }
                for ax in self._axes
            ],
        }


class SphereDomain(ManifoldDomain):
    """The n-sphere ``S^n``, embedded as unit vectors in ``R^(n+1)``.

    Authoring coords are hyperspherical angles ``(phi_0, .., phi_{n-1})``;
    ``phi_0..phi_{n-2}`` are polar angles in ``[0, pi]`` and ``phi_{n-1}``
    is the azimuth in ``[0, 2*pi)``.  For ``S^2`` this is
    ``(colatitude, longitude)``.  The metric is chordal (Euclidean of the
    unit-vector embeddings) -- chosen over great-circle because chordal
    ``r**3`` stays conditionally positive definite in the ambient
    ``R^(n+1)``; a great-circle metric is a documented future option.
    """

    def __init__(self, dim: int):
        if dim < 1:
            raise ValueError(f"SphereDomain needs dim >= 1, got {dim}")
        self._dim = dim

    @property
    def intrinsic_dim(self) -> int:
        return self._dim

    @property
    def embed_dim(self) -> int:
        return self._dim + 1

    def embed(self, coords: torch.Tensor) -> torch.Tensor:
        sins = torch.sin(coords)
        coss = torch.cos(coords)
        running = torch.ones(
            coords.shape[:-1], dtype=coords.dtype, device=coords.device,
        )
        parts: list[torch.Tensor] = []
        for k in range(self._dim):
            parts.append(running * coss[..., k])
            running = running * sins[..., k]
        parts.append(running)
        return torch.stack(parts, dim=-1)

    def embed_jacobian(self, coords: torch.Tensor) -> torch.Tensor:
        n = self._dim
        batch = coords.shape[:-1]
        sins = torch.sin(coords)  # (.., n)
        coss = torch.cos(coords)  # (.., n)
        J = torch.zeros(
            *batch, n + 1, n, dtype=coords.dtype, device=coords.device,
        )
        one = torch.ones(batch, dtype=coords.dtype, device=coords.device)
        for k in range(n + 1):
            for l in range(n):
                if k < n:
                    if l > k:
                        continue
                    if l == k:
                        prefix = one
                        for i in range(k):
                            prefix = prefix * sins[..., i]
                        J[..., k, l] = -prefix * sins[..., k]
                    else:
                        term = one
                        for i in range(k):
                            term = term * (
                                coss[..., l] if i == l else sins[..., i]
                            )
                        J[..., k, l] = term * coss[..., k]
                else:
                    term = one
                    for i in range(n):
                        term = term * (
                            coss[..., l] if i == l else sins[..., i]
                        )
                    J[..., k, l] = term
        return J

    def clamp_position(self, coords: torch.Tensor) -> torch.Tensor:
        out = coords.clone()
        for i in range(self._dim - 1):
            out[..., i] = coords[..., i].clamp(min=0.0, max=math.pi)
        out[..., self._dim - 1] = torch.remainder(
            coords[..., self._dim - 1], 2.0 * math.pi
        )
        return out

    def _unembed(self, e: torch.Tensor) -> torch.Tensor:
        """Recover hyperspherical angles ``(.., n)`` from a unit vector ``(.., n+1)``.

        Inverse of :meth:`embed`. The polar angles ``phi_0..phi_{n-2}`` come from
        ``atan2(||tail||, e_k) in [0, pi]``; the azimuth ``phi_{n-1}`` uses the
        signed ``atan2(e_n, e_{n-1})`` so the full circle is recovered (the
        polar formula would fold it into ``[0, pi]``). ``clamp_position`` wraps
        the azimuth into ``[0, 2*pi)``.
        """
        n = self._dim
        angles: list[torch.Tensor] = []
        for k in range(n):
            if k < n - 1:
                tail = torch.linalg.vector_norm(e[..., k + 1:], dim=-1)
                angles.append(torch.atan2(tail, e[..., k]))
            else:
                angles.append(torch.atan2(e[..., n], e[..., n - 1]))
        return torch.stack(angles, dim=-1)

    @staticmethod
    def _sphere_log(ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        """Log map at unit vector ``ea`` toward ``eb`` — tangent in ``R^(n+1)``."""
        dot = (ea * eb).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        omega = torch.arccos(dot)
        u = eb - dot * ea
        un = torch.linalg.vector_norm(u, dim=-1, keepdim=True)
        return torch.where(
            un < 1e-9, torch.zeros_like(u), omega * u / un.clamp_min(1e-9),
        )

    @staticmethod
    def _sphere_exp(ea: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exp map at ``ea`` of tangent ``v`` — a unit vector in ``R^(n+1)``."""
        theta = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        e = torch.cos(theta) * ea + torch.sin(theta) * v / theta.clamp_min(1e-9)
        return torch.where(theta < 1e-9, ea.expand_as(e), e)

    @staticmethod
    def _sphere_transport(
        v: torch.Tensor, ea: torch.Tensor, eb: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel-transport tangent ``v`` (at ``ea``) along the geodesic to ``eb``."""
        c = (ea * eb).sum(dim=-1, keepdim=True)
        coeff = (v * eb).sum(dim=-1, keepdim=True) / (1.0 + c).clamp_min(1e-9)
        return v - coeff * (ea + eb)

    def translate_foot(
        self,
        p: torch.Tensor,
        origin: torch.Tensor,
        target: torch.Tensor,
        frac: float | torch.Tensor,
    ) -> torch.Tensor:
        """Curvature-correct translate in the unit-vector embedding.

        Parallel-transports the neutral→target tangent ``log_origin(target)`` to
        ``p`` (the flat ``target − origin`` is *not* parallel on a curved sphere)
        and exponentiates the ``frac``-scaled offset.  See
        :meth:`ManifoldDomain.translate_foot`.
        """
        ep, eo, et = self.embed(p), self.embed(origin), self.embed(target)
        offset = self._sphere_transport(self._sphere_log(eo, et), eo, ep)
        e_new = self._sphere_exp(ep, frac * offset)
        return self.clamp_position(self._unembed(e_new))

    def to_spec(self) -> dict[str, Any]:
        return {"type": "sphere", "dim": self._dim}


class KleinBottleDomain(ManifoldDomain):
    """Flat Klein quotient (u + 2π, v) ~ (u, -v), embedded without crossings in R⁴.

    Coordinates and parallel transport use the flat quotient metric, not the
    induced metric of the visualization embedding. Shortest lifts choose a
    deterministic path at the cut locus; no globally unique path exists there.
    """

    @property
    def intrinsic_dim(self) -> int:
        return 2

    @property
    def embed_dim(self) -> int:
        return 4

    def embed(self, coords: torch.Tensor) -> torch.Tensor:
        u, v = coords.unbind(-1)
        radius = 2 + v.cos()
        return torch.stack((radius * u.cos(), radius * u.sin(),
                            v.sin() * (u / 2).cos(), v.sin() * (u / 2).sin()), -1)

    def embed_jacobian(self, coords: torch.Tensor) -> torch.Tensor:
        u, v = coords.unbind(-1)
        radius = 2 + v.cos()
        du = torch.stack((-radius * u.sin(), radius * u.cos(),
                          -0.5 * v.sin() * (u / 2).sin(),
                          0.5 * v.sin() * (u / 2).cos()), -1)
        dv = torch.stack((-v.sin() * u.cos(), -v.sin() * u.sin(),
                          v.cos() * (u / 2).cos(), v.cos() * (u / 2).sin()), -1)
        return torch.stack((du, dv), -1)

    def clamp_position(self, coords: torch.Tensor) -> torch.Tensor:
        u, v = coords.unbind(-1)
        turns = torch.floor(u / (2 * math.pi))
        sign = 1 - 2 * torch.remainder(turns, 2)
        return torch.stack((u.remainder(2 * math.pi),
                            (sign * v).remainder(2 * math.pi)), -1)

    def _lift(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = torch.broadcast_tensors(self.clamp_position(a), self.clamp_position(b))
        turns = a.new_tensor([0, -1, 1])
        signs = 1 - 2 * turns.remainder(2)
        du = b[..., 0, None] + 2 * math.pi * turns - a[..., 0, None]
        dv = (signs * b[..., 1, None] - a[..., 1, None] + math.pi).remainder(2 * math.pi) - math.pi
        choice = (du.square() + dv.square()).argmin(-1, keepdim=True)
        delta = torch.cat((du.gather(-1, choice), dv.gather(-1, choice)), -1)
        return delta, signs.expand_as(du).gather(-1, choice)

    def translate_foot(self, p: torch.Tensor, origin: torch.Tensor, target: torch.Tensor,
                       frac: float | torch.Tensor) -> torch.Tensor:
        offset, _ = self._lift(origin, target)
        _, sign = self._lift(origin, p)
        du, dv = torch.broadcast_tensors(offset[..., 0], offset[..., 1] * sign[..., 0])
        transported = torch.stack((du, dv), -1)
        return self.clamp_position(self.clamp_position(p) + frac * transported)

    def to_spec(self) -> dict[str, Any]:
        return {"type": "klein"}

    def project_mean(self, mean: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        radius = torch.hypot(mean[..., 0], mean[..., 1])
        u = torch.atan2(mean[..., 1], mean[..., 0]).remainder(2 * math.pi)
        sine = mean[..., 2] * (u / 2).cos() + mean[..., 3] * (u / 2).sin()
        v = torch.atan2(sine, radius - 2)
        position = self.clamp_position(torch.stack((u, v), -1))
        ambiguous = (radius < 1e-6) | (torch.hypot(sine, radius - 2) < 1e-6)
        return torch.where(ambiguous[..., None], reference, position)


class ProjectivePlaneDomain(ManifoldDomain):
    """RP² as antipodal unit vectors, with a nonsingular Veronese embedding in R⁵.

    Authoring coordinates are the two spherical angles of a representative.
    Paths use the round quotient metric and lift to the nearest antipode.
    Orthogonal representatives lie on the cut locus: the canonical lift breaks
    that genuine path ambiguity deterministically.
    """

    def __init__(self):
        self._sphere = SphereDomain(2)

    @property
    def intrinsic_dim(self) -> int:
        return 2

    @property
    def embed_dim(self) -> int:
        return 5

    def embed(self, coords: torch.Tensor) -> torch.Tensor:
        x, y, z = self._sphere.embed(coords).unbind(-1)
        return torch.stack(((x.square() - y.square()) / math.sqrt(2),
                            (x.square() + y.square() - 2 * z.square()) / math.sqrt(6),
                            math.sqrt(2) * x * y, math.sqrt(2) * x * z,
                            math.sqrt(2) * y * z), -1)

    def embed_jacobian(self, coords: torch.Tensor) -> torch.Tensor:
        return self._veronese_jacobian(self._sphere.embed(coords)) @ self._sphere.embed_jacobian(coords)

    @staticmethod
    def _veronese_jacobian(vector: torch.Tensor) -> torch.Tensor:
        x, y, z = vector.unbind(-1)
        zero = torch.zeros_like(x)
        r2, r6 = math.sqrt(2), math.sqrt(6)
        return torch.stack((
            torch.stack((r2 * x, -r2 * y, zero), -1),
            torch.stack((2 * x / r6, 2 * y / r6, -4 * z / r6), -1),
            torch.stack((r2 * y, r2 * x, zero), -1),
            torch.stack((r2 * z, zero, r2 * x), -1),
            torch.stack((zero, r2 * z, r2 * y), -1),
        ), -2)

    def _cover_frame(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vector = self._representative(coords)
        axis = torch.nn.functional.one_hot(vector.abs().argmin(-1), 3).to(vector)
        first = torch.linalg.cross(vector, axis)
        first = first / torch.linalg.vector_norm(first, dim=-1, keepdim=True)
        return vector, torch.stack((first, torch.linalg.cross(vector, first)), -1)

    def tangent_jacobian(self, coords: torch.Tensor) -> torch.Tensor:
        vector, frame = self._cover_frame(coords)
        return self._veronese_jacobian(vector) @ frame

    def retract_position(self, coords: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        vector, frame = self._cover_frame(coords)
        tangent = (frame @ step.unsqueeze(-1)).squeeze(-1)
        return self.clamp_position(self._sphere._unembed(self._sphere._sphere_exp(vector, tangent)))

    @staticmethod
    def _canonical(vector: torch.Tensor) -> torch.Tensor:
        pivot = vector.abs().argmax(-1, keepdim=True)
        return vector * torch.where(vector.gather(-1, pivot) < 0, -1.0, 1.0)

    def clamp_position(self, coords: torch.Tensor) -> torch.Tensor:
        vector = self._canonical(self._sphere.embed(coords))
        return self._sphere.clamp_position(self._sphere._unembed(vector))

    def _representative(self, coords: torch.Tensor) -> torch.Tensor:
        return self._canonical(self._sphere.embed(coords))

    @staticmethod
    def _nearest(reference: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return vector * torch.where((reference * vector).sum(-1, keepdim=True) < 0, -1.0, 1.0)

    def translate_foot(self, p: torch.Tensor, origin: torch.Tensor, target: torch.Tensor,
                       frac: float | torch.Tensor) -> torch.Tensor:
        eo = self._representative(origin)
        ep = self._nearest(eo, self._representative(p))
        et = self._nearest(eo, self._representative(target))
        offset = self._sphere._sphere_transport(self._sphere._sphere_log(eo, et), eo, ep)
        point = self._sphere._sphere_exp(ep, frac * offset)
        return self.clamp_position(self._sphere._unembed(point))

    def to_spec(self) -> dict[str, Any]:
        return {"type": "projective", "dim": 2}

    def project_mean(self, mean: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        a, b, c, d, e = mean.unbind(-1)
        r2, r6 = math.sqrt(2), math.sqrt(6)
        matrix = torch.stack((
            torch.stack((a/r2 + b/r6, c/r2, d/r2), -1),
            torch.stack((c/r2, -a/r2 + b/r6, e/r2), -1),
            torch.stack((d/r2, e/r2, -2*b/r6), -1),
        ), -2)
        values, vectors = torch.linalg.eigh(matrix.cpu() if matrix.device.type == "mps" else matrix)
        values, vectors = values.to(mean.device), vectors.to(mean.device)
        position = self.clamp_position(self._sphere._unembed(vectors[..., -1]))
        ambiguous = values[..., -1] - values[..., -2] < 1e-6
        return torch.where(ambiguous[..., None], reference, position)


class CustomDomain(ManifoldDomain):
    """An explicit immersion: authoring coords *are* the embedding coords.

    The escape hatch for topologies the structured domains do not cover,
    reachable by authoring an explicit set of embedding coordinates per
    node, and the identity carrier for discover-mode coordinates and
    synthesized affine subspaces.  The metric is chordal, so intrinsic
    (geodesic) distances are approximated; near a self-approaching seam
    this distorts the fit.
    """

    def __init__(self, embed_dim: int, bounds: Sequence[Sequence[float]] | None = None):
        if embed_dim < 1:
            raise ValueError(f"CustomDomain needs embed_dim >= 1, got {embed_dim}")
        self._embed_dim = embed_dim
        self._bounds = (
            tuple((float(lo), float(hi)) for lo, hi in bounds)
            if bounds is not None
            else None
        )

    @property
    def intrinsic_dim(self) -> int:
        return self._embed_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embed(self, coords: torch.Tensor) -> torch.Tensor:
        return coords

    def embed_jacobian(self, coords: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(
            self._embed_dim, dtype=coords.dtype, device=coords.device,
        )
        batch = coords.shape[:-1]
        if batch:
            eye = eye.expand(*batch, self._embed_dim, self._embed_dim)
        return eye

    def clamp_position(self, coords: torch.Tensor) -> torch.Tensor:
        return coords

    def to_spec(self) -> dict[str, Any]:
        return {
            "type": "custom",
            "embed_dim": self._embed_dim,
            "bounds": (
                None
                if self._bounds is None
                else [[lo, hi] for lo, hi in self._bounds]
            ),
        }


def validate_domain_spec(spec: Any) -> dict[str, Any]:
    """Validate the exact current tagged-union domain wire shape."""
    if not isinstance(spec, dict):
        raise ValueError("manifold domain must be an object")
    kind = spec.get("type")
    if kind == "box":
        if set(spec) != {"type", "axes"} or not isinstance(spec["axes"], list) or not spec["axes"]:
            raise ValueError("box domain requires exactly type + non-empty axes")
        for index, axis in enumerate(spec["axes"]):
            if not isinstance(axis, dict) or set(axis) != {
                "name", "periodic", "period", "lo", "hi",
            }:
                raise ValueError(f"box axis {index} has a non-current schema")
            if not isinstance(axis["name"], str) or not axis["name"]:
                raise ValueError(f"box axis {index} name must be non-empty")
            if not isinstance(axis["periodic"], bool):
                raise ValueError(f"box axis {index} periodic must be bool")
            for key in ("period", "lo", "hi"):
                value = axis[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"box axis {index} {key} must be finite")
            if float(axis["period"]) <= 0 or float(axis["hi"]) <= float(axis["lo"]):
                raise ValueError(f"box axis {index} has invalid bounds/period")
    elif kind == "sphere":
        if set(spec) != {"type", "dim"} or isinstance(spec["dim"], bool) or not isinstance(spec["dim"], int) or spec["dim"] < 1:
            raise ValueError("sphere domain requires exactly type + positive integer dim")
    elif kind == "klein":
        if set(spec) != {"type"}:
            raise ValueError("Klein bottle domain requires exactly type")
    elif kind == "projective":
        if set(spec) != {"type", "dim"} or type(spec["dim"]) is not int or spec["dim"] != 2:
            raise ValueError("projective plane domain requires exactly type + dim=2")
    elif kind == "custom":
        if set(spec) != {"type", "embed_dim", "bounds"}:
            raise ValueError("custom domain requires exactly type, embed_dim, bounds")
        dim = spec["embed_dim"]
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise ValueError("custom domain embed_dim must be a positive integer")
        bounds = spec["bounds"]
        if bounds is not None and (
            not isinstance(bounds, list)
            or len(bounds) != dim
            or any(
                not isinstance(row, list)
                or len(row) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in row)
                or float(row[1]) <= float(row[0])
                for row in bounds
            )
        ):
            raise ValueError("custom domain bounds must be null or one finite [lo, hi] pair per dimension")
    else:
        raise ValueError(f"unknown manifold domain type {kind!r}")
    return spec


def normalize_domain_spec(spec: Any) -> dict[str, Any]:
    """Normalize ergonomic authoring input into the exact persisted union."""
    if not isinstance(spec, dict):
        raise ValueError("manifold domain must be an object")
    kind = spec.get("type")
    if kind == "box" and set(spec) == {"type", "axes"} and isinstance(spec["axes"], list):
        normalized = {
            "type": "box",
            "axes": [
                {
                    "name": axis.get("name", f"axis{index}"),
                    "periodic": axis.get("periodic", False),
                    "period": axis.get("period", 1.0),
                    "lo": axis.get("lo", 0.0),
                    "hi": axis.get("hi", 1.0),
                }
                for index, axis in enumerate(spec["axes"])
                if isinstance(axis, dict)
            ],
        }
    elif kind == "custom" and set(spec) <= {"type", "embed_dim", "bounds"}:
        normalized = {**spec, "bounds": spec.get("bounds")}
    else:
        normalized = dict(spec)
    return validate_domain_spec(normalized)


def domain_from_spec(spec: dict[str, Any]) -> ManifoldDomain:
    """Build a :class:`ManifoldDomain` from a ``manifold.json`` domain object."""
    spec = validate_domain_spec(spec)
    kind = spec["type"]
    if kind == "box":
        axes = [
            BoxAxis(
                name=a["name"],
                periodic=a["periodic"],
                period=float(a["period"]),
                lo=float(a["lo"]),
                hi=float(a["hi"]),
            )
            for a in spec["axes"]
        ]
        return BoxDomain(axes)
    if kind == "sphere":
        return SphereDomain(int(spec["dim"]))
    if kind == "klein":
        return KleinBottleDomain()
    if kind == "projective":
        return ProjectivePlaneDomain()
    if kind == "custom":
        return CustomDomain(
            spec["embed_dim"], bounds=spec["bounds"],
        )
    raise ValueError(f"unknown manifold domain type {kind!r}")
