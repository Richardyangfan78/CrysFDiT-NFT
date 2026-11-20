"""Geometry helpers: periodic minimum image, lattice parameterisation, graphs.

Conventions
-----------
* Lattice matrices are *row* matrices: ``L[i]`` is the i-th lattice vector,
  so Cartesian = fractional @ L.
* The generative lattice state is the 6-vector log-Cholesky parameterisation
  of the metric tensor ``G = L L^T``.  It is rotation-invariant, guarantees a
  positive-definite (valid) cell on decoding and fixes the rotational gauge
  to the lower-triangular lattice (a along x, b in the xy-plane).
"""
from __future__ import annotations

import contextlib
import functools
import itertools
import math

import numpy as np
import torch

Tensor = torch.Tensor


# ---------------------------------------------------------------- precision
_HALF_DTYPES = (torch.float16, torch.bfloat16)


def _active_autocast_devices() -> list[str]:
    """Device types with an active autocast region (``torch.is_autocast_enabled(device)`` needs torch 2.4)."""
    try:
        return [device for device in ("cuda", "cpu") if torch.is_autocast_enabled(device)]
    except TypeError:                                        # torch < 2.4: no device argument
        return ([*(["cuda"] if torch.is_autocast_enabled() else []),
                 *(["cpu"] if torch.is_autocast_cpu_enabled() else [])])


def _float32_geometry(function):
    """Keep a geometry routine in full precision inside a mixed-precision (autocast) region.

    Outside autocast -- every caller unless a training/sampling loop opts in -- this is a direct
    call, so results stay bitwise unchanged. Inside autocast it suspends autocast (otherwise the
    matmuls of the image search, the lattice inverse and the log-Cholesky factorisation would run
    in bf16/fp16) and upcasts half-precision tensor arguments to float32.
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        devices = _active_autocast_devices()
        if not devices:
            return function(*args, **kwargs)

        def upcast(value):
            return value.float() if isinstance(value, Tensor) and value.dtype in _HALF_DTYPES else value

        with contextlib.ExitStack() as stack:
            for device in devices:
                stack.enter_context(torch.autocast(device_type=device, enabled=False))
            return function(*(upcast(a) for a in args), **{k: upcast(v) for k, v in kwargs.items()})
    return wrapper


# ----------------------------------------------------------------- periodic
def wrap_frac(x: Tensor) -> Tensor:
    """Wrap fractional coordinates into [0, 1)."""
    return x - torch.floor(x)


def min_image_diff(a: Tensor, b: Tensor) -> Tensor:
    """Minimum-image fractional displacement ``a - b`` in (-0.5, 0.5]."""
    d = a - b
    return d - torch.round(d)


@_float32_geometry
def frac_to_cart(frac: Tensor, lattice: Tensor) -> Tensor:
    """(..., N, 3) fractional x (..., 3, 3) row lattice -> Cartesian."""
    return torch.matmul(frac, lattice)


@_float32_geometry
def cart_to_frac(cart: Tensor, lattice: Tensor) -> Tensor:
    return torch.matmul(cart, torch.linalg.inv(lattice))


def _reduced_basis_transform(cell: Tensor) -> Tensor:
    """Integer LLL basis change; its discrete choice has no coordinate gradient.

    Reducing the basis prevents a shear (e.g. b + 1000*a) from requiring a
    thousand periodic images. The final Cartesian vectors still use the
    original torch lattice, so coordinate and lattice gradients are retained.
    """
    basis = cell.detach().to(device="cpu", dtype=torch.float64).numpy()
    transform = _cached_lll_transform(np.ascontiguousarray(basis).tobytes())
    return torch.as_tensor(transform.copy(), dtype=cell.dtype, device=cell.device)


@functools.lru_cache(maxsize=1 << 17)
def _cached_lll_transform(key: bytes) -> np.ndarray:
    """``_lll_transform`` memoised on the exact float64 bytes of the basis (a pure function of them).

    One training step searches the same data cells several times (encoder pair bias, coordinate
    loss, local-distance loss) and every epoch revisits them. Failures raise and are not cached.
    """
    transform = _lll_transform(np.frombuffer(key, dtype=np.float64).reshape(3, 3))
    transform.setflags(write=False)
    return transform


def _lll_transform(basis: np.ndarray) -> np.ndarray:
    """Integer LLL (delta = 3/4) basis change of one float64 (3, 3) row basis; raises ValueError."""
    basis = basis.copy()
    transform = np.eye(3, dtype=np.int64)

    def gram_schmidt():
        ortho = basis.copy()
        mu = np.zeros((3, 3))
        for i in range(3):
            for j in range(i):
                denominator = np.dot(ortho[j], ortho[j])
                if denominator <= 0:
                    raise ValueError("singular periodic lattice")
                mu[i, j] = np.dot(basis[i], ortho[j]) / denominator
                ortho[i] -= mu[i, j] * ortho[j]
        return mu, (ortho * ortho).sum(axis=1)

    k = 1
    for _ in range(256):
        if k == 3:
            return transform
        mu, lengths = gram_schmidt()
        for j in range(k - 1, -1, -1):
            integer = int(np.rint(mu[k, j]))
            if integer:
                basis[k] -= integer * basis[j]
                transform[k] -= integer * transform[j]
                mu, lengths = gram_schmidt()
        if lengths[k] >= (0.75 - mu[k, k - 1] ** 2) * lengths[k - 1]:
            k += 1
        else:
            basis[[k, k - 1]] = basis[[k - 1, k]]
            transform[[k, k - 1]] = transform[[k - 1, k]]
            k = max(k - 1, 1)
    raise ValueError("periodic lattice basis reduction did not converge")


_IMAGE_SEARCH_CHUNK = 1 << 21          # (cell, entry, image) triples per vectorised search chunk


def _image_shifts(bounds: tuple[int, int, int], dtype: torch.dtype, device) -> Tensor:
    """Integer shifts with |s_j| <= bounds_j except 0, in ``itertools.product`` order: (S, 3).

    Built on the host (a boolean mask on a GPU tensor would synchronise) and copied once.
    """
    shifts = torch.cartesian_prod(*(torch.arange(-n, n + 1, dtype=torch.float64) for n in bounds))
    return shifts[(shifts != 0).any(-1)].to(dtype=dtype, device=device)


def _scan_first_minimum(distance2: Tensor, value: Tensor, index: Tensor, blocked: Tensor, start: int
                        ) -> tuple[Tensor, Tensor, Tensor]:
    """One chunk of the reference's sequential image scan, vectorised over the chunk's last axis.

    The reference loops ``take = d < best2; best2 = torch.minimum(best2, d)`` over the images in a
    fixed order: it keeps the FIRST image of minimal length (ties keep the earlier one, the starting
    vector first) and, because ``torch.minimum`` propagates NaN, a NaN distance blocks every later
    image. ``distance2`` (..., C) holds images ``start .. start + C - 1``; ``value``/``index`` (...)
    are the best length and image so far (index -1 = the starting vector) and ``blocked`` marks scans
    that met a NaN. ``min`` returns the first index of a tie.
    """
    nan = torch.isnan(distance2)
    excluded = (nan.cumsum(-1, dtype=torch.int32) > 0) | blocked.unsqueeze(-1)
    chunk_value, chunk_index = distance2.masked_fill(excluded, math.inf).min(-1)
    take = chunk_value < value
    return (torch.where(take, chunk_value, value), torch.where(take, chunk_index + start, index),
            blocked | nan.any(-1))


@torch.no_grad()
def _first_minimal_shift(delta: Tensor, reduced: Tensor, best2: Tensor, bounds: tuple[int, int, int]
                         ) -> tuple[Tensor, Tensor]:
    """The image the sequential reference loop picks, for cells that share ``bounds``.

    ``delta`` (G, P, 3) wrapped reduced-basis fractions, ``reduced`` (G, 3, 3), ``best2`` (G, P) the
    squared length of the starting vector. The shifts are visited in ``itertools.product`` order
    (``_scan_first_minimum``); each chunk is one batched matmul with the reference's arithmetic,
    ``(delta + s) @ reduced``. Returns (shift (G, P, 3), taken (G, P)).
    """
    shifts = _image_shifts(bounds, delta.dtype, delta.device)
    groups, entries = best2.shape
    index = torch.full_like(best2, -1, dtype=torch.long)
    if shifts.shape[0] == 0:
        return torch.zeros_like(delta), index >= 0
    value, blocked = best2, torch.zeros_like(best2, dtype=torch.bool)
    step = max(1, _IMAGE_SEARCH_CHUNK // max(1, groups * entries))
    for start in range(0, shifts.shape[0], step):
        part = shifts[start:start + step]
        candidate = torch.bmm((delta.unsqueeze(-2) + part).reshape(groups, -1, 3), reduced)
        distance2 = candidate.square().sum(-1).view(groups, entries, part.shape[0])
        value, index, blocked = _scan_first_minimum(distance2, value, index, blocked, start)
    taken = index >= 0
    shift = torch.where(taken.unsqueeze(-1), shifts[index.clamp(min=0)], torch.zeros_like(delta))
    return shift, taken


@_float32_geometry
def _periodic_vectors(diff: Tensor, cell: Tensor, self_image: bool = False) -> Tensor:
    """Exact image search with reciprocal-lattice bounds, in a reduced basis.

    ``cell`` (3, 3) with ``diff`` (..., 3) searches one cell. ``cell`` (B, 3, 3) with ``diff``
    (B, ..., 3) searches B cells at once and equals ``torch.stack`` of the one-cell results up to
    floating-point rounding (see ``_batched_periodic_vectors``).
    """
    if cell.dim() == 3:
        return _batched_periodic_vectors(diff, cell, self_image)
    if not bool(torch.isfinite(cell).all()) or not bool(torch.isfinite(diff).all()):
        return torch.full_like(diff, float("nan"))
    if float(torch.linalg.det(cell.detach()).abs()) == 0.0:
        return torch.full_like(diff, float("nan"))
    try:
        change = _reduced_basis_transform(cell)
    except (ValueError, OverflowError, np.linalg.LinAlgError):
        return torch.full_like(diff, float("nan"))
    reduced = change @ cell
    try:
        inverse_change = torch.linalg.inv(change)
        inverse = torch.linalg.inv(reduced.detach())
    except RuntimeError:  # singular/numerically singular inverse on supported torch versions
        return torch.full_like(diff, float("nan"))
    delta = diff @ inverse_change
    delta = delta - torch.round(delta)
    gram = reduced.detach() @ reduced.detach().T
    diagonal = torch.diag(torch.diagonal(gram))
    orthogonal = bool((gram == diagonal).all())
    if not bool(torch.isfinite(inverse).all()):
        return torch.full_like(diff, float("nan"))
    if self_image:
        index = int(reduced.detach().norm(dim=-1).argmin())
        best = reduced[index].expand_as(delta)
    else:
        best = delta @ reduced
    best2 = best.square().sum(-1)
    if orthogonal:
        return best
    # If |v| <= R then each fractional component |(v @ L^-1)_j|
    # <= R ||(L^-1)[:,j]||. This bounds ALL possible improving images,
    # rather than assuming an orthogonal cell or a fixed 27-image search.
    radius = float(best2.detach().max().sqrt())
    offset = 0.0 if self_image else 0.5
    continuous_bounds = radius * inverse.norm(dim=0) + offset + 1e-5
    if not bool(torch.isfinite(continuous_bounds).all()) or bool((continuous_bounds > 4096).any()):
        return torch.full_like(diff, float("nan"))
    bounds = torch.floor(continuous_bounds).long().tolist()
    if math.prod(2 * n + 1 for n in bounds) > 4096:
        # Generated pathological cells must fail closed rather than hanging a
        # training step. A NaN distance is explicitly invalidated by verifiers.
        return torch.full_like(diff, float("nan"))
    # The search is vectorised over images; the gradient flows only through the chosen image.
    shift, taken = _first_minimal_shift(delta.detach().reshape(1, -1, 3), reduced.detach().unsqueeze(0),
                                        best2.detach().reshape(1, -1), tuple(bounds))
    candidate = (delta + shift.view(delta.shape)) @ reduced
    return torch.where(taken.view(best2.shape).unsqueeze(-1), candidate, best)


def _batched_periodic_vectors(diff: Tensor, cells: Tensor, self_image: bool = False) -> Tensor:
    """``torch.stack([_periodic_vectors(diff[i], cells[i]) ...])`` without a per-cell Python loop.

    The same operations as the one-cell path, batched over cells: per-cell validity, the integer LLL
    basis change (one host transfer, memoised NumPy reduction), the reciprocal-lattice bounds with the
    same 4096-image budget, and the first-minimal-image search grouped by bounds. Cells that the
    one-cell path rejects (non-finite, singular, unreducible, over budget) are NaN; their inputs are
    replaced by an identity cell and zero displacement before any arithmetic, so no NaN can reach
    the gradients of the valid cells.

    Values and coordinate gradients equal that stack bitwise (on CPU, as tested) when each cell holds
    two or more vectors, as in every pairwise call with two or more atoms. With a single vector per
    cell (``diff`` (B, 3) or (B, 1, 3)) the one-cell ``(1, 3) @ (3, 3)`` products take another BLAS
    kernel than the batched ``bmm``, so the two agree only up to rounding (an exact tie between images
    could then resolve differently). Lattice gradients accumulate in another order than the one-cell
    loop over images and agree up to rounding (float32 ~1e-5 relative).
    """
    count = cells.shape[0]
    shape = diff.shape
    if count == 0:
        return diff.new_empty(shape)
    eye = torch.eye(3, dtype=cells.dtype, device=cells.device).expand(count, 3, 3)
    detached = cells.detach()
    flat = diff.reshape(count, -1, 3)
    valid = torch.isfinite(detached).flatten(1).all(-1) & torch.isfinite(flat.detach()).flatten(1).all(-1)
    valid = valid & (torch.linalg.det(torch.where(valid.view(-1, 1, 1), detached, eye)).abs() != 0)
    basis = torch.where(valid.view(-1, 1, 1), detached, eye).to(device="cpu", dtype=torch.float64).numpy()
    flags = valid.tolist()
    transforms = np.tile(np.eye(3, dtype=np.int64), (count, 1, 1))
    for i, ok in enumerate(flags):
        if ok:
            try:
                transforms[i] = _cached_lll_transform(np.ascontiguousarray(basis[i]).tobytes())
            except (ValueError, OverflowError, np.linalg.LinAlgError):
                flags[i] = False
    valid = torch.tensor(flags, dtype=torch.bool, device=cells.device)
    change = torch.as_tensor(transforms, dtype=cells.dtype, device=cells.device)
    reduced = change @ torch.where(valid.view(-1, 1, 1), cells, eye)
    inverse_change, info_change = torch.linalg.inv_ex(change)
    inverse, info = torch.linalg.inv_ex(reduced.detach())
    valid = valid & (info_change == 0) & (info == 0) & torch.isfinite(inverse).flatten(1).all(-1)
    delta = torch.bmm(torch.where(valid.view(-1, 1, 1), flat, torch.zeros_like(flat)), inverse_change)
    delta = delta - torch.round(delta)
    detached_reduced = reduced.detach()
    gram = detached_reduced @ detached_reduced.mT
    orthogonal = (gram == torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))).flatten(1).all(-1)
    if self_image:
        index = detached_reduced.norm(dim=-1).argmin(-1)
        best = reduced[torch.arange(count, device=cells.device), index].unsqueeze(1).expand_as(delta)
    else:
        best = torch.bmm(delta, reduced)
    best2 = best.square().sum(-1)
    radius = best2.detach().amax(-1).sqrt()
    offset = 0.0 if self_image else 0.5
    continuous_bounds = radius.unsqueeze(-1) * inverse.norm(dim=-2) + offset + 1e-5
    within = torch.isfinite(continuous_bounds).all(-1) & ~(continuous_bounds > 4096).any(-1)
    bounds = torch.floor(torch.where(within.unsqueeze(-1), continuous_bounds, 0.0)).long()
    within = within & ((2 * bounds + 1).prod(-1) <= 4096)
    # An exactly orthogonal reduced cell keeps its rounded vector before any budget check.
    failed = ~valid | (~orthogonal & ~within)
    search = (valid & ~orthogonal & within).nonzero(as_tuple=False).flatten()
    shift = torch.zeros_like(delta)
    taken = torch.zeros_like(best2, dtype=torch.bool)
    if search.numel():
        groups: dict[tuple[int, ...], list[int]] = {}
        for row, key in zip(search.tolist(), bounds[search].tolist()):
            groups.setdefault(tuple(key), []).append(row)
        for key, rows in groups.items():
            members = torch.tensor(rows, dtype=torch.long, device=cells.device)
            part_shift, part_taken = _first_minimal_shift(delta.detach()[members], detached_reduced[members],
                                                          best2.detach()[members], key)
            shift = shift.index_copy(0, members, part_shift)
            taken = taken.index_copy(0, members, part_taken)
    candidate = torch.bmm(delta + shift, reduced)
    result = torch.where(taken.unsqueeze(-1), candidate, best)
    result = torch.where(failed.view(-1, 1, 1), torch.full_like(result, math.nan), result)
    return result.reshape(shape)


@_float32_geometry
def _certified_batch_images(diff: Tensor, lattice: Tensor, self_image: bool = False) -> tuple[Tensor, Tensor]:
    """Vectorized image search with a per-cell proof that 27 images suffice.

    All tensor algebra stays on the input device. Only the caller's fallback
    cells need a CPU basis reduction. ``diff`` has shape (B, ..., 3).
    """
    shape = diff.shape
    delta = diff.reshape(diff.shape[0], -1, 3)
    delta = delta - torch.round(delta)
    detached = lattice.detach()
    inverse, info = torch.linalg.inv_ex(detached)
    finite = torch.isfinite(detached).flatten(1).all(-1) & (info == 0)
    finite &= torch.isfinite(inverse).flatten(1).all(-1) & torch.isfinite(delta.detach()).flatten(1).all(-1)
    gram = detached @ detached.transpose(-1, -2)
    off_diagonal = gram - torch.diag_embed(torch.diagonal(gram, dim1=-2, dim2=-1))
    # Exact orthogonality is sufficient to prove independent rounding optimal.
    # Approximate orthogonality is handled by the general bounds instead.
    orthogonal = (off_diagonal == 0).flatten(1).all(-1) & finite
    if self_image:
        shortest_row = detached.square().sum(-1).argmin(-1)
        best = lattice[torch.arange(lattice.shape[0], device=lattice.device), shortest_row].unsqueeze(1)
    else:
        best = torch.bmm(delta, lattice)
    certified = orthogonal.clone()
    search = (finite & ~orthogonal).nonzero(as_tuple=False).flatten()
    if search.numel():
        local_delta, local_lattice = delta[search], lattice[search]
        local_best = best[search]
        local_d2 = local_best.square().sum(-1)
        # Transform each pair once and all 26 translations in one small batched operation.
        base_cart = torch.bmm(local_delta, local_lattice)
        shifts = lattice.new_tensor([shift for shift in itertools.product((-1, 0, 1), repeat=3) if shift != (0, 0, 0)])
        translations = torch.einsum("sk,bkl->bsl", shifts, local_lattice)
        # The 26 translations in chunks of at most _IMAGE_SEARCH_CHUNK (cell, pair, image) triples,
        # scanned in order (``_scan_first_minimum``): the vector and the NaN-propagating squared length
        # are bitwise those of the former loop over single translations, and so is the coordinate
        # gradient. The lattice gradient gathers through the chosen translation (an indexed
        # accumulation instead of the loop's torch.where chain), so it matches the loop only up to
        # float rounding (float32: ~1e-5 relative). The training losses pass data lattices, which need
        # no gradient; only a differentiable reward through a decoded lattice would see the difference.
        cells, entries = local_d2.shape
        value, index = local_d2.detach(), torch.full_like(local_d2, -1, dtype=torch.long)
        blocked = torch.zeros_like(local_d2, dtype=torch.bool)
        step = max(1, _IMAGE_SEARCH_CHUNK // max(1, cells * entries))
        with torch.no_grad():
            for start in range(0, shifts.shape[0], step):
                candidate = base_cart.unsqueeze(2) + translations[:, start:start + step].unsqueeze(1)
                value, index, blocked = _scan_first_minimum(candidate.square().sum(-1), value, index, blocked, start)
        taken = index >= 0
        rows = torch.arange(cells, device=lattice.device).unsqueeze(-1)
        chosen = base_cart + translations[rows, index.clamp(min=0)]
        local_best = torch.where(taken.unsqueeze(-1), chosen, local_best)
        local_d2 = torch.where(blocked, torch.full_like(value, math.nan), value)
        best = best.index_copy(0, search, local_best)
        radius = local_d2.detach().amax(dim=-1).sqrt()
        offset = 0.0 if self_image else 0.5
        # For |v| <= R, |(v L^-1)_j| <= R ||L^-1[:,j]||.
        # Wrapped delta_j is in [-.5,.5], so every improving integer shift
        # lies within floor(R ||L^-1[:,j]|| + .5). For self-images delta=0.
        bounds = torch.floor(radius.unsqueeze(-1) * inverse[search].norm(dim=-2) + offset + 1e-5)
        certified[search] = (bounds <= 1).all(-1) & torch.isfinite(bounds).all(-1)
    return best.reshape(shape), certified


@_float32_geometry
def pairwise_displacements(
    coords: Tensor,
    lattice: Tensor,
    pbc: Tensor,
    max_image: int = 1,
) -> Tensor:
    """Cartesian minimum-image vectors, with certified fast and skew-cell paths.

    Normal cells use a batched 27-image search whose completeness is checked
    from the reciprocal lattice. Only uncertified cells require basis reduction.
    ``max_image`` remains for compatibility; it no longer limits correctness.
    Diagonal entries are zero; self-images are checked separately.
    """
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    periodic = pbc.nonzero(as_tuple=False).flatten()
    if periodic.numel() == 0:
        return diff
    local_diff, local_lattice = diff[periodic], lattice[periodic]
    best, certified = _certified_batch_images(local_diff, local_lattice)
    fallback = (~certified).nonzero(as_tuple=False).flatten()
    if fallback.numel():
        exact = _periodic_vectors(local_diff[fallback], local_lattice[fallback])     # all fallback cells at once
        best = best.index_copy(0, fallback, exact)
    return diff.index_copy(0, periodic, best)


@_float32_geometry
def shortest_periodic_translation(lattice: Tensor, pbc: Tensor) -> Tensor:
    """Shortest nonzero lattice vector; certified in batches when possible."""
    distances = lattice.new_full((lattice.shape[0],), float("inf"))
    periodic = pbc.nonzero(as_tuple=False).flatten()
    if periodic.numel() == 0:
        return distances
    cells = lattice[periodic]
    zeros = cells.new_zeros(cells.shape[0], 1, 3)
    best, certified = _certified_batch_images(zeros, cells, self_image=True)
    fallback = (~certified).nonzero(as_tuple=False).flatten()
    if fallback.numel():
        exact = _periodic_vectors(zeros[fallback], cells[fallback], self_image=True)
        best = best.index_copy(0, fallback, exact)
    return distances.index_copy(0, periodic, best.norm(dim=-1).squeeze(-1))


def remove_center_of_mass(coords: Tensor, atom_mask: Tensor) -> Tensor:
    """Subtract the masked mean of ``coords`` along the atom axis."""
    m = atom_mask.unsqueeze(-1).to(coords.dtype)
    n = m.sum(dim=1, keepdim=True).clamp(min=1.0)
    com = (coords * m).sum(dim=1, keepdim=True) / n
    return (coords - com) * m


# ------------------------------------------------------------------ lattice
@_float32_geometry
def lattice_to_log_cholesky(lattice: Tensor, eps: float = 1e-6) -> Tensor:
    """(B, 3, 3) row lattice -> (B, 6) log-Cholesky vector of G = L L^T."""
    metric = torch.matmul(lattice, lattice.transpose(-1, -2))
    metric = metric + eps * torch.eye(3, device=lattice.device, dtype=lattice.dtype)
    chol = torch.linalg.cholesky(metric)                      # lower triangular
    return torch.stack(
        [
            torch.log(chol[..., 0, 0]),
            chol[..., 1, 0],
            torch.log(chol[..., 1, 1]),
            chol[..., 2, 0],
            chol[..., 2, 1],
            torch.log(chol[..., 2, 2]),
        ],
        dim=-1,
    )


@_float32_geometry
def log_cholesky_to_lattice(params: Tensor) -> Tensor:
    """(B, 6) log-Cholesky vector -> (B, 3, 3) lower-triangular row lattice."""
    zeros = torch.zeros_like(params[..., 0])
    row0 = torch.stack([torch.exp(params[..., 0]), zeros, zeros], dim=-1)
    row1 = torch.stack([params[..., 1], torch.exp(params[..., 2]), zeros], dim=-1)
    row2 = torch.stack([params[..., 3], params[..., 4], torch.exp(params[..., 5])], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


@_float32_geometry
def canonicalize_lattice(lattice: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate a lattice into the lower-triangular gauge.

    Returns ``(L_canonical, R)`` with ``L_canonical = L @ R`` and ``R`` the
    orthogonal matrix that must also be applied to Cartesian vectors.
    """
    canonical = log_cholesky_to_lattice(lattice_to_log_cholesky(lattice))
    rotation = torch.linalg.solve(lattice, canonical)          # L R = C
    return canonical, rotation


@_float32_geometry
def lattice_parameters(lattice: Tensor) -> Tensor:
    """(B, 3, 3) -> (B, 6) [a, b, c, alpha, beta, gamma] with angles in degrees."""
    a, b, c = lattice[..., 0, :], lattice[..., 1, :], lattice[..., 2, :]
    la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)

    def angle(u: Tensor, v: Tensor, lu: Tensor, lv: Tensor) -> Tensor:
        cos = (u * v).sum(-1) / (lu * lv).clamp(min=1e-9)
        return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))

    return torch.stack([la, lb, lc, angle(b, c, lb, lc), angle(a, c, la, lc), angle(a, b, la, lb)], dim=-1)


@_float32_geometry
def lattice_from_parameters(params: Tensor) -> Tensor:
    """(B, 6) [a, b, c, alpha, beta, gamma] (degrees) -> lower-triangular row lattice."""
    a, b, c = params[..., 0], params[..., 1], params[..., 2]
    alpha, beta, gamma = (torch.deg2rad(params[..., i]) for i in (3, 4, 5))
    cos_a, cos_b, cos_g, sin_g = torch.cos(alpha), torch.cos(beta), torch.cos(gamma), torch.sin(gamma)
    zeros = torch.zeros_like(a)
    v1 = torch.stack([a, zeros, zeros], dim=-1)
    v2 = torch.stack([b * cos_g, b * sin_g, zeros], dim=-1)
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g.clamp(min=1e-9)
    cz = torch.sqrt((c**2 - cx**2 - cy**2).clamp(min=1e-9))
    v3 = torch.stack([cx, cy, cz], dim=-1)
    return torch.stack([v1, v2, v3], dim=-2)


@_float32_geometry
def cell_volume(lattice: Tensor) -> Tensor:
    return torch.linalg.det(lattice).abs()


def reference_log_cholesky(num_atoms: Tensor, volume_per_atom: float = 15.0) -> Tensor:
    """Cubic reference cell whose volume scales with the atom count (MatterGen-style N^{1/3})."""
    edge = torch.pow(num_atoms.to(torch.float32).clamp(min=1.0) * volume_per_atom, 1.0 / 3.0)
    log_edge = torch.log(edge)
    zeros = torch.zeros_like(log_edge)
    return torch.stack([log_edge, zeros, log_edge, zeros, zeros, log_edge], dim=-1)


# -------------------------------------------------------------------- graph
@_float32_geometry
def periodic_radius_graph(
    coords: Tensor,
    lattice: Tensor,
    pbc: Tensor,
    atom_mask: Tensor,
    cutoff: float,
    max_neighbors: int = 32,
    max_image: int = 1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Dense k-nearest neighbour graph with image search for small cells.

    Returns ``(neighbor_index, neighbor_vec, neighbor_mask)`` with shapes
    (B, N, K), (B, N, K, 3), (B, N, K).  Non-periodic samples ignore images.
    """
    bsz, n, _ = coords.shape
    device = coords.device
    shifts = torch.arange(-max_image, max_image + 1, device=device, dtype=coords.dtype)
    grid = torch.stack(torch.meshgrid(shifts, shifts, shifts, indexing="ij"), dim=-1).reshape(-1, 3)  # (S, 3)
    num_shifts = grid.shape[0]
    zero_shift = int(((grid == 0).all(dim=-1)).nonzero()[0])

    diff = coords.unsqueeze(2) - coords.unsqueeze(1)                       # (B, N, N, 3) frac or cart
    is_pbc = pbc.view(bsz, 1, 1, 1, 1)
    frac_all = diff.unsqueeze(3) + grid.view(1, 1, 1, num_shifts, 3)      # (B, N, N, S, 3)
    cart_all = torch.einsum("bijsk,bkl->bijsl", frac_all, lattice)
    cart_nonpbc = diff.unsqueeze(3).expand(-1, -1, -1, num_shifts, -1)
    vec = torch.where(is_pbc, cart_all, cart_nonpbc)                      # (B, N, N, S, 3)
    dist = vec.norm(dim=-1)                                                # (B, N, N, S)

    valid = atom_mask.unsqueeze(1).unsqueeze(-1) & atom_mask.unsqueeze(2).unsqueeze(-1)
    valid = valid.expand(-1, -1, -1, num_shifts).clone()
    self_idx = torch.arange(n, device=device)
    valid[:, self_idx, self_idx, zero_shift] = False                        # drop self, keep self-images
    nonpbc_mask = torch.ones(num_shifts, dtype=torch.bool, device=device)
    nonpbc_mask[:] = False
    nonpbc_mask[zero_shift] = True
    valid = valid & torch.where(pbc.view(bsz, 1, 1, 1), torch.ones_like(valid), nonpbc_mask.view(1, 1, 1, -1))
    valid = valid & (dist < cutoff)

    dist_masked = torch.where(valid, dist, torch.full_like(dist, float("inf")))
    flat_dist = dist_masked.reshape(bsz, n, n * num_shifts)
    k = min(max_neighbors, n * num_shifts)
    top_dist, top_idx = torch.topk(flat_dist, k=k, dim=-1, largest=False)
    neighbor_mask = torch.isfinite(top_dist)
    neighbor_index = torch.div(top_idx, num_shifts, rounding_mode="floor")
    flat_vec = vec.reshape(bsz, n, n * num_shifts, 3)
    neighbor_vec = torch.gather(flat_vec, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
    neighbor_vec = torch.where(neighbor_mask.unsqueeze(-1), neighbor_vec, torch.zeros_like(neighbor_vec))
    return neighbor_index, neighbor_vec, neighbor_mask


def random_rotation_matrices(batch_size: int, device: torch.device | None = None, dtype=torch.float32) -> Tensor:
    """Uniformly random rotations via QR of Gaussian matrices (det fixed to +1)."""
    gaussian = torch.randn(batch_size, 3, 3, device=device, dtype=dtype)
    q, r = torch.linalg.qr(gaussian)
    signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    q = q * signs.unsqueeze(-2)
    det = torch.linalg.det(q)
    q[det < 0, :, 0] = -q[det < 0, :, 0]
    return q


def sinusoidal_embedding(values: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Standard sinusoidal features of a (B,) or (B, N) tensor -> (..., dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=values.device, dtype=values.dtype) / max(half - 1, 1))
    phase = values.unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb
