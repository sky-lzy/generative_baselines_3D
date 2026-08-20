"""Shared geometry representation / normalization strategies for the ablation.

ONE class, `GeometryBuilder`, implements five published schemes so every geometry loader
gets them without repeating code. It plugs into the single existing seam that all loaders
already call -- `_scale_norm.apply_scale_norm(raymaps_raw, depth_metric, cfg, sentinel)` --
so enabling a scheme requires ZERO per-dataset edits.

Select with the hydra flag `dataset.scale_mode=<scheme>`, one of:

    vggt        VGGT, arXiv 2503.11651
    vggt_omega  VGGT-Omega, arXiv 2605.15195
    pi3         pi^3, arXiv 2507.13347
    da3         Depth Anything 3, arXiv 2511.10647
    genception  GenCeption, arXiv 2607.09024

Every scheme returns the same triple as the existing code path:

    (raymaps_norm  (T, 6, H, W),
     depth_channel (same shape as depth_metric, values in [-1, 1]),
     scale         float, the metric->normalized divisor)

============================ HARD CONSTRAINTS HONOURED ============================

C1. The depth channel goes through the frozen Wan VAE, so it MUST land in [-1, 1].
    None of the five papers regress a bounded depth -- four of them (VGGT, VGGT-Omega,
    pi^3, DA3) regress unbounded metric-ish depth with an L1/L2 loss and specify no range
    map at all. GenCeption is the ONLY one that specifies a bounded map, and it does so
    precisely (Sec. 3.5). So the pipeline is split in two stages:

      Stage A (paper-faithful): reproduce the paper's gauge, scale scalar and ray
               representation exactly. No clamping, no invented constants.
      Stage B (range adapter):  map the resulting dimensionless depth into [-1, 1].
               For GenCeption this IS the paper's own map. For the other four the paper
               is silent, so ONE shared adapter is used for all of them -- identical
               across arms -- so the ablation isolates the paper's gauge/scale/
               representation rather than the choice of squash. See `_to_unit_range`.

C2. Raymaps and depth should stay scale-aligned so that, after denormalization, predicted
    depth and predicted translation live in the same metric frame. Four of the five schemes
    give this for free: they define ONE scalar that divides depth and translation together
    (VGGT/VGGT-Omega/DA3/pi^3 all state this explicitly). GenCeption states no shared scale
    -- see `_genception` for exactly how that gap is handled and how to toggle it.

Ray channels bypass the VAE (see wan_t2v_ray_depth_mot_concat_5b.py:541), so only the depth
channel is range-constrained. Ray channels are therefore never clamped, which keeps the
(depth, pose) mapping exactly invertible.

================================ CHANNEL LAYOUT ==================================

The model consumes a fixed (T, 6, H, W) raymap as two 3-channel blocks: `ray_d` = ch 0:3
and `ray_m` = ch 3:6. That slot layout is fixed by the architecture, so each scheme fills
the two slots with ITS OWN semantics (documented per scheme). Where a paper uses a
different channel ORDER (DA3 stores origins first, directions second) the slots are
remapped, not the semantics -- a permutation, not a change of representation.

============================== DERIVED QUANTITIES ================================

Everything below is taken verbatim from the papers or their reference implementations, with
exactly three exceptions, each flagged in-line with a `DERIVED:` comment and each exposed as
a config knob rather than hard-coded:

  1. pi^3 never states a standalone ground-truth normalizer -- its scale exists only inside
     the ROE alignment of Eq. (4). `_pi3_scale` transposes Eq. (4)'s own 1/z weighting to
     the data side. See `_pi3_scale`.
  2. GenCeption's alpha is described but never given a numeric value (Sec. 3.5).
  3. GenCeption's Rothko centre-rectangle size is shown in Figure 5 but never stated.

Assumption that is NOT derivable here and must be confirmed per dataset: `depth_metric` is
taken to be Z-DEPTH (camera-frame z), not along-ray distance. That is the convention of
TartanAir / VKITTI2 / SceneNet. Override with `dataset.geom_depth_is_distance=true`.
It matters because VGGT/DA3 define their scale over 3-D POINTS, which requires a correct
unprojection; the legacy disparity path never unprojected and so never cared.
"""
from typing import Optional, Tuple

import math
import torch

_EPS = 1e-8

# ---- scheme names (values of dataset.scale_mode) ----
VGGT_MODE = "vggt"
VGGT_OMEGA_MODE = "vggt_omega"
PI3_MODE = "pi3"
DA3_MODE = "da3"
GENCEPTION_MODE = "genception"

GEOMETRY_BUILDER_MODES = (
    VGGT_MODE, VGGT_OMEGA_MODE, PI3_MODE, DA3_MODE, GENCEPTION_MODE,
)

# ---- shared Stage-B range adapter defaults ----
# "log" = v = clip(alpha * ln(1 + u), 0, 1), u = depth / scale. Chosen as the shared adapter
# because (a) it is the exact functional form GenCeption specifies, so the GenCeption arm
# needs no separate machinery, and (b) the frozen Wan2.2 VAE round-trips it at ~3% abs_rel
# vs ~45% for signed disparity (see _scale_norm.PARALLAX_DEPTH_MAP_DEFAULT).
RANGE_MAP_DEFAULT = "log"
RANGE_ALPHA_DEFAULT = 0.30
# Orientation of the depth channel. "near_positive" (near -> +1, far -> -1) is this repo's
# existing convention and what every visualiser / inference path assumes, so it is the
# default for the four schemes whose papers specify no orientation. GenCeption DOES specify
# one (d' in [0,1] with near = 0), so its arm defaults to "near_negative" for faithfulness;
# set `dataset.geom_depth_orientation=near_positive` to force the repo convention on every
# arm for a strictly like-for-like comparison. The two differ by a sign, which is trivially
# invertible and carries no information, but it is a real deviation so it is not silent.
ORIENTATION_NEAR_POSITIVE = "near_positive"
ORIENTATION_NEAR_NEGATIVE = "near_negative"

SCALE_FLOOR = 1e-6
SCALE_CEIL = 1e6


# ======================================================================================
# config accessors
# ======================================================================================

def _get(cfg, key, default):
    try:
        v = cfg.get(key, default)
    except Exception:
        return default
    return default if v is None else v


def _get_float(cfg, key, default):
    try:
        return float(_get(cfg, key, default))
    except Exception:
        return default


def _get_bool(cfg, key, default):
    try:
        return bool(_get(cfg, key, default))
    except Exception:
        return default


def _get_str(cfg, key, default):
    try:
        return str(_get(cfg, key, default))
    except Exception:
        return default


def geometry_scheme(cfg) -> Optional[str]:
    """The selected scheme name, or None if `dataset.scale_mode` isn't one of the five."""
    mode = _get_str(cfg, "scale_mode", "")
    return mode if mode in GEOMETRY_BUILDER_MODES else None


def is_geometry_builder_mode(cfg) -> bool:
    return geometry_scheme(cfg) is not None


# ======================================================================================
# camera recovery from a Plucker raymap
# ======================================================================================
#
# VGGT / VGGT-Omega / DA3 define their scale over 3-D POINTS, and DA3 additionally needs the
# UNNORMALIZED ray direction, so the per-frame (K, R, t) must be known. The loaders build the
# raymap from (K, E) and then discard them before the normalization seam, so they are
# recovered from the raymap itself. This is exact for loader-built raymaps and -- for the
# rotation/intrinsics half -- is literally DA3's own published procedure (their Eq. 2).
#
# Callers that already hold (K, E) may pass them via `GeometryBuilder.__call__(cameras=...)`
# to skip recovery entirely; the recovery path exists so no loader has to be touched.


def origins_from_plucker(raymaps: torch.Tensor, max_pixels: int = 4096) -> torch.Tensor:
    """Camera origin o per frame from a Plucker raymap [d | m = o x d], (T,6,H,W) -> (T,3).

    d x m = d x (o x d) = (I - d d^T) o for unit d, so summing over pixels gives the 3x3
    system [sum(I - d d^T)] o = sum(d x m). Exact when all rays share one origin (always
    true for loader-built raymaps); A is well conditioned for any nonzero FOV.

    Duplicated from `_scale_norm.origins_from_plucker` on purpose: this module must not
    import from the legacy scheme-C module, so that the ablation branch can be reasoned
    about (and reverted) as one self-contained file.
    """
    T, C, H, W = raymaps.shape
    d = raymaps[:, 0:3].reshape(T, 3, -1).permute(0, 2, 1)   # (T, N, 3)
    m = raymaps[:, 3:6].reshape(T, 3, -1).permute(0, 2, 1)
    n = d.shape[1]
    if n > max_pixels:
        idx = torch.linspace(0, n - 1, max_pixels, device=d.device).long()
        d, m = d[:, idx], m[:, idx]
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    A = (d.shape[1] * torch.eye(3, dtype=d.dtype, device=d.device).expand(T, 3, 3)
         - torch.einsum("tni,tnj->tij", d, d))
    b = torch.cross(d, m, dim=-1).sum(dim=1)                 # (T, 3)
    return torch.linalg.solve(A, b)


def _pixel_grid(H: int, W: int, dtype, device) -> torch.Tensor:
    """Homogeneous pixel centres (3, H*W), matching `camera_to_raymap`'s +0.5 convention."""
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    x = (x + 0.5).reshape(-1)
    y = (y + 0.5).reshape(-1)
    return torch.stack([x, y, torch.ones_like(x)], dim=0)    # (3, N)


def _upper_cholesky(M: torch.Tensor) -> torch.Tensor:
    """Factor SPD M = K K^T with K UPPER triangular (the 'anti-transpose' Cholesky).

    torch.linalg.cholesky gives M = L L^T with L LOWER. With the exchange matrix J
    (J X J reverses both axes), let L = chol(J M J) and K = J L J. Then:
      - K is upper triangular, because reversing both axes of a lower-triangular matrix
        maps its nonzero entries into the upper triangle;
      - K K^T = J L J . J L^T J = J L L^T J = J (J M J) J = M,   since J J = I.
    Note it must be J L J and NOT J L^T J: the latter yields J L^T L J, which is a
    different matrix and silently produces wrong intrinsics.

    Used to split H = R K^-1 without a full RQ decomposition.
    """
    J = torch.flip(torch.eye(3, dtype=M.dtype, device=M.device), dims=(0,))
    L = torch.linalg.cholesky(J @ M @ J)
    return J @ L @ J


def cameras_from_raymap(
    raymaps: torch.Tensor, max_pixels: int = 4096
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover (K, R_c2w, t) per frame from a Plucker raymap. (T,6,H,W) -> (T,3,3),(T,3,3),(T,3).

    The loaders build directions as  d_unit = normalize(R_c2w K^-1 p)  (see
    re10k.camera_to_raymap). So H := R_c2w K^-1 maps homogeneous pixels to world directions
    up to a per-pixel positive scale, i.e.  H p x d_unit = 0  for every pixel. That is exactly
    DA3's Eq. (2), solved here by DLT:

        H* = argmin_{||H||=1} sum_p || H p x d_unit(p) ||

    The cross-product constraint is invariant to the (unknown, per-pixel) length of d, which
    is why UNIT directions suffice to recover the unnormalized ray field.

    H is then split without an RQ decomposition: H = R K^-1 implies H^T H = K^-T K^-1, so
    (H^T H)^-1 = K K^T, and an upper Cholesky yields K directly; R = H K. K is normalized to
    K[2,2] = 1, which is what makes K^-1 p have camera-frame z = 1 -- the property DA3's
    unnormalized direction depends on.

    Pixel coordinates are Hartley-normalized before the SVD and undone afterwards; without it
    the DLT design matrix is badly conditioned at typical image sizes.
    """
    T, _, H_img, W_img = raymaps.shape
    device = raymaps.device
    dt = torch.float64                                  # DLT + Cholesky in double

    d = raymaps[:, 0:3].to(dt).reshape(T, 3, -1)        # (T, 3, N)
    p = _pixel_grid(H_img, W_img, dt, device)           # (3, N)

    n = d.shape[-1]
    if n > max_pixels:
        idx = torch.linspace(0, n - 1, max_pixels, device=device).long()
        d, p = d[..., idx], p[..., idx]
    d = d / d.norm(dim=1, keepdim=True).clamp_min(_EPS)

    # Hartley normalization: p' = N p, centred and mean-norm sqrt(2).
    ctr = p[:2].mean(dim=1)
    span = (p[:2] - ctr[:, None]).norm(dim=0).mean().clamp_min(_EPS)
    s_h = math.sqrt(2.0) / float(span)
    Nmat = torch.tensor(
        [[s_h, 0.0, -s_h * float(ctr[0])],
         [0.0, s_h, -s_h * float(ctr[1])],
         [0.0, 0.0, 1.0]], dtype=dt, device=device)
    pn = Nmat @ p                                        # (3, N)

    # Rows of the DLT design matrix. For each pixel, [d]_x (H p) = 0 gives 3 equations
    # (rank 2). Stacking all three is standard and keeps the system symmetric.
    Ks, Rs = [], []
    eye = torch.eye(3, dtype=dt, device=device)
    for i in range(T):
        di = d[i]                                        # (3, N)
        # skew(d) @ kron-style expansion -> A (3N, 9), with H flattened row-major.
        zero = torch.zeros_like(di[0])
        A = torch.cat([
            torch.stack([zero, -di[2], di[1]], dim=-1),   # (N,3) coefficients on rows of H
            torch.stack([di[2], zero, -di[0]], dim=-1),
            torch.stack([-di[1], di[0], zero], dim=-1),
        ], dim=0)                                         # (3N, 3)
        P = pn.transpose(0, 1)                            # (N, 3)
        Pr = torch.cat([P, P, P], dim=0)                  # (3N, 3)
        # row k of A picks coefficients (a0,a1,a2) on the three ROWS of H; each row of H
        # contracts with p. So the 9-vector is outer(a, p).
        Adlt = (A[:, :, None] * Pr[:, None, :]).reshape(A.shape[0], 9)
        # Smallest right-singular vector of Adlt == smallest eigenvector of Adlt^T Adlt.
        # Going through the 9x9 normal matrix keeps this O(N) instead of materializing a
        # (3N x 3N) U from a full SVD, which at 4096 sampled pixels is a ~1 GB allocation.
        _, evec = torch.linalg.eigh(Adlt.transpose(0, 1) @ Adlt)
        Hi = evec[:, 0].reshape(3, 3)
        Hi = Hi @ Nmat                                    # undo Hartley normalization

        # Split H = R K^-1.
        M = torch.linalg.inv(Hi.transpose(0, 1) @ Hi)     # = K K^T
        M = 0.5 * (M + M.transpose(0, 1))                 # symmetrize against round-off
        Ki = _upper_cholesky(M)
        Ki = Ki / Ki[2, 2].clamp_min(_EPS)                # K[2,2] = 1
        # Make focal lengths positive (Cholesky sign convention is free per column).
        sgn = torch.sign(torch.diagonal(Ki))
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        Ki = Ki * sgn[None, :]
        # The DLT recovers H only up to an overall factor c (sign included), but both
        # unknowns cancel: M = inv(H^T H) = K K^T / c^2, so normalizing K[2,2] = 1 above
        # gives K EXACTLY, independent of c. Then R = H K = +-c R_true, and projecting onto
        # SO(3) leaves +-R_true -- of which only +R_true has det = +1. So requiring
        # det(R) > 0 pins the last degree of freedom exactly, and the rebuilt directions
        # then agree with the input ones by construction (assert via raymap_recovery_error).
        U, _, Vt = torch.linalg.svd(Hi @ Ki)
        Ri = U @ Vt
        if torch.det(Ri) < 0:
            Ri = -Ri
        Ks.append(Ki)
        Rs.append(Ri)

    K = torch.stack(Ks).to(raymaps.dtype)
    R = torch.stack(Rs).to(raymaps.dtype)
    t = origins_from_plucker(raymaps)
    return K, R, t


def raymap_recovery_error(raymaps: torch.Tensor, max_pixels: int = 1024) -> float:
    """Max angular error (degrees) between the input directions and those rebuilt from the
    recovered (K, R). A correctness probe for `cameras_from_raymap`; call it in a test or a
    one-off script rather than in the data path. 0 means the recovery is exact.

    Computed in float64 and via the chord (asin of the cross-product norm) rather than
    arccos: near cos = 1, arccos loses half the mantissa, so a float32 arccos probe bottoms
    out around 0.026 deg and would report a spurious error for an exact recovery.
    """
    K, R, _ = cameras_from_raymap(raymaps, max_pixels=max_pixels)
    T, _, H_img, W_img = raymaps.shape
    dt = torch.float64
    p = _pixel_grid(H_img, W_img, dt, raymaps.device)
    n = p.shape[-1]
    idx = torch.linspace(0, n - 1, min(n, max_pixels), device=raymaps.device).long()
    p = p[:, idx]
    d_ref = raymaps[:, 0:3].to(dt).reshape(T, 3, -1)[..., idx]
    d_ref = d_ref / d_ref.norm(dim=1, keepdim=True).clamp_min(_EPS)
    d_hat = R.to(dt) @ torch.linalg.solve(K.to(dt), p.expand(T, 3, -1))
    d_hat = d_hat / d_hat.norm(dim=1, keepdim=True).clamp_min(_EPS)
    sin = torch.cross(d_ref, d_hat, dim=1).norm(dim=1).clamp(0.0, 1.0)
    return float(torch.rad2deg(torch.asin(sin)).max().item())


# ======================================================================================
# GeometryBuilder
# ======================================================================================

class _Ctx:
    """Per-clip working set, so scheme methods don't take ten positional arguments.

    rays      (T,6,Hr,Wr) raw Plucker raymap, world frame
    depth     (T, Hd*Wd)  stored depth, metres, invalid pixels = sentinel
    z         (T, Hd*Wd)  camera-frame z (== depth unless geom_depth_is_distance)
    valid     (T, Hd*Wd)  bool
    K,R,t     per-frame intrinsics (RAYMAP pixel units), c2w rotation, camera centre
    d_depth   (T,3,Hd*Wd) unnormalized world dirs on the DEPTH grid
    d_ray     (T,3,Hr*Wr) unnormalized world dirs on the RAYMAP grid
    """

    __slots__ = ("rays", "depth", "z", "valid", "K", "R", "t",
                 "ray_hw", "depth_hw", "d_depth", "d_ray")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class GeometryBuilder:
    """Builds (raymaps_norm, depth_channel, scale) under one of five published schemes.

    Stateless apart from `cfg`; construct per clip or once per dataset, either is fine.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.scheme = geometry_scheme(cfg)
        if self.scheme is None:
            raise ValueError(
                f"scale_mode={_get_str(cfg, 'scale_mode', '<unset>')!r} is not a "
                f"GeometryBuilder scheme; expected one of {GEOMETRY_BUILDER_MODES}"
            )

    # ---------------------------------------------------------------- public entry point

    def __call__(
        self,
        raymaps_raw: torch.Tensor,
        depth_metric: torch.Tensor,
        invalid_sentinel: float,
        cameras: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """raymaps_raw (T,6,Hr,Wr) Plucker [unit dir | moment o x d], world frame.
        depth_metric (T,1,Hd,Wd) or (T,Hd,Wd), metres, invalid pixels stuffed with
        `invalid_sentinel` (use float('inf') if they are +inf).
        cameras: optional pre-known (K, R_c2w, t), K in RAYMAP pixel units, to skip recovery.

        The raymap and depth grids are generally DIFFERENT (the 1.3B recipe uses a 30x40
        raymap against 240x320 depth), so every quantity below carries the grid it lives on:
        scale statistics are computed on the DEPTH grid -- which uses all 76,800 depth pixels
        rather than 1,200, and is what "all 3D points" means in VGGT/DA3 -- while the ray
        channels are written on the RAYMAP grid."""
        depth_shape = depth_metric.shape
        ray_hw = (raymaps_raw.shape[-2], raymaps_raw.shape[-1])
        depth_hw = (depth_metric.shape[-2], depth_metric.shape[-1])
        depth = depth_metric.reshape(depth_metric.shape[0], -1)          # (T, Hd*Wd)
        valid = torch.isfinite(depth) & (depth > 0) & (depth < invalid_sentinel)

        K, R, t = cameras if cameras is not None else cameras_from_raymap(raymaps_raw)

        ctx = _Ctx(
            rays=raymaps_raw, depth=depth, valid=valid, K=K, R=R, t=t,
            ray_hw=ray_hw, depth_hw=depth_hw,
            d_depth=self._dirs(K, R, ray_hw, depth_hw),   # unnormalized dirs, depth grid
            d_ray=self._dirs(K, R, ray_hw, ray_hw),       # unnormalized dirs, raymap grid
        )
        ctx.z = self._to_z(ctx.depth, ctx.d_depth)

        fn = {
            VGGT_MODE: self._vggt,
            VGGT_OMEGA_MODE: self._vggt_omega,
            PI3_MODE: self._pi3,
            DA3_MODE: self._da3,
            GENCEPTION_MODE: self._genception,
        }[self.scheme]
        rays_norm, depth_u, scale = fn(ctx)

        depth_channel = self._to_unit_range(depth_u, valid).reshape(depth_shape)
        return rays_norm, depth_channel, float(scale)

    # ---------------------------------------------------------------- shared geometry

    @staticmethod
    def _K_at(K: torch.Tensor, from_hw, to_hw) -> torch.Tensor:
        """Rescale intrinsics between two pixel grids, matching the loaders' OWN convention
        (`_generate_raymaps_raw`: fx, cx *= W_to/W_from and fy, cy *= H_to/H_from).

        This is load-bearing. The loaders build the raymap on a COARSE grid (1.3B runs use
        30x40) while depth stays at frame resolution (240x320), so `cameras_from_raymap`
        returns K in RAYMAP pixel units. Unprojecting depth with that K would be wrong by the
        stride factor (8x here) and would silently corrupt every point-based scale.
        """
        if tuple(from_hw) == tuple(to_hw):
            return K
        sy = float(to_hw[0]) / float(from_hw[0])
        sx = float(to_hw[1]) / float(from_hw[1])
        out = K.clone()
        out[:, 0, 0] *= sx
        out[:, 0, 2] *= sx
        out[:, 1, 1] *= sy
        out[:, 1, 2] *= sy
        return out

    def _dirs(self, K_ray, R, ray_hw, hw) -> torch.Tensor:
        """Unnormalized world ray directions d = R K^-1 p on the grid `hw`. NOT unit length:
        camera-frame z is exactly 1 (K[2,2] = 1). This is DA3's ray direction (their Sec.
        3.1) and is also what turns z-depth into a 3-D point: X = o + z * d. (T, 3, H*W)."""
        K = self._K_at(K_ray, ray_hw, hw)
        p = _pixel_grid(hw[0], hw[1], K.dtype, K.device)
        return R @ torch.linalg.solve(K, p.expand(R.shape[0], 3, -1))

    def _to_z(self, depth: torch.Tensor, d_un: torch.Tensor) -> torch.Tensor:
        """Stored depth -> camera-frame z. Datasets here store z-depth by default (TartanAir:
        "z-buffer depth in metres"; VKITTI2 and SceneNet likewise). For a loader that stores
        along-ray Euclidean distance set `dataset.geom_depth_is_distance=true`: ||d_un|| is
        exactly the distance-per-unit-z factor, since d_un has camera-frame z = 1."""
        if _get_bool(self.cfg, "geom_depth_is_distance", False):
            return depth / d_un.norm(dim=1).clamp_min(_EPS)
        return depth

    def _recentre_to_first_camera(self, R, t):
        """Express rotations/translations in the FIRST camera's frame -- the gauge VGGT,
        VGGT-Omega and DA3 all adopt (VGGT normalization.py:80-91; VGGT-Omega App. A.1;
        DA3 api.py:330 `ex @ affine_inverse(ex[:, :1])`). Returns (R_rel, t_rel)."""
        R0T = R[0].transpose(0, 1)
        return R0T[None] @ R, (R0T[None] @ (t - t[0])[..., None]).squeeze(-1)

    @staticmethod
    def _clamp_scale(s: torch.Tensor) -> float:
        """VGGT's own guard: avg_scale.clamp(min=1e-6, max=1e6) (normalization.py:103)."""
        v = float(s)
        if not math.isfinite(v) or v <= 0.0:
            return 1.0
        return float(min(max(v, SCALE_FLOOR), SCALE_CEIL))

    # ---------------------------------------------------------------- Stage B: [-1,1]

    def _to_unit_range(self, depth_u: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Map dimensionless normalized depth `depth_u` (>= 0, ~1 = typical) into [-1, 1].

        GenCeption specifies this map itself; the other four papers specify none, so all
        four share this one adapter to keep the ablation about the scheme, not the squash.

        "log" (default): v = clip(alpha * ln(1 + u), 0, 1)   -- GenCeption Sec. 3.5's form.
        "disparity":     v = 1 - 1/(1 + u)                   -- the repo's legacy map.

        Invalid pixels carry the caller's large sentinel, so they saturate to the FAR end
        under either map with no special-casing -- and are then written to the far end
        explicitly so an `inf` sentinel can't produce a NaN.
        """
        which = _get_str(self.cfg, "geom_depth_range_map", RANGE_MAP_DEFAULT)
        u = torch.nan_to_num(depth_u, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        if which == "disparity":
            v = 1.0 - 1.0 / (1.0 + u)
        else:
            v = torch.clamp(self._range_alpha() * torch.log1p(u), 0.0, 1.0)
        v = torch.where(valid, v, torch.ones_like(v))        # invalid -> far end
        # v in [0,1] with 0 = near, 1 = far.
        if self._orientation() == ORIENTATION_NEAR_POSITIVE:
            return 1.0 - 2.0 * v                             # near -> +1, far -> -1
        return 2.0 * v - 1.0                                 # near -> -1, far -> +1

    def _range_alpha(self) -> float:
        """alpha for the log map. `dataset.geom_range_alpha` wins if set, so a sweep can pin
        one alpha across every arm. Otherwise GenCeption reads its own knob, since alpha is
        part of ITS method (Sec. 3.5) rather than of this shared adapter."""
        explicit = _get(self.cfg, "geom_range_alpha", None)
        if explicit is not None:
            return _get_float(self.cfg, "geom_range_alpha", RANGE_ALPHA_DEFAULT)
        if self.scheme == GENCEPTION_MODE:
            return _get_float(self.cfg, "genception_alpha", RANGE_ALPHA_DEFAULT)
        return RANGE_ALPHA_DEFAULT

    def _orientation(self) -> str:
        """GenCeption fixes near = 0 (dark); the other four fix nothing, so they take the
        repo convention that every visualiser and the inference path already assume."""
        default = (ORIENTATION_NEAR_NEGATIVE if self.scheme == GENCEPTION_MODE
                   else ORIENTATION_NEAR_POSITIVE)
        return _get_str(self.cfg, "geom_depth_orientation", default)

    # ==================================================================================
    # Scheme 1 -- VGGT (arXiv 2503.11651)
    # ==================================================================================

    def _vggt(self, c):
        """Source of truth: vggt/training/train_utils/normalization.py
        `normalize_camera_extrinsics_and_points_batch`, lines 80-111.

        Verbatim recipe:
          1. Transform every quantity into the FIRST camera's coordinate frame
             (`new_extrinsics = extrinsics_homog @ inv(extrinsics[:, 0])`, line 80-82).
          2. avg_scale = sum(||X|| * mask) / count            (lines 100-103)
             i.e. the MEAN Euclidean distance of all valid points to the origin, where the
             origin is camera 0. Clamped to [1e-6, 1e6].
          3. world_points /= avg_scale; extrinsic translations /= avg_scale;
             depths /= avg_scale                              (lines 106-111)

        One scalar divides depth AND translation, so constraint C2 holds exactly.

        Ray channels: VGGT has no raymap -- it predicts camera parameters plus point/depth
        maps -- so the paper constrains only that translation is divided by avg_scale. The
        Plucker moment m = o x d is linear in o, so m/avg_scale is precisely "translation
        divided by the scale" expressed in this repo's existing channel. Keeping the moment
        makes this arm a pure gauge+scale change relative to the current baseline.
        """
        # World points on the DEPTH grid, then rotated/translated into the camera-0 frame.
        X = c.t[:, :, None] + c.z[:, None, :] * c.d_depth                  # (T,3,Hd*Wd)
        R0T = c.R[0].transpose(0, 1)
        X_rel = R0T[None] @ (X - c.t[0][None, :, None])

        dist = X_rel.norm(dim=1)                                          # (T, Hd*Wd)
        # `+ 1e-3` in the denominator is VGGT's own guard (normalization.py:103).
        scale = self._clamp_scale((dist * c.valid).sum() / (c.valid.sum() + 1e-3))

        rays = c.rays.clone()
        rays[:, 3:6] = c.rays[:, 3:6] / scale
        return rays, c.depth / scale, scale

    # ==================================================================================
    # Scheme 2 -- VGGT-Omega (arXiv 2605.15195)
    # ==================================================================================

    def _vggt_omega(self, c):
        """Source of truth: arXiv 2605.15195 Appendix A.1, verbatim:

          "For each scene, following [181, 186], we normalize the ground truth to a unit
           space. Concretely, we first transform all quantities into the first camera's
           coordinate frame and compute the average distance of all 3D points to the
           origin; we then scale the depth maps and translation vectors by this value."

        [181] is VGGT, so the SCALAR IS IDENTICAL TO `_vggt` -- this is not an oversight,
        it is what the paper says. The two arms therefore isolate REPRESENTATION, not scale.

        The representational difference is stated in Sec. 3 (lines 353-363): VGGT-Omega
        keeps "a single dense head for depth prediction", does "not directly predict point
        maps and tracks", and explicitly declines ray maps -- it predicts "only depth maps
        and camera parameters". So this arm carries the camera as explicit parameters
        broadcast over the frame (origin t/scale, spatially constant) rather than as a
        Plucker moment, which is the field VGGT-Omega would actually regress.

        Channel 0:3 keeps the unit directions (they encode rotation + intrinsics and cost
        nothing); channel 3:6 carries the normalized camera origin.
        """
        _, _, scale = self._vggt(c)
        _, t_rel = self._recentre_to_first_camera(c.R, c.t)

        rays = c.rays.clone()
        rays[:, 3:6] = (t_rel / scale).view(-1, 3, 1, 1).expand_as(rays[:, 3:6])
        return rays, c.depth / scale, scale

    # ==================================================================================
    # Scheme 3 -- pi^3 (arXiv 2507.13347)
    # ==================================================================================

    def _pi3_scale(self, X_local, depth, valid) -> float:
        """DERIVED (1 of 3). pi^3 never publishes a standalone ground-truth normalizer: its
        scale exists only inside the loss, as the ROE-solved s* of Eq. (4),

            s* = argmin_s  sum_i sum_j  (1/z_ij) * || s * x_hat_ij - x_ij ||_1

        so the faithful data-side transposition is the normalizer induced by that SAME
        objective and its SAME 1/z weighting -- the 1/z-weighted mean of the point-map L1
        magnitude over the whole clip:

            s = sum_ij w_ij ||x_ij||_1 / sum_ij w_ij ,   w_ij = 1 / z_ij

        This inherits the two properties the paper does state: it is a SINGLE scalar shared
        across all N views (Sec. 3.2, "up to an unknown, yet consistent, scale factor across
        all N images"), and it is 1/z-weighted so near geometry dominates, exactly as in
        Eq. (4). Switch to the unweighted mean-L2 convention with
        `dataset.pi3_scale_weighting=none` if you want to isolate the weighting itself.
        """
        l1 = X_local.abs().sum(dim=1)                                     # (T, HW)
        if _get_str(self.cfg, "pi3_scale_weighting", "inv_z") == "none":
            num = (X_local.norm(dim=1) * valid).sum()
            den = valid.sum().clamp_min(1)
            return self._clamp_scale(num / den)
        w = torch.where(valid, 1.0 / depth.clamp_min(_EPS), torch.zeros_like(depth))
        return self._clamp_scale((w * l1).sum() / w.sum().clamp_min(_EPS))

    def _pi3(self, c):
        """Source of truth: arXiv 2507.13347 Secs. 3.2-3.3, and pi3/models/pi3.py:198-209.

        Two defining choices, both reproduced here:

        (a) SCALE-INVARIANT LOCAL GEOMETRY (Sec. 3.2). Geometry lives in each frame's OWN
            camera coordinate system, not a shared world frame. The reference implementation
            is explicit:  `local_points = torch.cat([xy * z, z], dim=-1)`  (pi3.py:198), i.e.
            X_local = z * K^-1 p, and only afterwards are they lifted to the world by the
            predicted poses (pi3.py:209). There is NO reference view -- the model is
            permutation-equivariant by construction.

        (b) AFFINE-INVARIANT CAMERA POSE (Sec. 3.3). Poses are defined only up to a
            similarity. Rotation is invariant to it; the translation magnitude is not, and
            is rectified by "this single, consistent scale factor" s* -- the SAME scalar that
            normalizes the point maps. So C2 (depth/translation scale alignment) holds by
            construction, and it is the one scheme here where the paper argues for it
            directly.

        Channel effect: because directions are expressed in each frame's own camera frame,
        channel 0:3 becomes normalize(K^-1 p) -- identical across frames when intrinsics are
        fixed. That is not a bug; it is the content of "local": all rotation information is
        moved out of the dense field and into the pose channels. Channel 3:6 carries the
        normalized camera origin in the first-frame gauge (a gauge is unavoidable for a
        dense field; pi^3's reference-freeness is a property of its LOSS, which is out of
        scope for a data-side normalization).
        """
        # Rotate the world-frame unnormalized directions back into each frame's OWN camera
        # frame -> K^-1 p, then X_local = z * K^-1 p exactly as pi3.py:198. The scale needs
        # the DEPTH grid; the ray channel needs the RAYMAP grid.
        Rt = c.R.transpose(1, 2)
        X_local = c.z[:, None, :] * (Rt @ c.d_depth)                      # (T,3,Hd*Wd)
        scale = self._pi3_scale(X_local, c.z, c.valid)

        d_cam_ray = Rt @ c.d_ray                                          # (T,3,Hr*Wr)
        Hr, Wr = c.ray_hw
        rays = c.rays.clone()
        rays[:, 0:3] = (d_cam_ray / d_cam_ray.norm(dim=1, keepdim=True).clamp_min(_EPS)
                        ).reshape(c.rays.shape[0], 3, Hr, Wr)
        _, t_rel = self._recentre_to_first_camera(c.R, c.t)
        rays[:, 3:6] = (t_rel / scale).view(-1, 3, 1, 1).expand_as(rays[:, 3:6])
        return rays, c.depth / scale, scale

    # ==================================================================================
    # Scheme 4 -- Depth Anything 3 (arXiv 2511.10647)
    # ==================================================================================

    def _da3(self, c):
        """Source of truth: arXiv 2511.10647 Sec. 3.1 ("Depth-ray representation") and the
        training-objectives paragraph of Sec. 3.3.

        Representation, verbatim from Sec. 3.1:
          - ray r = (t, d), origin t and direction d = R K^-1 p;
          - "The dense ray map M in R^{HxWx6} stores these parameters for all pixels. We do
             not normalize d, so its magnitude preserves the projection scale."
          - "P = t + D(u,v) . d"   (so D is camera-frame z, since K^-1 p has z = 1)
          - M(:,:,:3) = origins, M(:,:,3:) = directions.

        Scale, verbatim from Sec. 3.3:
          "Prior to loss computation, all ground-truth signals are normalized by a common
           scale factor. This scale is defined as the mean l2 norm of the valid reprojected
           point maps P."

        Note this evaluates to the same quantity as VGGT's avg_scale in the same gauge --
        the DA3 arm's novelty is the REPRESENTATION, not the scalar. The distinguishing
        feature is the unnormalized direction: |d| = |K^-1 p| varies across the frame as a
        function of focal length, so the direction channel carries the intrinsics, which our
        current unit-normalized directions throw away. Because d is dimensionless it is
        NOT divided by the scale -- only t, D and P are. That is what "preserves the
        projection scale" means, and getting it wrong would silently destroy the intrinsics.

        Channel slots: DA3 stores [origins | directions]; this repo's blocks are
        [ray_d | ray_m]. The slots are remapped so directions stay in 0:3 and origins go to
        3:6 -- a permutation of storage, not of representation.

        DA3's inference path additionally normalizes INPUT extrinsics by the median camera-
        centre distance (api.py:327-338). That is pose conditioning, not ground-truth
        normalization, so it is not the default here; set `dataset.da3_scale=median_campos`
        to use it instead (it is also the only one of the five that works with no depth at
        all, e.g. for the NVS-only subsets).
        """
        _, t_rel = self._recentre_to_first_camera(c.R, c.t)
        R0T = c.R[0].transpose(0, 1)
        # Reprojected point map P on the DEPTH grid, in the first-camera gauge.
        P = t_rel[:, :, None] + c.z[:, None, :] * (R0T[None] @ c.d_depth)

        if _get_str(self.cfg, "da3_scale", "mean_point_l2") == "median_campos":
            # api.py:334-337 -- median over frames of ||camera centre||, floored at 1e-1.
            scale = self._clamp_scale(max(float(t_rel.norm(dim=-1).median()), 1e-1))
        else:
            scale = self._clamp_scale(
                (P.norm(dim=1) * c.valid).sum() / c.valid.sum().clamp_min(1))

        Hr, Wr = c.ray_hw
        rays = c.rays.clone()
        # UNNORMALIZED directions on the raymap grid, deliberately NOT divided by `scale`
        # (they are dimensionless -- dividing them would destroy the intrinsics they carry).
        rays[:, 0:3] = (R0T[None] @ c.d_ray).reshape(c.rays.shape[0], 3, Hr, Wr)
        rays[:, 3:6] = (t_rel / scale).view(-1, 3, 1, 1).expand_as(rays[:, 3:6])
        return rays, c.depth / scale, scale

    # ==================================================================================
    # Scheme 5 -- GenCeption (arXiv 2607.09024)
    # ==================================================================================

    def _genception(self, c):
        """Source of truth: arXiv 2607.09024 Secs. 3.3 and 3.5, and Figure 5.

        DEPTH, verbatim from Sec. 3.5:
          "we normalize the depth map of each video using the median depth of the scene,
           thereby inherently eliminating scale ambiguity."
          "d' = clip(alpha * log(d + 1), 0, 1)"
        and from Sec. 3.3: "three RGB channels are replicated for single-dimensional outputs
        like depth" -- which is already what this pipeline does
        (wan_t2v_ray_depth_mot_concat_5b.py:662, `depths.repeat(1,1,3,1,1)`).

        Note the scale is the PLAIN median over the whole clip -- not kappa * median. That
        is the substantive difference from this repo's current scheme-C log map, which is
        otherwise the same functional form.

        POSE, verbatim from Sec. 3.3:
          "we employ a pixel-space raymap ... To fit its 6-channel format within our
           3-channel constraint, we spatially partition the frame -- allocating ray origins
           to the central region and ray directions to the periphery."
        Figure 5 ("Rothko" Raymap) shows the direction field filling the frame with a single
        centred rectangle of flat colour (the spatially constant origin) painted over it.

        DERIVED (2 of 3): alpha is described ("dynamically adjusts the model's focus between
        near-field details and far-field structures") but never given a value. Default is
        this repo's already-calibrated 0.30; override with `dataset.genception_alpha`.

        DERIVED (3 of 3): the centre rectangle's size is shown but never stated. Measured off
        Figure 5 it is close to one third of each dimension; default 1/3 x 1/3, centred.
        Override with `dataset.genception_rothko_frac`.

        SCALE ALIGNMENT (C2): GenCeption states no shared depth/pose scale -- it normalizes
        depth by the scene median and says nothing about the raymap. Something must gauge
        the origin channel, and the least-invented choice is the paper's own scalar, the
        scene median depth, which also happens to satisfy C2. Set
        `dataset.genception_pose_gauge=independent` to gauge origins by their own per-clip
        max instead, which is closer to "the paper specifies nothing" but breaks C2.

        CHANNEL COUNT: the Rothko map is genuinely 3-channel, while this architecture has two
        3-channel ray blocks. The Rothko map is written to BOTH blocks, so the second block
        carries no information the first does not -- the architecture is untouched and no
        extra signal is smuggled in. That redundancy is inherent to running a 3-channel
        scheme on a 6-channel interface, and is the honest way to do it.
        """
        # Plain median of the scene's valid depth, at FULL depth resolution -- this statistic
        # needs no rays, so there is no reason to sample it on the coarse raymap grid.
        med = c.depth[c.valid].median() if bool(c.valid.any()) else torch.tensor(1.0)
        scale = self._clamp_scale(med)

        Hr, Wr = c.ray_hw
        if _get_str(self.cfg, "genception_pose_gauge", "shared") == "independent":
            pose_scale = self._clamp_scale(max(float(c.t.abs().max()), 1.0))
        else:
            pose_scale = scale
        _, t_rel = self._recentre_to_first_camera(c.R, c.t)

        # Rothko: unit directions everywhere, origins painted over a centred rectangle.
        rothko = c.rays[:, 0:3].clone()
        frac = _get_float(self.cfg, "genception_rothko_frac", 1.0 / 3.0)
        rh, rw = max(1, int(round(Hr * frac))), max(1, int(round(Wr * frac)))
        y0, x0 = (Hr - rh) // 2, (Wr - rw) // 2
        origin_c = (t_rel / pose_scale).view(-1, 3, 1, 1)
        rothko[:, :, y0:y0 + rh, x0:x0 + rw] = origin_c.expand(-1, -1, rh, rw)

        rays = c.rays.clone()
        rays[:, 0:3] = rothko
        rays[:, 3:6] = rothko

        # GenCeption's own bounded map, applied by `_to_unit_range` with alpha from the
        # GenCeption-specific knob. depth/scale is d / median(d), exactly their "d".
        return rays, c.depth / scale, scale


def apply_geometry_scheme(
    raymaps_raw: torch.Tensor,
    depth_metric: torch.Tensor,
    cfg,
    invalid_sentinel: float,
    cameras=None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Module-level entry point matching `_scale_norm.apply_scale_norm`'s signature."""
    return GeometryBuilder(cfg)(raymaps_raw, depth_metric, invalid_sentinel, cameras=cameras)


# ======================================================================================
# INVERSE API — used at EVALUATION time. Must stay the exact inverse of the forward maps.
# ======================================================================================
#
# Evaluation is where a normalization mismatch does the most damage, because it is silent:
# scoring a log-map arm with the disparity inverse yields plausible-looking but meaningless
# numbers. The stock scorers hardcode `disparity_norm_to_metric_depth`, which is correct ONLY
# for the legacy `F` / global_metric path. Everything below exists so that training,
# verification and scoring all invert through ONE implementation.

PLUCKER = "plucker"                    # ch3:6 = Plucker moment / s      (F, global_metric, vggt)
COMPANDED_ORIGIN = "companded_origin"  # ch3:6 = mu-law companded t / s  (C / parallax)
BROADCAST_ORIGIN = "broadcast_origin"  # ch3:6 = t / s, spatially const  (vggt_omega, pi3, da3)
ROTHKO = "rothko"                      # centred rect = t / s, periphery = directions (genception)


def ray_encoding_for(cfg) -> str:
    """Which semantics channels 3:6 carry, given the arm's config. Drives pose decode."""
    scheme = geometry_scheme(cfg)
    if scheme is None:
        # Legacy paths: parallax may use companded origins; everything else is a Plucker moment.
        mode = _get_str(cfg, "scale_mode", "")
        if mode == "parallax":
            enc = _get_str(cfg, "parallax_ray_encoding", "companded_origin")
            return COMPANDED_ORIGIN if enc == "companded_origin" else PLUCKER
        return PLUCKER
    if scheme == VGGT_MODE:
        return PLUCKER
    if scheme == GENCEPTION_MODE:
        return ROTHKO
    return BROADCAST_ORIGIN            # vggt_omega, pi3, da3


def carries_rotation(cfg) -> bool:
    """False for pi^3: its direction channels live in each frame's OWN camera frame, so they
    are identical across frames and encode NO rotation (verified: cross-frame spread ~1e-7).
    pi^3's real model puts rotation in a separate pose head, which this architecture does not
    have. Depth / NVS stay meaningful for that arm; ROTATION metrics do not."""
    return geometry_scheme(cfg) != PI3_MODE


def invert_depth_channel(depth_channel, scale: float, cfg):
    """Normalized depth channel in [-1,1] -> metric depth. Exact inverse of `_to_unit_range`
    composed with the arm's `depth / scale`.

    Returns (metric_depth, unrecoverable_mask). Unrecoverable pixels are those the forward map
    clipped (log arms) or drove into the ill-conditioned tail (disparity); they carry no
    information and must be excluded identically for every arm when scoring.
    """
    scheme = geometry_scheme(cfg)
    x = depth_channel

    # ---- legacy disparity paths (F, global_metric): x = 2/(1+u) - 1 ----
    if scheme is None and _get_str(cfg, "scale_mode", "") != "parallax":
        v = (x + 1.0) / 2.0
        far = _get_float(cfg, "eval_far_plane_m", 1000.0)
        v_min = 1.0 / (1.0 + far / max(scale, _EPS))
        return (1.0 / v.clamp_min(v_min) - 1.0) * scale, v <= v_min

    # ---- scheme C (parallax): x = 1 - 2*clip(alpha*ln(d/zbar + 1), 0, 1) ----
    if scheme is None:
        if _get_str(cfg, "parallax_depth_map", "log") != "log":
            v = (x + 1.0) / 2.0
            far = _get_float(cfg, "eval_far_plane_m", 1000.0)
            v_min = 1.0 / (1.0 + far / max(scale, _EPS))
            return (1.0 / v.clamp_min(v_min) - 1.0) * scale, v <= v_min
        alpha = _get_float(cfg, "parallax_log_alpha", RANGE_ALPHA_DEFAULT)
        kappa = _get_float(cfg, "parallax_kappa", 4.0)
        zbar = scale / max(kappa, _EPS)          # C divides by the MEDIAN, not by s
        v = (1.0 - x) / 2.0
        return torch.expm1(v.clamp(0.0, 1.0 - 1e-9) / alpha) * zbar, v >= 1.0 - 1e-6

    # ---- the five published schemes: shared Stage-B adapter ----
    gb = GeometryBuilder(cfg)
    which = _get_str(cfg, "geom_depth_range_map", RANGE_MAP_DEFAULT)
    v = (1.0 - x) / 2.0 if gb._orientation() == ORIENTATION_NEAR_POSITIVE else (x + 1.0) / 2.0
    if which == "disparity":
        u = v / (1.0 - v).clamp_min(1e-12)
        return u * scale, v >= 1.0 - 1e-6
    u = torch.expm1(v.clamp(0.0, 1.0 - 1e-9) / gb._range_alpha())
    return u * scale, v >= 1.0 - 1e-6


def origins_from_normalized_raymap(rays_norm, scale: float, cfg):
    """Recover metric camera origins (T,3) from an arm's NORMALIZED raymap.

    Pose metrics are Sim(3)-aligned, so the scalar `scale` is absorbed; what matters is
    reading channels 3:6 with the right SEMANTICS. Reading a companded origin as a Plucker
    moment (or vice versa) silently produces a wrong trajectory rather than an error.
    """
    enc = ray_encoding_for(cfg)
    if rays_norm.dim() == 3:
        rays_norm = rays_norm.unsqueeze(0)
    if enc == PLUCKER:
        return origins_from_plucker(rays_norm) * scale
    if enc == COMPANDED_ORIGIN:
        co = rays_norm[:, 3:6].mean(dim=(2, 3))
        mu = _get_float(cfg, "parallax_mu", 255.0)
        return _mulaw_expand(co, mu) * scale
    if enc == BROADCAST_ORIGIN:
        return rays_norm[:, 3:6].mean(dim=(2, 3)) * scale
    # ROTHKO: the origin lives ONLY in the centred rectangle; the periphery is directions, so
    # a whole-frame mean would blend the two and corrupt the trajectory.
    T, _, H, W = rays_norm.shape
    frac = _get_float(cfg, "genception_rothko_frac", 1.0 / 3.0)
    rh, rw = max(1, int(round(H * frac))), max(1, int(round(W * frac)))
    y0, x0 = (H - rh) // 2, (W - rw) // 2
    return rays_norm[:, 0:3, y0:y0 + rh, x0:x0 + rw].mean(dim=(2, 3)) * scale


def _mulaw_expand(y, mu: float):
    """sign(y)*((1+mu)^|y| - 1)/mu — inverse of the mu-law companding used by scheme C."""
    return torch.sign(y) * torch.expm1(y.abs() * math.log1p(mu)) / mu
