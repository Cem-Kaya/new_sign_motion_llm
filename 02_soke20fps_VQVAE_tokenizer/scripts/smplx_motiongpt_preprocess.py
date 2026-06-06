from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


# SMPL-X 169 layout used in this workspace.
SMPLX_NUM_JOINTS = 52
SMPLX_AA_DIMS = 156
SMPLX_BETAS_DIMS = 10
SMPLX_TRANSL_DIMS = 3
SMPLX_TOTAL_DIMS = 169


@dataclass
class CanonicalMeta:
    yaw0: float
    floor_y: float
    origin_x: float
    origin_z: float

    def to_array(self) -> np.ndarray:
        return np.asarray([self.yaw0, self.floor_y, self.origin_x, self.origin_z], dtype=np.float32)

    @staticmethod
    def from_array(arr: np.ndarray) -> "CanonicalMeta":
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        if a.shape[0] != 4:
            raise ValueError(f"Expected meta array with 4 values, got shape={a.shape}")
        return CanonicalMeta(yaw0=float(a[0]), floor_y=float(a[1]), origin_x=float(a[2]), origin_z=float(a[3]))


def _safe_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def _skew(v: np.ndarray) -> np.ndarray:
    """
    v: [..., 3]
    return: [..., 3, 3]
    """
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    o = np.zeros_like(vx)
    out = np.stack(
        [
            np.stack([o, -vz, vy], axis=-1),
            np.stack([vz, o, -vx], axis=-1),
            np.stack([-vy, vx, o], axis=-1),
        ],
        axis=-2,
    )
    return out


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """
    axis_angle: [..., 3]
    return: [..., 3, 3]
    """
    aa = np.asarray(axis_angle, dtype=np.float32)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = _safe_normalize(aa, axis=-1)

    ct = np.cos(theta)[..., None]
    st = np.sin(theta)[..., None]

    k = _skew(axis)
    outer = axis[..., :, None] * axis[..., None, :]

    eye = np.zeros(aa.shape[:-1] + (3, 3), dtype=np.float32)
    eye[..., 0, 0] = 1.0
    eye[..., 1, 1] = 1.0
    eye[..., 2, 2] = 1.0

    r = ct * eye + (1.0 - ct) * outer + st * k

    small = (theta[..., 0] < 1e-8)
    if np.any(small):
        # First-order near zero is adequate for tiny angles.
        r[small] = eye[small] + _skew(aa[small])
    return r


def matrix_to_quaternion(r: np.ndarray) -> np.ndarray:
    """
    r: [..., 3, 3]
    return quaternion [..., 4] in (w, x, y, z)
    """
    m = np.asarray(r, dtype=np.float32)
    out = np.zeros(m.shape[:-2] + (4,), dtype=np.float32)

    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]

    mask = trace > 0.0
    if np.any(mask):
        s = np.sqrt(trace[mask] + 1.0) * 2.0
        out[mask, 0] = 0.25 * s
        out[mask, 1] = (m[mask, 2, 1] - m[mask, 1, 2]) / s
        out[mask, 2] = (m[mask, 0, 2] - m[mask, 2, 0]) / s
        out[mask, 3] = (m[mask, 1, 0] - m[mask, 0, 1]) / s

    mask1 = (~mask) & (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2])
    if np.any(mask1):
        s = np.sqrt(1.0 + m[mask1, 0, 0] - m[mask1, 1, 1] - m[mask1, 2, 2]) * 2.0
        out[mask1, 0] = (m[mask1, 2, 1] - m[mask1, 1, 2]) / s
        out[mask1, 1] = 0.25 * s
        out[mask1, 2] = (m[mask1, 0, 1] + m[mask1, 1, 0]) / s
        out[mask1, 3] = (m[mask1, 0, 2] + m[mask1, 2, 0]) / s

    mask2 = (~mask) & (~mask1) & (m[..., 1, 1] > m[..., 2, 2])
    if np.any(mask2):
        s = np.sqrt(1.0 + m[mask2, 1, 1] - m[mask2, 0, 0] - m[mask2, 2, 2]) * 2.0
        out[mask2, 0] = (m[mask2, 0, 2] - m[mask2, 2, 0]) / s
        out[mask2, 1] = (m[mask2, 0, 1] + m[mask2, 1, 0]) / s
        out[mask2, 2] = 0.25 * s
        out[mask2, 3] = (m[mask2, 1, 2] + m[mask2, 2, 1]) / s

    mask3 = (~mask) & (~mask1) & (~mask2)
    if np.any(mask3):
        s = np.sqrt(1.0 + m[mask3, 2, 2] - m[mask3, 0, 0] - m[mask3, 1, 1]) * 2.0
        out[mask3, 0] = (m[mask3, 1, 0] - m[mask3, 0, 1]) / s
        out[mask3, 1] = (m[mask3, 0, 2] + m[mask3, 2, 0]) / s
        out[mask3, 2] = (m[mask3, 1, 2] + m[mask3, 2, 1]) / s
        out[mask3, 3] = 0.25 * s

    # Standardize sign to reduce axis-angle ambiguity.
    neg = out[..., 0] < 0
    out[neg] = -out[neg]
    out = _safe_normalize(out, axis=-1)
    return out


def quaternion_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """
    quat: [..., 4] in (w, x, y, z)
    return axis-angle [..., 3]
    """
    q = np.asarray(quat, dtype=np.float32)
    q = _safe_normalize(q, axis=-1)

    w = np.clip(q[..., 0], -1.0, 1.0)
    xyz = q[..., 1:]
    n = np.linalg.norm(xyz, axis=-1, keepdims=True)

    angle = 2.0 * np.arctan2(n, np.maximum(w[..., None], 1e-8))
    axis = _safe_normalize(xyz, axis=-1)

    aa = axis * angle
    small = (n[..., 0] < 1e-8)
    if np.any(small):
        aa[small] = 0.0
    return aa.astype(np.float32)


def matrix_to_axis_angle(r: np.ndarray) -> np.ndarray:
    q = matrix_to_quaternion(r)
    return quaternion_to_axis_angle(q)


def matrix_to_rotation_6d(r: np.ndarray) -> np.ndarray:
    """
    r: [..., 3, 3]
    return: [..., 6] (first two columns)
    """
    return np.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1).astype(np.float32)


def rotation_6d_to_matrix(x: np.ndarray) -> np.ndarray:
    """
    x: [..., 6]
    return: [..., 3, 3]
    """
    v = np.asarray(x, dtype=np.float32)
    a1 = v[..., 0:3]
    a2 = v[..., 3:6]

    b1 = _safe_normalize(a1, axis=-1)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = _safe_normalize(a2 - dot * b1, axis=-1)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def rot_y(yaw_rad: float) -> np.ndarray:
    c = float(math.cos(yaw_rad))
    s = float(math.sin(yaw_rad))
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


class SMPLXMotionGPTCanonicalizer:
    """
    MotionGPT-style canonicalization for SMPL-X with an invertible feature pack.

    Feature pack per frame:
    - root local velocity xz: 2 dims
    - canonical root position xz: 2 dims
    - canonical root height y: 1 dim
    - all-joint rotation 6D (52 joints): 312 dims
    - betas: 10 dims
    Total: 327 dims

    This keeps root/facing/floor canonicalization while remaining invertible back to
    SMPL-X 169-D pose stream.
    """

    def __init__(self, include_betas: bool = True):
        self.include_betas = bool(include_betas)
        self.rot6d_dim = SMPLX_NUM_JOINTS * 6
        self.beta_dim = SMPLX_BETAS_DIMS if self.include_betas else 0
        self.feature_dim = 2 + 2 + 1 + self.rot6d_dim + self.beta_dim

    def _split_smplx_169(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(arr, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] < SMPLX_TOTAL_DIMS:
            raise ValueError(f"Expected [T,169+] SMPL-X array, got shape={x.shape}")
        x = x[:, :SMPLX_TOTAL_DIMS]
        aa = x[:, :SMPLX_AA_DIMS].reshape(-1, SMPLX_NUM_JOINTS, 3)
        betas = x[:, SMPLX_AA_DIMS : SMPLX_AA_DIMS + SMPLX_BETAS_DIMS]
        transl = x[:, SMPLX_AA_DIMS + SMPLX_BETAS_DIMS : SMPLX_TOTAL_DIMS]
        return aa, betas, transl

    def encode(self, smplx_169: np.ndarray) -> Tuple[np.ndarray, CanonicalMeta]:
        aa, betas, transl = self._split_smplx_169(smplx_169)
        t = aa.shape[0]

        r_all = axis_angle_to_matrix(aa.reshape(-1, 3)).reshape(t, SMPLX_NUM_JOINTS, 3, 3)
        r_root = r_all[:, 0]

        forward0 = r_root[0] @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        yaw0 = float(math.atan2(float(forward0[0]), float(forward0[2])))
        r_align = rot_y(-yaw0)

        r_root_can = np.einsum("ij,tjk->tik", r_align, r_root)
        transl_can = np.einsum("ij,tj->ti", r_align, transl)

        floor_y = float(np.min(transl_can[:, 1]))
        transl_can[:, 1] -= floor_y

        origin_x = float(transl_can[0, 0])
        origin_z = float(transl_can[0, 2])
        transl_can[:, 0] -= origin_x
        transl_can[:, 2] -= origin_z

        r_all[:, 0] = r_root_can
        rot6d = matrix_to_rotation_6d(r_all.reshape(-1, 3, 3)).reshape(t, -1)

        root_vel_local = np.zeros((t, 2), dtype=np.float32)
        if t > 1:
            d_world = transl_can[1:] - transl_can[:-1]
            d_local = np.einsum("tij,tj->ti", np.transpose(r_root_can[1:], (0, 2, 1)), d_world)
            root_vel_local[1:, :] = d_local[:, [0, 2]].astype(np.float32)

        root_pos_xz = transl_can[:, [0, 2]].astype(np.float32)
        root_h = transl_can[:, 1:2].astype(np.float32)

        parts = [root_vel_local, root_pos_xz, root_h, rot6d.astype(np.float32)]
        if self.include_betas:
            parts.append(betas.astype(np.float32))
        feat = np.concatenate(parts, axis=-1).astype(np.float32)

        meta = CanonicalMeta(yaw0=yaw0, floor_y=floor_y, origin_x=origin_x, origin_z=origin_z)
        return feat, meta

    def decode(self, feat: np.ndarray, meta: CanonicalMeta | np.ndarray | Dict[str, float]) -> np.ndarray:
        x = np.asarray(feat, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected feature array [T,F], got {x.shape}")
        if x.shape[1] < self.feature_dim:
            raise ValueError(f"Expected feature dim >= {self.feature_dim}, got {x.shape[1]}")

        if isinstance(meta, dict):
            m = CanonicalMeta(
                yaw0=float(meta["yaw0"]),
                floor_y=float(meta["floor_y"]),
                origin_x=float(meta["origin_x"]),
                origin_z=float(meta["origin_z"]),
            )
        elif isinstance(meta, CanonicalMeta):
            m = meta
        else:
            m = CanonicalMeta.from_array(np.asarray(meta, dtype=np.float32))

        t = x.shape[0]

        i = 0
        root_vel = x[:, i : i + 2]
        i += 2
        root_pos_xz = x[:, i : i + 2]
        i += 2
        root_h = x[:, i : i + 1]
        i += 1

        rot6d = x[:, i : i + self.rot6d_dim].reshape(t * SMPLX_NUM_JOINTS, 6)
        i += self.rot6d_dim
        betas = x[:, i : i + self.beta_dim] if self.include_betas else np.zeros((t, 10), dtype=np.float32)

        r_all = rotation_6d_to_matrix(rot6d).reshape(t, SMPLX_NUM_JOINTS, 3, 3)
        r_root_can = r_all[:, 0]

        # Use absolute canonical root xz and y for stable window-wise inversion.
        transl_can = np.zeros((t, 3), dtype=np.float32)
        transl_can[:, [0, 2]] = root_pos_xz
        transl_can[:, 1:2] = root_h

        # Optional consistency check hook (unused in payload, but computed once for debugging).
        _ = root_vel

        transl_can[:, 0] += float(m.origin_x)
        transl_can[:, 2] += float(m.origin_z)
        transl_can[:, 1] += float(m.floor_y)

        r_unalign = rot_y(float(m.yaw0))
        r_root_world = np.einsum("ij,tjk->tik", r_unalign, r_root_can)
        transl_world = np.einsum("ij,tj->ti", r_unalign, transl_can)

        r_all[:, 0] = r_root_world
        aa = matrix_to_axis_angle(r_all.reshape(-1, 3, 3)).reshape(t, SMPLX_NUM_JOINTS, 3)

        out = np.zeros((t, SMPLX_TOTAL_DIMS), dtype=np.float32)
        out[:, :SMPLX_AA_DIMS] = aa.reshape(t, -1)
        out[:, SMPLX_AA_DIMS : SMPLX_AA_DIMS + SMPLX_BETAS_DIMS] = betas[:, :SMPLX_BETAS_DIMS]
        out[:, SMPLX_AA_DIMS + SMPLX_BETAS_DIMS : SMPLX_TOTAL_DIMS] = transl_world
        return out


def geodesic_rotation_error_deg(aa_a: np.ndarray, aa_b: np.ndarray) -> np.ndarray:
    """
    aa_a, aa_b: [T, J, 3]
    returns per-frame-per-joint angle error in degrees: [T, J]
    """
    r_a = axis_angle_to_matrix(aa_a.reshape(-1, 3)).reshape(*aa_a.shape[:-1], 3, 3)
    r_b = axis_angle_to_matrix(aa_b.reshape(-1, 3)).reshape(*aa_b.shape[:-1], 3, 3)

    r_rel = np.einsum("...ij,...jk->...ik", r_a, np.transpose(r_b, (0, 1, 3, 2)))
    tr = np.trace(r_rel, axis1=-2, axis2=-1)
    cos = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    ang = np.arccos(cos)
    return np.degrees(ang).astype(np.float32)


def roundtrip_metrics_smplx169(x_ref: np.ndarray, x_rst: np.ndarray) -> Dict[str, float]:
    ref = np.asarray(x_ref, dtype=np.float32)
    rst = np.asarray(x_rst, dtype=np.float32)
    if ref.shape != rst.shape:
        raise ValueError(f"Shape mismatch: ref={ref.shape}, rst={rst.shape}")

    ref169 = ref[:, :SMPLX_TOTAL_DIMS]
    rst169 = rst[:, :SMPLX_TOTAL_DIMS]

    diff = rst169 - ref169
    mse_all = float(np.mean(diff * diff))
    mae_all = float(np.mean(np.abs(diff)))

    t_ref = ref169[:, 166:169]
    t_rst = rst169[:, 166:169]
    t_rmse = float(np.sqrt(np.mean((t_rst - t_ref) ** 2)))
    t_mae = float(np.mean(np.abs(t_rst - t_ref)))

    aa_ref = ref169[:, :156].reshape(-1, SMPLX_NUM_JOINTS, 3)
    aa_rst = rst169[:, :156].reshape(-1, SMPLX_NUM_JOINTS, 3)
    geod = geodesic_rotation_error_deg(aa_ref, aa_rst)

    return {
        "mse_all": mse_all,
        "mae_all": mae_all,
        "trans_rmse": t_rmse,
        "trans_mae": t_mae,
        "rot_geod_deg_mean": float(np.mean(geod)),
        "rot_geod_deg_p95": float(np.percentile(geod, 95)),
        "root_geod_deg_mean": float(np.mean(geod[:, 0])),
        "bodyhands_geod_deg_mean": float(np.mean(geod[:, 1:])),
    }
