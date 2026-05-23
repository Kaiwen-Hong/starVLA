"""
Quaternion (xyzw, scalar-last) <-> 6D continuous rotation.

Why custom: starVLA's existing RotationTransform wraps pytorch3d, whose
quaternion convention is wxyz (scalar-first). Our HDF5 endposes store xyzw
(SAPIEN/ManiSkill default). Going through pytorch3d would silently swap w
with x.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _SciR


def quat_xyzw_to_6d(quat: np.ndarray) -> np.ndarray:
    """
    (..., 4) xyzw -> (..., 6) = flatten of R[:, 0:2] (column-stack).

    Identity quat (0, 0, 0, 1) -> (1, 0, 0,  0, 1, 0).
    """
    quat = np.asarray(quat)
    flat = quat.reshape(-1, 4)
    Rmat = _SciR.from_quat(flat).as_matrix()
    six = np.concatenate([Rmat[:, :, 0], Rmat[:, :, 1]], axis=-1)
    return six.reshape(*quat.shape[:-1], 6).astype(quat.dtype, copy=False)


def six_d_to_quat_xyzw(d6: np.ndarray) -> np.ndarray:
    """
    (..., 6) -> (..., 4) xyzw. Gram-Schmidt orthogonalize, then matrix -> quat.
    """
    d6 = np.asarray(d6)
    a1 = d6[..., :3]
    a2 = d6[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    a2_proj = (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = a2 - a2_proj
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    Rmat = np.stack([b1, b2, b3], axis=-1)
    flat = Rmat.reshape(-1, 3, 3)
    quat = _SciR.from_matrix(flat).as_quat()
    return quat.reshape(*d6.shape[:-1], 4).astype(d6.dtype, copy=False)


def _sanity():
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    six = quat_xyzw_to_6d(identity)
    assert np.allclose(six, [1, 0, 0, 0, 1, 0]), f"identity quat -> 6D got {six}"

    rng = np.random.default_rng(0)
    quats = rng.standard_normal((50, 4))
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    six = quat_xyzw_to_6d(quats)
    quats_round = six_d_to_quat_xyzw(six)
    # Quaternion double-cover: q and -q describe the same rotation.
    dots = np.abs((quats * quats_round).sum(axis=-1))
    assert np.allclose(dots, 1.0, atol=1e-5), f"round-trip max dev {1 - dots.min():.2e}"

    batched = rng.standard_normal((3, 7, 4))
    batched /= np.linalg.norm(batched, axis=-1, keepdims=True)
    assert quat_xyzw_to_6d(batched).shape == (3, 7, 6)
    print("rotation.py sanity OK")


if __name__ == "__main__":
    _sanity()
