"""
Shared configuration and utility functions.
Imported by all other scripts in this project.
DO NOT RUN THIS FILE
"""
from __future__ import annotations

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import numpy as np
import h5py
import re
import json
import csv
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


# =============================================================================
# PATHS
# =============================================================================
DATA_ROOT   = Path("data/h5")
PROJECT_DIR = Path("kpms_project")

# =============================================================================
# DATA LOADING
# =============================================================================
FPS = 30
N_RANDOM_VIDEOS = 26
RANDOM_SEED     = 123

# Track-score masking
USE_TRACKING_SCORE_MASK    = True
TRACKING_SCORE_THRESHOLD   = 0.50

# QA anti-swaps (report only, no masking)
RUN_SWAP_QA = True

# =============================================================================
# NORMALISATION
# =============================================================================
NORMALIZE_COORDS  = True
NORMALIZE_BY_BONE = True   # must be True; global-quantile normaliser removed
BONE_A = "base_head"
BONE_B = "base_body"

# =============================================================================
# BODYPARTS
# =============================================================================
EXCLUDE_TAIL   = True
ANTERIOR_BPS   = ["nose", "upper_head"]
POSTERIOR_BPS  = ["base_body", "base_tail"]

# =============================================================================
# TRAINING
# =============================================================================
MULTI_SEED_PREFIX = "multi_seed"
NUM_MODEL_FITS    = 20
AR_ONLY_KAPPA     = 1e6
FULL_MODEL_KAPPA  = 1e4
NUM_AR_ITERS      = 50
NUM_FULL_ITERS    = 500

# =============================================================================
# MALE TRACK ASSIGNMENT
# =============================================================================
MALE_TRACK2_IDS = {
    "50_S2", "50_S3", "51_S2", "55_S3",
    "85_S3", "86_S2", "92_S2", "92_S3", "128_S3"
}


# =============================================================================
# I/O UTILITIES
# =============================================================================
def collect_h5_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.h5") if p.is_file()])


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(path: Path, rows: List[Dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def robust_mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-12


# =============================================================================
# SESSION / TRACK HELPERS
# =============================================================================
def extract_session_id_strict(stem: str) -> str:
    m = re.fullmatch(r"(\d+_S[23])", stem)
    if m is None:
        raise ValueError(
            f"\nNombre de archivo inválido: '{stem}'\n"
            "Se esperaba formato exacto: <numero>_S2 o <numero>_S3"
        )
    return m.group(1)


def male_track_index_from_stem_strict(stem: str) -> Tuple[int, str]:
    sid = extract_session_id_strict(stem)
    return (1, sid) if sid in MALE_TRACK2_IDS else (0, sid)


def subsample_files(h5_files: List[Path]) -> List[Path]:
    rng = np.random.default_rng(RANDOM_SEED)
    strata: Dict[str, List[Path]] = {"S2": [], "S3": []}
    for p in h5_files:
        sid = extract_session_id_strict(p.stem)
        strata[sid.split("_")[1]].append(p)

    n_use = min(N_RANDOM_VIDEOS, len(h5_files))
    n2, n3 = n_use // 2, n_use - n_use // 2

    out: List[Path] = []
    for key, n in [("S2", n2), ("S3", n3)]:
        pool = strata[key]
        if not pool:
            continue
        n = min(n, len(pool))
        idx = rng.choice(len(pool), size=n, replace=False)
        out.extend([pool[i] for i in idx])
    return sorted(out)


# =============================================================================
# SLEAP LOADER
# =============================================================================
@dataclass
class SleapLoaded:
    tracks_TK2R:       np.ndarray                    # (T,K,2,R)
    conf_TKR:          np.ndarray                    # (T,K,R)
    bodyparts:         List[str]
    skeleton_edges:    Optional[List[List[str]]]
    tracking_scores_RT: Optional[np.ndarray]         # (R,T)


def load_sleap_h5(h5_path: Path) -> SleapLoaded:
    with h5py.File(h5_path, "r") as f:
        tracks      = np.array(f["/tracks"],          dtype=np.float32)   # (R,2,K,T)
        occ         = np.array(f["/track_occupancy"], dtype=np.uint8)     # (T,R)
        node_names  = f["/node_names"][...]

        point_scores = (
            np.array(f["/point_scores"], dtype=np.float32)
            if "/point_scores" in f else None
        )
        tracking_scores = (
            np.array(f["/tracking_scores"], dtype=np.float32)
            if "/tracking_scores" in f else None
        )
        skeleton_edges = None
        if "/edge_names" in f:
            skeleton_edges = [
                [
                    a.decode("utf-8") if isinstance(a, (bytes, np.bytes_)) else str(a),
                    b.decode("utf-8") if isinstance(b, (bytes, np.bytes_)) else str(b),
                ]
                for a, b in f["/edge_names"][...]
            ]

    bodyparts = []
    for x in np.array(node_names):
        if isinstance(x, (bytes, np.bytes_)):
            bodyparts.append(x.decode("utf-8"))
        else:
            try:
                bodyparts.append(bytes(x).decode("utf-8"))
            except Exception:
                bodyparts.append(str(x))

    if tracks.ndim != 4:
        raise ValueError(f"{h5_path}: tracks.ndim={tracks.ndim}, expected 4")
    R, D, K, T = tracks.shape
    if D != 2:
        raise ValueError(f"{h5_path}: D={D}, expected 2 (x,y)")
    if len(bodyparts) != K:
        raise ValueError(f"{h5_path}: len(node_names)={len(bodyparts)} != K={K}")
    if occ.shape != (T, R):
        raise ValueError(f"{h5_path}: occupancy {occ.shape} != (T,R)=({T},{R})")

    tracks_TK2R = np.transpose(tracks, (3, 2, 1, 0))   # (T,K,2,R)

    if point_scores is not None:
        if point_scores.shape != (R, K, T):
            raise ValueError(f"{h5_path}: point_scores {point_scores.shape} != (R,K,T)")
        conf_TKR = np.transpose(point_scores, (2, 1, 0))
    else:
        conf_TKR = np.ones((T, K, R), dtype=np.float32)

    for r in range(R):
        missing = occ[:, r] == 0
        if np.any(missing):
            tracks_TK2R[missing, :, :, r] = np.nan
            conf_TKR[missing, :, r]        = 0.0

    invalid = np.isnan(tracks_TK2R[:, :, 0, :]) | np.isnan(tracks_TK2R[:, :, 1, :])
    conf_TKR[invalid] = 0.0

    return SleapLoaded(
        tracks_TK2R=tracks_TK2R,
        conf_TKR=conf_TKR,
        bodyparts=bodyparts,
        skeleton_edges=skeleton_edges,
        tracking_scores_RT=tracking_scores,
    )


# =============================================================================
# QA ANTI-SWAPS
# =============================================================================
def swap_qa_report(tracks_TK2R: np.ndarray) -> Dict[str, float]:
    T, K, _, R = tracks_TK2R.shape
    cent = np.full((T, 2, R), np.nan, dtype=np.float32)
    for r in range(R):
        xy    = tracks_TK2R[:, :, :, r]
        valid = np.isfinite(xy).all(axis=2)
        for t in range(T):
            v = valid[t]
            if np.any(v):
                cent[t, :, r] = np.nanmean(xy[t, v, :], axis=0)

    report: Dict[str, float] = {}
    for r in range(R):
        c  = cent[:, :, r]
        dc = np.linalg.norm(np.diff(c, axis=0), axis=1)
        mad = robust_mad(dc)
        med = np.nanmedian(dc)
        thr = med + 10.0 * mad
        report[f"track{r+1}_speed_median"]   = float(med)
        report[f"track{r+1}_speed_mad"]      = float(mad)
        report[f"track{r+1}_jumps_10MAD"]    = float(np.nansum(dc > thr))

    if R >= 2:
        d12 = np.linalg.norm(cent[:, :, 0] - cent[:, :, 1], axis=1)
        fin = d12[np.isfinite(d12)]
        report["intertrack_dist_median"] = float(np.nanmedian(d12))
        report["intertrack_dist_p05"]    = float(np.nanquantile(fin, 0.05)) if fin.size else float("nan")

    return report


# =============================================================================
# NORMALISATION
# =============================================================================
def normalize_by_bone_length(
    coordinates_dict: Dict[str, np.ndarray],
    bodyparts: List[str],
    a: str,
    b: str,
) -> Tuple[Dict[str, np.ndarray], float]:
    if a not in bodyparts or b not in bodyparts:
        raise ValueError(
            f"NORMALIZE_BY_BONE requires '{a}' and '{b}' in bodyparts.\n"
            f"Available: {bodyparts}"
        )
    ia, ib = bodyparts.index(a), bodyparts.index(b)

    all_lengths: List[np.ndarray] = []
    for c in coordinates_dict.values():
        pa, pb = c[:, ia, :], c[:, ib, :]
        valid  = np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1)
        if np.any(valid):
            all_lengths.append(np.linalg.norm(pa[valid] - pb[valid], axis=1))

    if not all_lengths:
        raise ValueError(f"Could not compute bone length for '{a}'→'{b}': no valid frames.")

    scale = float(np.median(np.concatenate(all_lengths))) + 1e-12
    print(f"[NORM] Global median bone length ({a}→{b}): {scale:.4f} px")

    out: Dict[str, np.ndarray] = {}
    for rec, c in coordinates_dict.items():
        c = c.copy()
        T = c.shape[0]
        centroid = np.full((T, 2), np.nan, dtype=c.dtype)
        for t in range(T):
            valid_kp = np.isfinite(c[t]).all(axis=1)
            if np.any(valid_kp):
                centroid[t] = np.nanmean(c[t, valid_kp, :], axis=0)
        c = (c - centroid[:, np.newaxis, :]) / scale
        out[rec] = c

    return out, scale
