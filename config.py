"""
Shared configuration and utility functions.
Imported by all other scripts in this project. Never executed directly.
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
FPS             = 30
N_RANDOM_VIDEOS = 26
RANDOM_SEED     = 123

# Track-score masking
USE_TRACKING_SCORE_MASK  = True
TRACKING_SCORE_THRESHOLD = 0.50

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
EXCLUDE_TAIL    = True
ANTERIOR_BPS    = ["nose", "upper_head"]
POSTERIOR_BPS   = ["base_body", "base_tail"]

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
# KAPPA SCAN  (used by 01_explore.py)
# =============================================================================
KAPPA_SCAN_PREFIX     = "kappa_scan"
KAPPA_SCAN_VALUES     = np.logspace(3, 7, 5)   # [1e3, ~3e4, 1e5, ~3e5, 1e7]
KAPPA_DECREASE_FACTOR = 10
NUM_AR_ITERS_SCAN     = 50
NUM_FULL_ITERS_SCAN   = 200

# =============================================================================
# MALE TRACK ASSIGNMENT
# =============================================================================
MALE_TRACK2_IDS = {
    "50_S2", "50_S3", "51_S2", "55_S3",
    "85_S3", "86_S2", "92_S2", "92_S3", "128_S3"
}

# =============================================================================
# DEFAULT SKELETON  (fallback if not present in .h5)
# =============================================================================
DEFAULT_SKELETON = [
    ["nose",       "upper_head"],
    ["upper_head", "base_head"],
    ["base_head",  "upper_body"],
    ["upper_body", "base_body"],
    ["base_body",  "base_tail"],
    ["base_head",  "L_ear"],
    ["base_head",  "R_ear"],
    ["base_body",  "L_hip"],
    ["base_body",  "R_hip"],
    ["upper_body", "L_sh"],
    ["upper_body", "R_sh"],
]


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
    tracks_TK2R:        np.ndarray
    conf_TKR:           np.ndarray
    bodyparts:          List[str]
    skeleton_edges:     Optional[List[List[str]]]
    tracking_scores_RT: Optional[np.ndarray]


def load_sleap_h5(h5_path: Path) -> SleapLoaded:
    with h5py.File(h5_path, "r") as f:
        tracks     = np.array(f["/tracks"],          dtype=np.float32)
        occ        = np.array(f["/track_occupancy"], dtype=np.uint8)
        node_names = f["/node_names"][...]

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

    tracks_TK2R = np.transpose(tracks, (3, 2, 1, 0))

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
        c   = cent[:, :, r]
        dc  = np.linalg.norm(np.diff(c, axis=0), axis=1)
        mad = robust_mad(dc)
        med = np.nanmedian(dc)
        thr = med + 10.0 * mad
        report[f"track{r+1}_speed_median"] = float(med)
        report[f"track{r+1}_speed_mad"]    = float(mad)
        report[f"track{r+1}_jumps_10MAD"]  = float(np.nansum(dc > thr))

    if R >= 2:
        d12 = np.linalg.norm(cent[:, :, 0] - cent[:, :, 1], axis=1)
        fin = d12[np.isfinite(d12)]
        report["intertrack_dist_median"] = float(np.nanmedian(d12))
        report["intertrack_dist_p05"]    = float(np.nanquantile(fin, 0.05)) if fin.size else float("nan")

    return report


# =============================================================================
# PER-RECORDING QA METRICS  (anterior / posterior validity)
# =============================================================================
def recording_qa_metrics(
    rec: str,
    h5_name: str,
    T: int,
    K: int,
    coords_male: np.ndarray,
    bodyparts: List[str],
    swap_rep: Dict[str, float],
) -> Dict:
    valid_xy = np.isfinite(coords_male).all(axis=2)
    finite   = coords_male[np.isfinite(coords_male)]

    anterior_ok  = float("nan")
    posterior_ok = float("nan")

    if all(x in bodyparts for x in ANTERIOR_BPS):
        idxs = [bodyparts.index(x) for x in ANTERIOR_BPS]
        anterior_ok = float(np.mean(np.isfinite(coords_male[:, idxs, :]).all(axis=2)))

    if all(x in bodyparts for x in POSTERIOR_BPS):
        idxs = [bodyparts.index(x) for x in POSTERIOR_BPS]
        posterior_ok = float(np.mean(np.isfinite(coords_male[:, idxs, :]).all(axis=2)))

    m: Dict = {
        "recording":            rec,
        "file":                 h5_name,
        "T_frames":             T,
        "K_keypoints":          K,
        "mask_frac_xy":         float(np.mean(valid_xy)) if valid_xy.size else 0.0,
        "coord_min":            float(np.min(finite))    if finite.size  else float("nan"),
        "coord_max":            float(np.max(finite))    if finite.size  else float("nan"),
        "anterior_valid_frac":  anterior_ok,
        "posterior_valid_frac": posterior_ok,
    }
    m.update({f"swapQA_{k}": v for k, v in swap_rep.items()})
    return m


# =============================================================================
# SKELETON / BODYPART VALIDATION  (hard checks)
# =============================================================================
def validate_skeleton_and_bodyparts(
    skeleton: List[List[str]],
    bodyparts: List[str],
    anterior_bodyparts: List[str],
    posterior_bodyparts: List[str],
) -> None:
    bp_set = set(bodyparts)
    for a, b in skeleton:
        if a not in bp_set or b not in bp_set:
            raise ValueError(f"Skeleton edge ({a}, {b}) not in bodyparts")
    for x in anterior_bodyparts + posterior_bodyparts:
        if x not in bp_set:
            raise ValueError(f"Bodypart '{x}' not in bodyparts")


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


# =============================================================================
# SHARED DATA LOADING PIPELINE
# Called by 01_explore.py, 02_train.py, 04_visualize.py, 05_merge_syllables.py
# =============================================================================
def load_and_preprocess(save_qa: bool = True):

    qa_dir = PROJECT_DIR / "QA_AUDIT"
    ensure_dir(qa_dir)

    h5_files = collect_h5_files(DATA_ROOT)
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {DATA_ROOT.resolve()}")
    h5_files = subsample_files(h5_files)

    if save_qa:
        save_json(qa_dir / "used_h5_files.json", [str(p.resolve()) for p in h5_files])

    print(f"\n[DATA] Using {len(h5_files)} videos:")
    for p in h5_files:
        print("  -", p.name)

    coordinates_dict:     Dict[str, np.ndarray]    = {}
    confidences_dict:     Dict[str, np.ndarray]    = {}
    bodyparts_ref:        Optional[List[str]]       = None
    skeleton_ref:         Optional[List[List[str]]] = None
    male_track1_sessions: List[str]                 = []
    male_track2_sessions: List[str]                 = []
    per_recording_metrics: List[Dict]               = []

    for h5_path in h5_files:
        loaded     = load_sleap_h5(h5_path)
        T, K, _, R = loaded.tracks_TK2R.shape

        if bodyparts_ref is None:
            bodyparts_ref = loaded.bodyparts
            skeleton_ref  = loaded.skeleton_edges
        elif loaded.bodyparts != bodyparts_ref:
            raise ValueError(
                f"Bodypart mismatch in {h5_path}.\n"
                f"Expected: {bodyparts_ref}\n"
                f"Got:      {loaded.bodyparts}"
            )

        rep      = swap_qa_report(loaded.tracks_TK2R) if RUN_SWAP_QA else {}
        male_r, sid = male_track_index_from_stem_strict(h5_path.stem)

        if not (0 <= male_r < R):
            raise ValueError(f"{h5_path.name}: male_r={male_r} out of range for R={R}")

        if USE_TRACKING_SCORE_MASK and loaded.tracking_scores_RT is not None:
            ts  = loaded.tracking_scores_RT[male_r]
            low = ts < TRACKING_SCORE_THRESHOLD
            if np.any(low):
                loaded.tracks_TK2R[low, :, :, male_r] = np.nan
                loaded.conf_TKR[low, :, male_r]        = 0.0

        print(f"[MALE SELECT] {h5_path.name} → SID={sid} → Track {male_r + 1}")
        (male_track2_sessions if male_r == 1 else male_track1_sessions).append(sid)

        rec         = f"{sid}__male_track{male_r + 1}"
        coords_male = loaded.tracks_TK2R[:, :, :, male_r]
        conf_male   = loaded.conf_TKR[:, :, male_r]

        per_recording_metrics.append(
            recording_qa_metrics(rec, h5_path.name, T, K, coords_male, bodyparts_ref, rep)
        )
        coordinates_dict[rec] = coords_male
        confidences_dict[rec] = conf_male

    if save_qa:
        save_csv(qa_dir / "per_recording_metrics.csv", per_recording_metrics)

    # Hard check: no unexpected Track2 sessions
    detected_track2 = set(male_track2_sessions)
    if not detected_track2.issubset(MALE_TRACK2_IDS):
        extras = sorted(detected_track2 - MALE_TRACK2_IDS)
        raise ValueError(
            "\nERROR: Track2 sessions detected that are NOT in MALE_TRACK2_IDS.\n"
            f"Unexpected: {extras}\n"
            f"MALE_TRACK2_IDS: {sorted(MALE_TRACK2_IDS)}"
        )

    print("\n=== TRACK SUMMARY ===")
    print("Track 1 (male):", sorted(male_track1_sessions))
    print("Track 2 (male):", sorted(male_track2_sessions))
    print("====================\n")

    # Normalisation
    if NORMALIZE_COORDS:
        if not NORMALIZE_BY_BONE:
            raise ValueError("NORMALIZE_BY_BONE must be True in config.py")
        coordinates_dict, scale = normalize_by_bone_length(
            coordinates_dict, bodyparts_ref, BONE_A, BONE_B
        )
        if save_qa:
            save_json(qa_dir / "normalization.json", {
                "method":   "per_recording_centering_then_bone_median_scale",
                "bone":     [BONE_A, BONE_B],
                "scale_px": scale,
            })

    return coordinates_dict, confidences_dict, bodyparts_ref, skeleton_ref


# =============================================================================
# SHARED kpms CONFIG + FORMAT PIPELINE
# =============================================================================
def build_kpms_data(
    coordinates_dict: Dict[str, np.ndarray],
    confidences_dict: Dict[str, np.ndarray],
    bodyparts: List[str],
    skeleton: Optional[List[List[str]]],
    estimate_sigmasq: bool = True,
):

    import keypoint_moseq as kpms
    from jax_moseq.utils.debugging import convert_data_precision

    skeleton = skeleton or DEFAULT_SKELETON

    if EXCLUDE_TAIL:
        use_bodyparts = [bp for bp in bodyparts if "tail" not in bp.lower()]
        if "base_tail" in bodyparts and "base_tail" not in use_bodyparts:
            use_bodyparts.append("base_tail")
    else:
        use_bodyparts = bodyparts

    anterior_bodyparts  = [bp for bp in ANTERIOR_BPS  if bp in bodyparts]
    posterior_bodyparts = [bp for bp in POSTERIOR_BPS if bp in bodyparts]

    # Hard checks
    validate_skeleton_and_bodyparts(
        skeleton, bodyparts, anterior_bodyparts, posterior_bodyparts
    )

    ensure_dir(PROJECT_DIR)
    if not (PROJECT_DIR / "config.yml").exists():
        kpms.setup_project(str(PROJECT_DIR), overwrite=True)

    kpms.update_config(
        str(PROJECT_DIR),
        fps=FPS,
        bodyparts=bodyparts,
        use_bodyparts=use_bodyparts,
        skeleton=skeleton,
        anterior_bodyparts=anterior_bodyparts,
        posterior_bodyparts=posterior_bodyparts,
        outlier_scale_factor=6.0,
    )
    cfg = kpms.load_config(str(PROJECT_DIR))

    coordinates, confidences = kpms.outlier_removal(
        coordinates_dict, confidences_dict, str(PROJECT_DIR),
        overwrite=False, **cfg
    )
    data, metadata = kpms.format_data(coordinates, confidences, **cfg)
    data = convert_data_precision(data)

    if estimate_sigmasq:
        # Only estimate and write sigmasq_loc during explore / training.
        # After training, config.yml already holds the correct value.
        kpms.update_config(
            str(PROJECT_DIR),
            sigmasq_loc=kpms.estimate_sigmasq_loc(
                data["Y"], data["mask"], filter_size=cfg["fps"]
            )
        )
    else:
        # Verify that sigmasq_loc was already set during training.
        if cfg.get("sigmasq_loc") is None:
            raise RuntimeError(
                "sigmasq_loc not found in config.yml."
                "Make sure 01_explore.py and 02_train.py have been run first."
            )

    cfg = kpms.load_config(str(PROJECT_DIR))
    return data, metadata, coordinates, cfg
