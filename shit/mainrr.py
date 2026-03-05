from __future__ import annotations

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import numpy as np
import h5py
from pathlib import Path
import keypoint_moseq as kpms
import re
import json
import csv
import os
import jax
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on headless servers
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


# =============================
# TRACKS MACHO
# =============================
MALE_TRACK2_IDS = {
    "50_S2", "50_S3", "51_S2", "55_S3",
    "85_S3", "86_S2", "92_S2", "92_S3", "128_S3"
}

def extract_session_id_strict(stem: str) -> str:
    m = re.fullmatch(r"(\d+_S[23])", stem)
    if m is None:
        raise ValueError(
            f"\nNombre de archivo inválido: '{stem}'\n"
            "Se esperaba formato exacto: <numero>_S2 o <numero>_S3"
        )
    return m.group(1)

def male_track_index_from_stem_strict(stem: str) -> tuple[int, str]:
    sid = extract_session_id_strict(stem)
    if sid in MALE_TRACK2_IDS:
        return 1, sid   # Track 2
    else:
        return 0, sid   # Track 1


# =============================
# CONFIG AJUSTABLE
# =============================
FPS = 30
DATA_ROOT = Path("../data/h5")
PROJECT_DIR = Path("kpms_project")

N_RANDOM_VIDEOS = 26
RANDOM_SEED = 123

# Enmascara frames donde tracking_score es bajo (si existe)
USE_TRACKING_SCORE_MASK = True
TRACKING_SCORE_THRESHOLD = 0.50

# Métricas QA anti-swaps (no enmascara; solo reporta)
RUN_SWAP_QA = True

# ---- Normalización de escala ----
NORMALIZE_COORDS = True
NORMALIZE_BY_BONE = True
BONE_A = "base_head"
BONE_B = "base_body"

# ---- Bodyparts ----
EXCLUDE_TAIL = True
ANTERIOR_BPS = ["nose", "upper_head"]
POSTERIOR_BPS = ["base_body", "base_tail"]

# ---- Kappa scan ----
KAPPA_SCAN_PREFIX = "kappa_scan"
KAPPA_SCAN_VALUES = np.logspace(3, 7, 5)      # [1e3, ~3e4, 1e5, ~3e5, 1e7]
KAPPA_DECREASE_FACTOR = 10
NUM_AR_ITERS_SCAN = 50
NUM_FULL_ITERS_SCAN = 200

# ---- Multi-seed fitting ----
MULTI_SEED_PREFIX = "multi_seed"
NUM_MODEL_FITS = 20
AR_ONLY_KAPPA = 1e6
FULL_MODEL_KAPPA = 1e4
NUM_AR_ITERS = 50
NUM_FULL_ITERS = 500

# ---- Merging similar syllables ----
# Edit this list to specify which syllables to merge before running.
# Each inner list groups syllables that will be collapsed into one.
# Leave empty ([]) to skip merging.
SYLLABLES_TO_MERGE: List[List[int]] = []
# Example: SYLLABLES_TO_MERGE = [[1, 3], [4, 5]]


# =============================
# UTILIDADES DE I/O / QA
# =============================
def collect_h5_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.h5") if p.is_file()])

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

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

# =============================
# SLEAP LOADER
# =============================
@dataclass
class SleapLoaded:
    tracks_TK2R: np.ndarray         # (T,K,2,R)
    conf_TKR: np.ndarray            # (T,K,R)  (point_scores)
    bodyparts: List[str]
    skeleton_edges: Optional[List[List[str]]]
    tracking_scores_RT: Optional[np.ndarray]   # (R,T)

def load_sleap_h5(h5_path: Path) -> SleapLoaded:
    with h5py.File(h5_path, "r") as f:
        tracks = np.array(f["/tracks"], dtype=np.float32)              # (R,2,K,T)
        occ = np.array(f["/track_occupancy"], dtype=np.uint8)          # (T,R)
        node_names = f["/node_names"][...]                             # (K,)

        point_scores = None
        if "/point_scores" in f:
            point_scores = np.array(f["/point_scores"], dtype=np.float32)  # (R,K,T)

        tracking_scores = None
        if "/tracking_scores" in f:
            tracking_scores = np.array(f["/tracking_scores"], dtype=np.float32)  # (R,T)

        skeleton_edges = None
        if "/edge_names" in f:
            edge_names = f["/edge_names"][...]
            sk = []
            for a, b in edge_names:
                aa = a.decode("utf-8") if isinstance(a, (bytes, np.bytes_)) else str(a)
                bb = b.decode("utf-8") if isinstance(b, (bytes, np.bytes_)) else str(b)
                sk.append([aa, bb])
            skeleton_edges = sk

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
        raise ValueError(f"{h5_path}: tracks.ndim={tracks.ndim}, esperado 4")
    R, D, K, T = tracks.shape
    if D != 2:
        raise ValueError(f"{h5_path}: D={D}, esperado 2 (x,y)")
    if len(bodyparts) != K:
        raise ValueError(f"{h5_path}: len(node_names)={len(bodyparts)} != K={K}")
    if occ.shape != (T, R):
        raise ValueError(f"{h5_path}: occupancy shape {occ.shape} != (T,R)=({T},{R})")

    tracks_TK2R = np.transpose(tracks, (3, 2, 1, 0))  # (T,K,2,R)

    if point_scores is not None:
        if point_scores.shape != (R, K, T):
            raise ValueError(f"{h5_path}: point_scores {point_scores.shape} esperado (R,K,T)=({R},{K},{T})")
        conf_TKR = np.transpose(point_scores, (2, 1, 0))  # (T,K,R)
    else:
        conf_TKR = np.ones((T, K, R), dtype=np.float32)

    for r in range(R):
        missing = (occ[:, r] == 0)
        if np.any(missing):
            tracks_TK2R[missing, :, :, r] = np.nan
            conf_TKR[missing, :, r] = 0.0

    invalid = np.isnan(tracks_TK2R[:, :, 0, :]) | np.isnan(tracks_TK2R[:, :, 1, :])
    conf_TKR[invalid] = 0.0

    return SleapLoaded(
        tracks_TK2R=tracks_TK2R,
        conf_TKR=conf_TKR,
        bodyparts=bodyparts,
        skeleton_edges=skeleton_edges,
        tracking_scores_RT=tracking_scores
    )


# =============================
# QA ANTI-SWAPS
# =============================
def swap_qa_report(tracks_TK2R: np.ndarray) -> Dict[str, float]:
    T, K, _, R = tracks_TK2R.shape
    cent = np.full((T, 2, R), np.nan, dtype=np.float32)
    for r in range(R):
        xy = tracks_TK2R[:, :, :, r]  # (T,K,2)
        valid = np.isfinite(xy).all(axis=2)
        for t in range(T):
            v = valid[t]
            if np.any(v):
                cent[t, :, r] = np.nanmean(xy[t, v, :], axis=0)

    report = {}
    for r in range(R):
        c = cent[:, :, r]
        dc = np.linalg.norm(np.diff(c, axis=0), axis=1)
        mad = robust_mad(dc)
        med = np.nanmedian(dc)
        thr = med + 10.0 * mad
        jumps = np.nansum(dc > thr)
        report[f"track{r+1}_speed_median"] = float(med)
        report[f"track{r+1}_speed_mad"] = float(mad)
        report[f"track{r+1}_jumps_10MAD"] = float(jumps)

    if R >= 2:
        d12 = np.linalg.norm(cent[:, :, 0] - cent[:, :, 1], axis=1)
        report["intertrack_dist_median"] = float(np.nanmedian(d12))
        report["intertrack_dist_p05"] = float(np.nanquantile(d12[np.isfinite(d12)], 0.05)) if np.any(np.isfinite(d12)) else float("nan")

    return report


# =============================
# NORMALIZACIÓN DE ESCALA
# =============================
def normalize_by_bone_length(
    coordinates_dict: Dict[str, np.ndarray],
    bodyparts: List[str],
    a: str,
    b: str
) -> Tuple[Dict[str, np.ndarray], float]:
    if a not in bodyparts or b not in bodyparts:
        raise ValueError(
            f"NORMALIZE_BY_BONE requires '{a}' and '{b}' in bodyparts.\n"
            f"Available: {bodyparts}"
        )
    ia, ib = bodyparts.index(a), bodyparts.index(b)

    all_lengths: List[np.ndarray] = []
    for c in coordinates_dict.values():
        pa = c[:, ia, :]
        pb = c[:, ib, :]
        valid = np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1)
        if np.any(valid):
            all_lengths.append(np.linalg.norm(pa[valid] - pb[valid], axis=1))

    if not all_lengths:
        raise ValueError(
            f"Could not compute bone length for '{a}'→'{b}': "
            "no valid frames found across all recordings."
        )

    scale = float(np.median(np.concatenate(all_lengths))) + 1e-12
    print(f"[NORM] Global median bone length ({a}→{b}): {scale:.4f} px  →  dividing all coords by this value.")

    out: Dict[str, np.ndarray] = {}
    for rec, c in coordinates_dict.items():
        c = c.copy()
        T, K, _ = c.shape
        centroid = np.full((T, 2), np.nan, dtype=c.dtype)
        for t in range(T):
            valid_kp = np.isfinite(c[t]).all(axis=1)
            if np.any(valid_kp):
                centroid[t] = np.nanmean(c[t, valid_kp, :], axis=0)
        c = c - centroid[:, np.newaxis, :]
        c = c / scale
        out[rec] = c

    return out, scale


# =============================
# SUBSAMPLING
# =============================
def subsample_files(h5_files: List[Path]) -> List[Path]:
    rng = np.random.default_rng(RANDOM_SEED)
    strata = {"S2": [], "S3": []}
    for p in h5_files:
        sid = extract_session_id_strict(p.stem)
        strata[sid.split("_")[1]].append(p)

    n_use = min(N_RANDOM_VIDEOS, len(h5_files))
    n2 = n_use // 2
    n3 = n_use - n2

    out = []
    for key, n in [("S2", n2), ("S3", n3)]:
        pool = strata[key]
        if not pool:
            continue
        n = min(n, len(pool))
        idx = rng.choice(len(pool), size=n, replace=False)
        out.extend([pool[i] for i in idx])
    return sorted(out)


# =============================
# PCA VARIANCE PLOT + USER INPUT
# =============================
def plot_pca_variance_and_ask(pca, qa_dir: Path, max_components: int = 15) -> int:
    """
    Plots individual and cumulative explained variance from the fitted PCA,
    saves the figure to qa_dir/pca_variance.png, and asks the user
    interactively how many latent dimensions to use.
    """
    # Extract explained variance from sklearn-style or array-style PCA object
    if hasattr(pca, "explained_variance_ratio_"):
        evr = np.array(pca.explained_variance_ratio_)
    elif hasattr(pca, "components_") and hasattr(pca, "singular_values_"):
        # Reconstruct from singular values if ratio not stored
        sv = np.array(pca.singular_values_)
        evr = sv**2 / np.sum(sv**2)
    else:
        # Fallback: try common attribute names used by kpms / jax-moseq
        for attr in ("variance_explained", "explained_variance", "pca_variance"):
            if hasattr(pca, attr):
                raw = np.array(getattr(pca, attr))
                evr = raw / raw.sum() if raw.sum() > 1.0 else raw
                break
        else:
            print(
                "[PCA PLOT] Could not extract explained variance from PCA object. "
                "Skipping plot — please enter the number of PCs manually."
            )
            return _ask_n_pcs(max_allowed=50)

    n_show = min(max_components, len(evr))
    evr_show = evr[:n_show]
    cumvar = np.cumsum(evr_show)
    components = np.arange(1, n_show + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Individual bars
    ax.bar(components, evr_show, color="#5a9e5a", alpha=0.85,
           label="Individual Explained Variance")

    # Cumulative line
    ax.plot(components, cumvar, color="red", marker="o", linewidth=2,
            label="Cumulative Explained Variance")

    # Percentage labels
    for i, (ind, cum) in enumerate(zip(evr_show, cumvar)):
        ax.annotate(f"{cum*100:.0f}%",
                    xy=(components[i], cum),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color="red")
        ax.annotate(f"{ind*100:.0f}%",
                    xy=(components[i], ind / 2),
                    ha="center", va="center", fontsize=7.5, color="white",
                    fontweight="bold")

    ax.set_xlabel("Principal Components", fontsize=11)
    ax.set_ylabel("Explained Variance", fontsize=11)
    ax.set_title("Explained Variance by Different Principal Components", fontsize=13)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(components)
    ax.legend(loc="center right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = qa_dir / "pca_variance.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"\n[PCA PLOT] Variance plot saved to: {plot_path.resolve()}")

    # Try to open the image for the user (best-effort, non-blocking)
    try:
        import subprocess, sys
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(plot_path)])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", str(plot_path)])
    except Exception:
        pass

    return _ask_n_pcs(max_allowed=n_show)


def _ask_n_pcs(max_allowed: int) -> int:
    """Blocks until the user enters a valid integer number of PCs."""
    print("\n" + "="*55)
    print("  PCA — SELECT NUMBER OF LATENT DIMENSIONS (PCs)")
    print("="*55)
    print(f"  Review the variance plot before answering.")
    print(f"  Valid range: 1 – {max_allowed}")
    print("="*55)
    while True:
        try:
            raw = input("  >> Enter number of principal components to use: ").strip()
            n = int(raw)
            if 1 <= n <= max_allowed:
                print(f"\n[PCA] Using {n} principal components.\n")
                return n
            else:
                print(f"  [!] Please enter a value between 1 and {max_allowed}.")
        except ValueError:
            print("  [!] Invalid input — please enter a whole number.")


# =============================
# PIPELINE PRINCIPAL
# =============================
def main():
    ensure_dir(PROJECT_DIR)
    qa_dir = PROJECT_DIR / "QA_AUDIT"
    ensure_dir(qa_dir)

    # =============================
    # 1) CARGA DE DATOS
    # =============================
    h5_files = collect_h5_files(DATA_ROOT)
    if not h5_files:
        raise FileNotFoundError(f"No encontré .h5 en {DATA_ROOT.resolve()}")

    h5_files = subsample_files(h5_files)

    save_json(qa_dir / "used_h5_files.json", [str(p.resolve()) for p in h5_files])

    print(f"\n[DATA] Usando {len(h5_files)} vídeos (.h5):")
    for p in h5_files:
        print("  -", p.name)
    print()

    coordinates_dict: Dict[str, np.ndarray] = {}
    confidences_dict: Dict[str, np.ndarray] = {}
    recording_names: List[str] = []
    bodyparts_ref: Optional[List[str]] = None
    skeleton_ref: Optional[List[List[str]]] = None

    male_track1_sessions = []
    male_track2_sessions = []
    per_recording_metrics = []

    for h5_path in h5_files:
        loaded = load_sleap_h5(h5_path)
        coords_TK2R = loaded.tracks_TK2R
        conf_TKR = loaded.conf_TKR
        bodyparts = loaded.bodyparts
        sk_edges = loaded.skeleton_edges
        tracking_scores = loaded.tracking_scores_RT

        T, K, _, R = coords_TK2R.shape

        if bodyparts_ref is None:
            bodyparts_ref = bodyparts
            skeleton_ref = sk_edges
        else:
            if bodyparts != bodyparts_ref:
                raise ValueError(
                    f"Bodyparts distintos entre archivos.\n"
                    f"Primero: {bodyparts_ref}\n"
                    f"Ahora:   {bodyparts}\n"
                    f"Archivo: {h5_path}"
                )

        rep = swap_qa_report(coords_TK2R) if RUN_SWAP_QA else {}

        male_r, sid = male_track_index_from_stem_strict(h5_path.stem)
        if not (0 <= male_r < R):
            raise ValueError(f"{h5_path.name}: male_r={male_r} fuera de rango para R={R}")

        if USE_TRACKING_SCORE_MASK and tracking_scores is not None:
            ts = tracking_scores[male_r]
            low = ts < TRACKING_SCORE_THRESHOLD
            if np.any(low):
                coords_TK2R[low, :, :, male_r] = np.nan
                conf_TKR[low, :, male_r] = 0.0

        print(f"[MALE SELECT] {h5_path.name} → SID={sid} → Track {male_r + 1}")

        if male_r == 0:
            male_track1_sessions.append(sid)
        else:
            male_track2_sessions.append(sid)

        rec = f"{sid}__male_track{male_r + 1}"
        coords_male = coords_TK2R[:, :, :, male_r]
        conf_male = conf_TKR[:, :, male_r]

        valid_xy = np.isfinite(coords_male).all(axis=2)
        mask_frac = float(np.mean(valid_xy)) if valid_xy.size else 0.0
        finite = coords_male[np.isfinite(coords_male)]
        coord_min = float(np.min(finite)) if finite.size else float("nan")
        coord_max = float(np.max(finite)) if finite.size else float("nan")

        anterior_ok = np.nan
        posterior_ok = np.nan
        if bodyparts_ref is not None:
            bp = bodyparts_ref
            if all(x in bp for x in ANTERIOR_BPS):
                idxs = [bp.index(x) for x in ANTERIOR_BPS]
                anterior_ok = float(np.mean(np.isfinite(coords_male[:, idxs, :]).all(axis=2)))
            if all(x in bp for x in POSTERIOR_BPS):
                idxs = [bp.index(x) for x in POSTERIOR_BPS]
                posterior_ok = float(np.mean(np.isfinite(coords_male[:, idxs, :]).all(axis=2)))

        m = {
            "recording": rec,
            "file": h5_path.name,
            "T_frames": T,
            "K_keypoints": K,
            "mask_frac_xy": mask_frac,
            "coord_min": coord_min,
            "coord_max": coord_max,
            "anterior_valid_frac": anterior_ok,
            "posterior_valid_frac": posterior_ok,
        }
        m.update({f"swapQA_{k}": v for k, v in rep.items()})
        per_recording_metrics.append(m)

        coordinates_dict[rec] = coords_male
        confidences_dict[rec] = conf_male
        recording_names.append(rec)

    save_csv(qa_dir / "per_recording_metrics.csv", per_recording_metrics)

    detected_track2 = set(male_track2_sessions)
    if not detected_track2.issubset(MALE_TRACK2_IDS):
        extras = sorted(detected_track2 - MALE_TRACK2_IDS)
        raise ValueError(
            "\nERROR: sesiones Track2 detectadas que NO están en MALE_TRACK2_IDS.\n"
            f"Extras detectados: {extras}\n"
            f"MALE_TRACK2_IDS:   {sorted(MALE_TRACK2_IDS)}"
        )

    print("\n=== RESUMEN FINAL ===")
    print("Macho = Track 1 en:", sorted(male_track1_sessions))
    print("Macho = Track 2 en:", sorted(male_track2_sessions))
    print("=====================\n")

    # =============================
    # 2) NORMALIZACIÓN
    # =============================
    if NORMALIZE_COORDS:
        if not NORMALIZE_BY_BONE:
            raise ValueError(
                "NORMALIZE_COORDS=True but NORMALIZE_BY_BONE=False. "
                "Please set NORMALIZE_BY_BONE=True."
            )
        coordinates_dict, scale = normalize_by_bone_length(
            coordinates_dict, bodyparts_ref, BONE_A, BONE_B
        )
        norm_info = {
            "enabled": True,
            "method": "per_recording_centering_then_bone_median_scale",
            "bone": [BONE_A, BONE_B],
            "scale_px": scale,
        }
        save_json(qa_dir / "normalization.json", norm_info)
        print(f"[NORM] Coordenadas normalizadas. Info: {norm_info}")

    # =============================
    # 3) CONFIG + SKELETON
    # =============================
    bodyparts = bodyparts_ref

    skeleton = skeleton_ref if skeleton_ref is not None else [
        ["nose", "upper_head"],
        ["upper_head", "base_head"],
        ["base_head", "upper_body"],
        ["upper_body", "base_body"],
        ["base_body", "base_tail"],
        ["base_head", "L_ear"],
        ["base_head", "R_ear"],
        ["base_body", "L_hip"],
        ["base_body", "R_hip"],
        ["upper_body", "L_sh"],
        ["upper_body", "R_sh"],
    ]

    if EXCLUDE_TAIL:
        use_bodyparts = [bp for bp in bodyparts if "tail" not in bp.lower()]
        if "base_tail" in bodyparts and "base_tail" not in use_bodyparts:
            use_bodyparts.append("base_tail")
    else:
        use_bodyparts = bodyparts

    anterior_bodyparts = [bp for bp in ANTERIOR_BPS if bp in bodyparts]
    posterior_bodyparts = [bp for bp in POSTERIOR_BPS if bp in bodyparts]

    bp_set = set(bodyparts)
    for a, b in skeleton:
        if a not in bp_set or b not in bp_set:
            raise ValueError(f"Skeleton edge ({a},{b}) not in bodyparts")
    for x in anterior_bodyparts + posterior_bodyparts:
        if x not in bp_set:
            raise ValueError(f"Bodypart {x} not in bodyparts")

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

    # =============================
    # 4) OUTLIER REMOVAL + FORMAT
    # =============================
    coordinates, confidences = kpms.outlier_removal(
        coordinates_dict,
        confidences_dict,
        str(PROJECT_DIR),
        overwrite=False,
        **cfg
    )
    data, metadata = kpms.format_data(coordinates, confidences, **cfg)

    from jax_moseq.utils.debugging import convert_data_precision
    data = convert_data_precision(data)

    # =============================
    # 5) ESTIMAR sigmasq_loc
    # =============================
    kpms.update_config(
        str(PROJECT_DIR),
        sigmasq_loc=kpms.estimate_sigmasq_loc(
            data["Y"], data["mask"], filter_size=cfg["fps"]
        )
    )
    cfg = kpms.load_config(str(PROJECT_DIR))

    # =============================
    # 6) PCA — fit, plot variance, ask user for n_components
    # =============================
    # Fit PCA with the maximum number of components first so the
    # variance plot shows the full spectrum before the user decides.
    pca = kpms.fit_pca(data["Y"], data["mask"], **cfg)
    try:
        kpms.save_pca(pca, str(PROJECT_DIR))
    except Exception:
        pass

    # Show cumulative variance plot and block until user picks n_components
    n_pcs = plot_pca_variance_and_ask(pca, qa_dir)

    # Re-fit PCA with the chosen number of components and update config
    pca = kpms.fit_pca(data["Y"], data["mask"], num_pcs=n_pcs, **cfg)
    try:
        kpms.save_pca(pca, str(PROJECT_DIR))
    except Exception:
        pass

    kpms.update_config(str(PROJECT_DIR), num_pcs=n_pcs)
    cfg = kpms.load_config(str(PROJECT_DIR))

    save_json(qa_dir / "pca_n_components.json", {"num_pcs": n_pcs})
    print(f"[PCA] Config updated with num_pcs={n_pcs}.")

    # =============================
    # 7) KAPPA SCAN
    # =============================
    print("\n" + "="*50)
    print("STAGE 1: KAPPA SCAN")
    print("="*50)

    for kappa in KAPPA_SCAN_VALUES:
        print(f"\n[KAPPA SCAN] Fitting model with kappa={kappa:.2e}")
        model_name = f"{KAPPA_SCAN_PREFIX}-{kappa:.2e}"
        model = kpms.init_model(data, pca=pca, **cfg)

        # AR-only stage
        model = kpms.update_hypparams(model, kappa=kappa)
        model = kpms.fit_model(
            model, data, metadata, str(PROJECT_DIR), model_name,
            ar_only=True,
            num_iters=NUM_AR_ITERS_SCAN,
            save_every_n_iters=25
        )[0]

        # Full model stage with reduced kappa
        model = kpms.update_hypparams(model, kappa=kappa / KAPPA_DECREASE_FACTOR)
        kpms.fit_model(
            model, data, metadata, str(PROJECT_DIR), model_name,
            ar_only=False,
            start_iter=NUM_AR_ITERS_SCAN,
            num_iters=NUM_FULL_ITERS_SCAN,
            save_every_n_iters=25
        )

    kpms.plot_kappa_scan(KAPPA_SCAN_VALUES, str(PROJECT_DIR), KAPPA_SCAN_PREFIX)
    print("\n[KAPPA SCAN] Done. Review the kappa scan plot to choose the best kappa.")

    # =============================
    # 8) MULTI-SEED FITTING
    # =============================
    print("\n" + "="*50)
    print("STAGE 2: MULTI-SEED MODEL FITTING")
    print("="*50)

    for restart in range(NUM_MODEL_FITS):
        print(f"\n[MULTI-SEED] Fitting model {restart} / {NUM_MODEL_FITS - 1}")
        model_name = f"{MULTI_SEED_PREFIX}-{restart}"

        model = kpms.init_model(
            data, pca=pca, **cfg, seed=jax.random.PRNGKey(restart)
        )

        # AR-only stage
        model = kpms.update_hypparams(model, kappa=AR_ONLY_KAPPA)
        model = kpms.fit_model(
            model, data, metadata, str(PROJECT_DIR), model_name,
            ar_only=True,
            num_iters=NUM_AR_ITERS
        )[0]

        # Full model stage
        model = kpms.update_hypparams(model, kappa=FULL_MODEL_KAPPA)
        kpms.fit_model(
            model, data, metadata, str(PROJECT_DIR), model_name,
            ar_only=False,
            start_iter=NUM_AR_ITERS,
            num_iters=NUM_FULL_ITERS
        )

        kpms.reindex_syllables_in_checkpoint(str(PROJECT_DIR), model_name)
        model, data, metadata, current_iter = kpms.load_checkpoint(
            str(PROJECT_DIR), model_name
        )
        results = kpms.extract_results(model, metadata, str(PROJECT_DIR), model_name)

    # =============================
    # 9) MODEL SELECTION (EML)
    # =============================
    print("\n" + "="*50)
    print("STAGE 3: MODEL SELECTION VIA EML SCORE")
    print("="*50)

    model_names = [f"{MULTI_SEED_PREFIX}-{i}" for i in range(NUM_MODEL_FITS)]

    # Pairwise confusion matrix between first two models as a sanity check
    print("[MODEL SELECTION] Plotting confusion matrix for models 0 vs 1 ...")
    results_0 = kpms.load_results(str(PROJECT_DIR), model_names[0])
    results_1 = kpms.load_results(str(PROJECT_DIR), model_names[1])
    kpms.plot_confusion_matrix(results_0, results_1)

    # EML scores
    print("[MODEL SELECTION] Computing EML scores ...")
    eml_scores, eml_std_errs = kpms.expected_marginal_likelihoods(
        str(PROJECT_DIR), model_names
    )
    best_model_name = model_names[np.argmax(eml_scores)]
    print(f"[MODEL SELECTION] Best model: {best_model_name}")

    kpms.plot_eml_scores(eml_scores, eml_std_errs, model_names)

    save_json(qa_dir / "model_selection.json", {
        "best_model": best_model_name,
        "model_names": model_names,
        "eml_scores": list(float(s) for s in eml_scores),
        "eml_std_errs": list(float(s) for s in eml_std_errs),
    })

    # =============================
    # 10) RESULTADOS FINALES + PLOTS
    # =============================
    print("\n" + "="*50)
    print("STAGE 4: EXTRACTING RESULTS FOR BEST MODEL")
    print("="*50)

    model, data, metadata, _ = kpms.load_checkpoint(str(PROJECT_DIR), best_model_name)
    results = kpms.extract_results(model, metadata, str(PROJECT_DIR), best_model_name)
    kpms.save_results_as_csv(results, str(PROJECT_DIR), best_model_name)

    results = kpms.load_results(str(PROJECT_DIR), best_model_name)
    kpms.generate_trajectory_plots(
        coordinates, results, str(PROJECT_DIR), best_model_name, **cfg
    )

    # =============================
    # 11) MERGING SIMILAR SYLLABLES (opcional)
    # =============================
    if SYLLABLES_TO_MERGE:
        print("\n" + "="*50)
        print("STAGE 5: MERGING SIMILAR SYLLABLES")
        print("="*50)
        print(f"[MERGE] Groups to merge: {SYLLABLES_TO_MERGE}")

        results_path = os.path.join(
            str(PROJECT_DIR), best_model_name, "results.h5"
        )
        results_raw = kpms.load_hdf5(results_path)

        syllable_mapping = kpms.generate_syllable_mapping(
            results_raw, SYLLABLES_TO_MERGE
        )
        new_results = kpms.apply_syllable_mapping(results_raw, syllable_mapping)

        new_results_path = os.path.join(
            str(PROJECT_DIR), best_model_name, "results_merged.h5"
        )
        kpms.save_hdf5(new_results_path, new_results)
        print(f"[MERGE] Merged results saved to: {new_results_path}")

        output_dir = os.path.join(
            str(PROJECT_DIR), best_model_name, "trajectory_plots_merged"
        )
        kpms.generate_trajectory_plots(
            coordinates, new_results, output_dir=output_dir, **cfg
        )
        print(f"[MERGE] Trajectory plots saved to: {output_dir}")
    else:
        print("\n[MERGE] SYLLABLES_TO_MERGE is empty — skipping merge step.")
        print("        Set SYLLABLES_TO_MERGE in the config section to enable this.")

    print("\nDONE. Best model:", (PROJECT_DIR / best_model_name).resolve())
    return best_model_name


if __name__ == "__main__":
    main()
