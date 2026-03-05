"""
Run ONCE before training.

Steps:
  1. Load & preprocess data (same pipeline as training).
  2. Fit PCA on all components → show variance plot → ask user for num_pcs.
  3. Run a kappa scan over a log-spaced grid → plot results.

After running this script you will know:
  - How many PCs to use  (saved to QA_AUDIT/pca_n_components.json)
  - Which kappa to use   (inspect kappa_scan plot, then set FULL_MODEL_KAPPA
                          in config.py before running 02_train.py)
"""
from __future__ import annotations

import os
import numpy as np
import keypoint_moseq as kpms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    PROJECT_DIR, DATA_ROOT, FPS,
    ANTERIOR_BPS, POSTERIOR_BPS, EXCLUDE_TAIL,
    NORMALIZE_COORDS, NORMALIZE_BY_BONE, BONE_A, BONE_B,
    RUN_SWAP_QA, USE_TRACKING_SCORE_MASK, TRACKING_SCORE_THRESHOLD,
    collect_h5_files, subsample_files, ensure_dir, save_json,
    load_sleap_h5, male_track_index_from_stem_strict, swap_qa_report,
    normalize_by_bone_length, save_csv, MALE_TRACK2_IDS,
)

# =============================================================================
# KAPPA SCAN CONFIG  (edit here)
# =============================================================================
KAPPA_SCAN_PREFIX     = "kappa_scan"
KAPPA_SCAN_VALUES     = np.logspace(3, 7, 5)   # [1e3, ~3e4, 1e5, ~3e5, 1e7]
KAPPA_DECREASE_FACTOR = 10
NUM_AR_ITERS_SCAN     = 50
NUM_FULL_ITERS_SCAN   = 200


# =============================================================================
# PCA VARIANCE PLOT + INTERACTIVE n_pcs SELECTION
# =============================================================================
def plot_pca_variance_and_ask(pca, qa_dir: Path, max_components: int = 15) -> int:
    if hasattr(pca, "explained_variance_ratio_"):
        evr = np.array(pca.explained_variance_ratio_)
    elif hasattr(pca, "singular_values_"):
        sv  = np.array(pca.singular_values_)
        evr = sv**2 / np.sum(sv**2)
    else:
        for attr in ("variance_explained", "explained_variance", "pca_variance"):
            if hasattr(pca, attr):
                raw = np.array(getattr(pca, attr))
                evr = raw / raw.sum() if raw.sum() > 1.0 else raw
                break
        else:
            print("[PCA PLOT] Cannot extract variance — enter PCs manually.")
            return _ask_n_pcs(max_allowed=50)

    n_show    = min(max_components, len(evr))
    evr_show  = evr[:n_show]
    cumvar    = np.cumsum(evr_show)
    comps     = np.arange(1, n_show + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(comps, evr_show, color="#5a9e5a", alpha=0.85,
           label="Individual Explained Variance")
    ax.plot(comps, cumvar, color="red", marker="o", linewidth=2,
            label="Cumulative Explained Variance")

    for i, (ind, cum) in enumerate(zip(evr_show, cumvar)):
        ax.annotate(f"{cum*100:.0f}%", xy=(comps[i], cum),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color="red")
        ax.annotate(f"{ind*100:.0f}%", xy=(comps[i], ind / 2),
                    ha="center", va="center", fontsize=7.5,
                    color="white", fontweight="bold")

    ax.set_xlabel("Principal Components", fontsize=11)
    ax.set_ylabel("Explained Variance",   fontsize=11)
    ax.set_title("Explained Variance by Different Principal Components", fontsize=13)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(comps)
    ax.legend(loc="center right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = qa_dir / "pca_variance.png"
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)
    print(f"\n[PCA PLOT] Saved to: {plot_path.resolve()}")

    try:
        import subprocess, sys
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, str(plot_path)])
    except Exception:
        pass

    return _ask_n_pcs(max_allowed=n_show)


def _ask_n_pcs(max_allowed: int) -> int:
    print("\n" + "="*55)
    print("  PCA — SELECT NUMBER OF LATENT DIMENSIONS (PCs)")
    print("="*55)
    print(f"  Review pca_variance.png before answering.")
    print(f"  Valid range: 1 – {max_allowed}")
    print("="*55)
    while True:
        try:
            n = int(input("  >> Enter number of principal components: ").strip())
            if 1 <= n <= max_allowed:
                print(f"\n[PCA] Using {n} principal components.\n")
                return n
            print(f"  [!] Enter a value between 1 and {max_allowed}.")
        except ValueError:
            print("  [!] Invalid input — please enter a whole number.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ensure_dir(PROJECT_DIR)
    qa_dir = PROJECT_DIR / "QA_AUDIT"
    ensure_dir(qa_dir)

    # ------------------------------------------------------------------
    # 1) Load data (identical preprocessing to 02_train.py)
    # ------------------------------------------------------------------
    h5_files = collect_h5_files(DATA_ROOT)
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {DATA_ROOT.resolve()}")
    h5_files = subsample_files(h5_files)
    save_json(qa_dir / "used_h5_files.json", [str(p.resolve()) for p in h5_files])

    print(f"\n[DATA] Using {len(h5_files)} videos:")
    for p in h5_files:
        print("  -", p.name)

    coordinates_dict, confidences_dict = {}, {}
    bodyparts_ref, skeleton_ref = None, None
    male_track1_sessions, male_track2_sessions = [], []
    per_recording_metrics = []

    for h5_path in h5_files:
        loaded = load_sleap_h5(h5_path)
        T, K, _, R = loaded.tracks_TK2R.shape

        if bodyparts_ref is None:
            bodyparts_ref = loaded.bodyparts
            skeleton_ref  = loaded.skeleton_edges
        elif loaded.bodyparts != bodyparts_ref:
            raise ValueError(f"Bodypart mismatch in {h5_path}")

        rep      = swap_qa_report(loaded.tracks_TK2R) if RUN_SWAP_QA else {}
        male_r, sid = male_track_index_from_stem_strict(h5_path.stem)

        if USE_TRACKING_SCORE_MASK and loaded.tracking_scores_RT is not None:
            ts  = loaded.tracking_scores_RT[male_r]
            low = ts < TRACKING_SCORE_THRESHOLD
            if np.any(low):
                loaded.tracks_TK2R[low, :, :, male_r] = np.nan
                loaded.conf_TKR[low, :, male_r]        = 0.0

        print(f"[MALE SELECT] {h5_path.name} → SID={sid} → Track {male_r + 1}")
        (male_track2_sessions if male_r == 1 else male_track1_sessions).append(sid)

        rec          = f"{sid}__male_track{male_r + 1}"
        coords_male  = loaded.tracks_TK2R[:, :, :, male_r]
        conf_male    = loaded.conf_TKR[:, :, male_r]

        valid_xy  = np.isfinite(coords_male).all(axis=2)
        finite    = coords_male[np.isfinite(coords_male)]
        m = {
            "recording": rec, "file": h5_path.name,
            "T_frames": T, "K_keypoints": K,
            "mask_frac_xy": float(np.mean(valid_xy)) if valid_xy.size else 0.0,
            "coord_min": float(np.min(finite)) if finite.size else float("nan"),
            "coord_max": float(np.max(finite)) if finite.size else float("nan"),
        }
        m.update({f"swapQA_{k}": v for k, v in rep.items()})
        per_recording_metrics.append(m)
        coordinates_dict[rec] = coords_male
        confidences_dict[rec] = conf_male

    save_csv(qa_dir / "per_recording_metrics.csv", per_recording_metrics)

    detected_track2 = set(male_track2_sessions)
    if not detected_track2.issubset(MALE_TRACK2_IDS):
        raise ValueError(f"Unexpected Track2 sessions: {sorted(detected_track2 - MALE_TRACK2_IDS)}")

    # ------------------------------------------------------------------
    # 2) Normalisation
    # ------------------------------------------------------------------
    if NORMALIZE_COORDS:
        if not NORMALIZE_BY_BONE:
            raise ValueError("Set NORMALIZE_BY_BONE=True in config.py")
        coordinates_dict, scale = normalize_by_bone_length(
            coordinates_dict, bodyparts_ref, BONE_A, BONE_B
        )
        save_json(qa_dir / "normalization.json", {
            "method": "per_recording_centering_then_bone_median_scale",
            "bone": [BONE_A, BONE_B], "scale_px": scale,
        })

    # ------------------------------------------------------------------
    # 3) kpms config + outlier removal + format
    # ------------------------------------------------------------------
    bodyparts = bodyparts_ref
    skeleton  = skeleton_ref or [
        ["nose","upper_head"],["upper_head","base_head"],
        ["base_head","upper_body"],["upper_body","base_body"],
        ["base_body","base_tail"],["base_head","L_ear"],
        ["base_head","R_ear"],["base_body","L_hip"],
        ["base_body","R_hip"],["upper_body","L_sh"],["upper_body","R_sh"],
    ]
    use_bodyparts = (
        [bp for bp in bodyparts if "tail" not in bp.lower()] +
        (["base_tail"] if "base_tail" in bodyparts and
         "base_tail" not in [bp for bp in bodyparts if "tail" not in bp.lower()] else [])
        if EXCLUDE_TAIL else bodyparts
    )
    anterior_bodyparts = [bp for bp in ANTERIOR_BPS if bp in bodyparts]
    posterior_bodyparts = [bp for bp in POSTERIOR_BPS if bp in bodyparts]

    if not (PROJECT_DIR / "config.yml").exists():
        kpms.setup_project(str(PROJECT_DIR), overwrite=True)

    kpms.update_config(
        str(PROJECT_DIR), fps=FPS, bodyparts=bodyparts,
        use_bodyparts=use_bodyparts, skeleton=skeleton,
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

    from jax_moseq.utils.debugging import convert_data_precision
    data = convert_data_precision(data)

    kpms.update_config(
        str(PROJECT_DIR),
        sigmasq_loc=kpms.estimate_sigmasq_loc(
            data["Y"], data["mask"], filter_size=cfg["fps"]
        )
    )
    cfg = kpms.load_config(str(PROJECT_DIR))

    # ------------------------------------------------------------------
    # 4) PCA — full fit → variance plot → user picks n_pcs → re-fit
    # ------------------------------------------------------------------
    print("\n[PCA] Fitting PCA (full spectrum) ...")
    pca = kpms.fit_pca(data["Y"], data["mask"], **cfg)

    n_pcs = plot_pca_variance_and_ask(pca, qa_dir)

    print(f"[PCA] Re-fitting with num_pcs={n_pcs} ...")
    pca = kpms.fit_pca(data["Y"], data["mask"], num_pcs=n_pcs, **cfg)
    try:
        kpms.save_pca(pca, str(PROJECT_DIR))
    except Exception:
        pass
    kpms.update_config(str(PROJECT_DIR), num_pcs=n_pcs)
    save_json(qa_dir / "pca_n_components.json", {"num_pcs": n_pcs})
    print(f"[PCA] config.yml updated with num_pcs={n_pcs}")

    # ------------------------------------------------------------------
    # 5) Kappa scan
    # ------------------------------------------------------------------
    print("\n" + "="*55)
    print("  KAPPA SCAN")
    print("="*55)

    for kappa in KAPPA_SCAN_VALUES:
        print(f"\n[KAPPA SCAN] kappa={kappa:.2e}")
        model_name = f"{KAPPA_SCAN_PREFIX}-{kappa:.2e}"
        cfg        = kpms.load_config(str(PROJECT_DIR))
        model      = kpms.init_model(data, pca=pca, **cfg)

        model = kpms.update_hypparams(model, kappa=kappa)
        model = kpms.fit_model(
            model, data, metadata, str(PROJECT_DIR), model_name,
            ar_only=True, num_iters=NUM_AR_ITERS_SCAN, save_every_n_iters=25
        )[0]

        model = kpms.update_hypparams(model, kappa=kappa / KAPPA_DECREASE_FACTOR)
        kpms.fit_model(
            model, data, metadata, str(PROJECT_DIR), model_name,
            ar_only=False, start_iter=NUM_AR_ITERS_SCAN,
            num_iters=NUM_FULL_ITERS_SCAN, save_every_n_iters=25
        )

    kpms.plot_kappa_scan(KAPPA_SCAN_VALUES, str(PROJECT_DIR), KAPPA_SCAN_PREFIX)
    print(
        "\n[KAPPA SCAN] Done.\n"
        "  → Inspect the kappa scan plot.\n"
        "  → Set FULL_MODEL_KAPPA (and AR_ONLY_KAPPA if needed) in config.py.\n"
        "  → Then run:  python 02_train.py\n"
    )


if __name__ == "__main__":
    main()
