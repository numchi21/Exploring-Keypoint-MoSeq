"""
Multi-seed model training pipeline.

Prerequisites:
  - Run 01_explore.py first to choose num_pcs and kappa values.
  - Update FULL_MODEL_KAPPA (and AR_ONLY_KAPPA) in config.py if needed.

What this script does:
  1. Loads and preprocesses data.
  2. Formats data for kpms.
  3. Fits NUM_MODEL_FITS models with different random seeds.
  4. Saves results (results.h5 + results.csv) for every model.

After this script completes, run:
  python 03_select_model.py
"""
from __future__ import annotations

import numpy as np
import jax
import keypoint_moseq as kpms
from pathlib import Path

from config import (
    PROJECT_DIR, DATA_ROOT, FPS,
    ANTERIOR_BPS, POSTERIOR_BPS, EXCLUDE_TAIL,
    NORMALIZE_COORDS, NORMALIZE_BY_BONE, BONE_A, BONE_B,
    RUN_SWAP_QA, USE_TRACKING_SCORE_MASK, TRACKING_SCORE_THRESHOLD,
    MULTI_SEED_PREFIX, NUM_MODEL_FITS,
    AR_ONLY_KAPPA, FULL_MODEL_KAPPA,
    NUM_AR_ITERS, NUM_FULL_ITERS,
    collect_h5_files, subsample_files, ensure_dir, save_json, save_csv,
    load_sleap_h5, male_track_index_from_stem_strict, swap_qa_report,
    normalize_by_bone_length, MALE_TRACK2_IDS,
)


def load_and_preprocess():
    """Load raw data, normalise, build kpms data/metadata. Returns
    (data, metadata, coordinates, cfg, pca) ready for model fitting."""

    ensure_dir(PROJECT_DIR)
    qa_dir = PROJECT_DIR / "QA_AUDIT"
    ensure_dir(qa_dir)

    h5_files = collect_h5_files(DATA_ROOT)
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {DATA_ROOT.resolve()}")
    h5_files = subsample_files(h5_files)

    print(f"\n[DATA] Using {len(h5_files)} videos:")
    for p in h5_files:
        print("  -", p.name)

    coordinates_dict, confidences_dict = {}, {}
    bodyparts_ref, skeleton_ref = None, None
    male_track1_sessions, male_track2_sessions = [], []
    per_recording_metrics = []

    for h5_path in h5_files:
        loaded   = load_sleap_h5(h5_path)
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

        rec         = f"{sid}__male_track{male_r + 1}"
        coords_male = loaded.tracks_TK2R[:, :, :, male_r]
        conf_male   = loaded.conf_TKR[:, :, male_r]

        valid_xy = np.isfinite(coords_male).all(axis=2)
        finite   = coords_male[np.isfinite(coords_male)]
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

    print("\n=== TRACK SUMMARY ===")
    print("Track 1 (male):", sorted(male_track1_sessions))
    print("Track 2 (male):", sorted(male_track2_sessions))
    print("====================\n")

    # Normalisation
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

    # kpms config
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
    anterior_bodyparts  = [bp for bp in ANTERIOR_BPS  if bp in bodyparts]
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

    # Load the PCA saved by 01_explore.py (num_pcs already in config.yml)
    try:
        pca = kpms.load_pca(str(PROJECT_DIR))
        print("[PCA] Loaded saved PCA from project directory.")
    except Exception:
        print("[PCA] No saved PCA found — fitting now with config num_pcs ...")
        pca = kpms.fit_pca(data["Y"], data["mask"], **cfg)
        try:
            kpms.save_pca(pca, str(PROJECT_DIR))
        except Exception:
            pass

    return data, metadata, coordinates, cfg, pca


# =============================================================================
# MAIN
# =============================================================================
def main():
    data, metadata, coordinates, cfg, pca = load_and_preprocess()

    print("\n" + "="*55)
    print(f"  MULTI-SEED TRAINING  ({NUM_MODEL_FITS} models)")
    print("="*55)
    print(f"  AR-only kappa  : {AR_ONLY_KAPPA:.1e}  ({NUM_AR_ITERS} iters)")
    print(f"  Full model kappa: {FULL_MODEL_KAPPA:.1e}  ({NUM_FULL_ITERS} iters)")
    print("="*55 + "\n")

    for restart in range(NUM_MODEL_FITS):
        print(f"\n[TRAIN] Model {restart + 1}/{NUM_MODEL_FITS}  (seed={restart})")
        model_name = f"{MULTI_SEED_PREFIX}-{restart}"

        model = kpms.init_model(
            data, pca=pca, **cfg, seed=jax.random.PRNGKey(restart)
        )

        # Stage 1: AR-only
        model = kpms.update_hypparams(model, kappa=AR_ONLY_KAPPA)
        model = kpms.fit_model(
            model, data, metadata, str(PROJECT_DIR), model_name,
            ar_only=True, num_iters=NUM_AR_ITERS
        )[0]

        # Stage 2: full model
        model = kpms.update_hypparams(model, kappa=FULL_MODEL_KAPPA)
        kpms.fit_model(
            model, data, metadata, str(PROJECT_DIR), model_name,
            ar_only=False, start_iter=NUM_AR_ITERS, num_iters=NUM_FULL_ITERS
        )

        kpms.reindex_syllables_in_checkpoint(str(PROJECT_DIR), model_name)
        model, data, metadata, _ = kpms.load_checkpoint(str(PROJECT_DIR), model_name)
        results = kpms.extract_results(model, metadata, str(PROJECT_DIR), model_name)
        kpms.save_results_as_csv(results, str(PROJECT_DIR), model_name)
        print(f"[TRAIN] Model {model_name} saved.")

    print(
        "\n[TRAIN] All models trained.\n"
        "  → Run:  python 03_select_model.py\n"
    )


if __name__ == "__main__":
    main()
