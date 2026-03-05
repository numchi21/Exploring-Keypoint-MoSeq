"""
Reads best_model from QA_AUDIT/model_selection.json (written by
03_select_model.py).  You can also override it by setting
BEST_MODEL_NAME below.

Outputs (all saved inside PROJECT_DIR/<model_name>/):
  trajectory_plots/         — median pose trajectory per syllable
  grid_movies/              — video clips of each syllable
  similarity_dendrogram.pdf — hierarchical clustering of syllables

After reviewing these outputs you may want to merge similar syllables.
  → Run:  python 05_merge_syllables.py
"""
from __future__ import annotations

import os
import json
import numpy as np
import keypoint_moseq as kpms
from pathlib import Path

from config import (
    PROJECT_DIR, DATA_ROOT, FPS,
    ANTERIOR_BPS, POSTERIOR_BPS, EXCLUDE_TAIL,
    NORMALIZE_COORDS, NORMALIZE_BY_BONE, BONE_A, BONE_B,
    RUN_SWAP_QA, USE_TRACKING_SCORE_MASK, TRACKING_SCORE_THRESHOLD,
    collect_h5_files, subsample_files, ensure_dir, load_json,
    load_sleap_h5, male_track_index_from_stem_strict,
    normalize_by_bone_length, MALE_TRACK2_IDS,
)

# Override this to skip reading from model_selection.json
BEST_MODEL_NAME: str | None = None


def load_coordinates_for_viz():
    """Re-load and normalise coordinates (needed by kpms visualisation calls)."""
    h5_files = collect_h5_files(DATA_ROOT)
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {DATA_ROOT.resolve()}")
    h5_files = subsample_files(h5_files)

    coordinates_dict, confidences_dict = {}, {}
    bodyparts_ref, skeleton_ref = None, None

    for h5_path in h5_files:
        loaded   = load_sleap_h5(h5_path)
        if bodyparts_ref is None:
            bodyparts_ref = loaded.bodyparts
            skeleton_ref  = loaded.skeleton_edges

        male_r, sid = male_track_index_from_stem_strict(h5_path.stem)

        if USE_TRACKING_SCORE_MASK and loaded.tracking_scores_RT is not None:
            ts  = loaded.tracking_scores_RT[male_r]
            low = ts < TRACKING_SCORE_THRESHOLD
            if np.any(low):
                loaded.tracks_TK2R[low, :, :, male_r] = np.nan
                loaded.conf_TKR[low, :, male_r]        = 0.0

        rec = f"{sid}__male_track{male_r + 1}"
        coordinates_dict[rec] = loaded.tracks_TK2R[:, :, :, male_r]
        confidences_dict[rec] = loaded.conf_TKR[:, :, male_r]

    if NORMALIZE_COORDS and NORMALIZE_BY_BONE:
        coordinates_dict, _ = normalize_by_bone_length(
            coordinates_dict, bodyparts_ref, BONE_A, BONE_B
        )

    # Run outlier removal to get the cleaned coordinates kpms expects
    cfg = kpms.load_config(str(PROJECT_DIR))
    coordinates, _ = kpms.outlier_removal(
        coordinates_dict, confidences_dict, str(PROJECT_DIR),
        overwrite=False, **cfg
    )
    return coordinates, cfg


# =============================================================================
# MAIN
# =============================================================================
def main():
    # ------------------------------------------------------------------
    # Resolve best model name
    # ------------------------------------------------------------------
    model_name = BEST_MODEL_NAME
    if model_name is None:
        selection_path = PROJECT_DIR / "QA_AUDIT" / "model_selection.json"
        if not selection_path.exists():
            raise FileNotFoundError(
                "model_selection.json not found.\n"
                "Run 03_select_model.py first, or set BEST_MODEL_NAME manually."
            )
        model_name = load_json(selection_path)["best_model"]

    print(f"\n[VIZ] Generating visualisations for model: {model_name}\n")

    # ------------------------------------------------------------------
    # Load coordinates + config
    # ------------------------------------------------------------------
    coordinates, cfg = load_coordinates_for_viz()

    # ------------------------------------------------------------------
    # Load results
    # ------------------------------------------------------------------
    results = kpms.load_results(str(PROJECT_DIR), model_name)

    # ------------------------------------------------------------------
    # Trajectory plots
    # ------------------------------------------------------------------
    print("[VIZ] Generating trajectory plots ...")
    kpms.generate_trajectory_plots(
        coordinates, results, str(PROJECT_DIR), model_name, **cfg
    )
    traj_dir = os.path.join(str(PROJECT_DIR), model_name, "trajectory_plots")
    print(f"[VIZ] Saved to: {traj_dir}")

    # ------------------------------------------------------------------
    # Grid movies
    # ------------------------------------------------------------------
    print("\n[VIZ] Generating grid movies ...")
    kpms.generate_grid_movies(
        results, str(PROJECT_DIR), model_name,
        coordinates=coordinates, **cfg
    )
    grid_dir = os.path.join(str(PROJECT_DIR), model_name, "grid_movies")
    print(f"[VIZ] Saved to: {grid_dir}")

    # ------------------------------------------------------------------
    # Syllable similarity dendrogram
    # ------------------------------------------------------------------
    print("\n[VIZ] Generating syllable similarity dendrogram ...")
    kpms.plot_similarity_dendrogram(
        coordinates, results, str(PROJECT_DIR), model_name, **cfg
    )
    dendro_path = os.path.join(str(PROJECT_DIR), model_name, "similarity_dendrogram.pdf")
    print(f"[VIZ] Saved to: {dendro_path}")

    print(
        "\n[VIZ] All visualisations complete.\n"
        "  → Review trajectory plots, grid movies, and the dendrogram.\n"
        "  → If you want to merge similar syllables, edit SYLLABLES_TO_MERGE\n"
        "    in 05_merge_syllables.py and run it.\n"
    )


if __name__ == "__main__":
    main()
