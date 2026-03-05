"""
Edit SYLLABLES_TO_MERGE below after inspecting:
  - trajectory plots   (PROJECT_DIR/<model_name>/trajectory_plots/)
  - grid movies        (PROJECT_DIR/<model_name>/grid_movies/)
  - dendrogram         (PROJECT_DIR/<model_name>/similarity_dendrogram.pdf)

Outputs (saved inside PROJECT_DIR/<model_name>/):
  results_merged.h5             — merged results file
  trajectory_plots_merged/      — trajectory plots for merged syllables
  grid_movies_merged/           — grid movies for merged syllables
  similarity_dendrogram_merged/ — dendrogram for merged syllables
"""
from __future__ import annotations

import os
import numpy as np
import keypoint_moseq as kpms
from pathlib import Path

from config import (
    PROJECT_DIR, DATA_ROOT,
    NORMALIZE_COORDS, NORMALIZE_BY_BONE, BONE_A, BONE_B,
    USE_TRACKING_SCORE_MASK, TRACKING_SCORE_THRESHOLD,
    collect_h5_files, subsample_files, load_json,
    load_sleap_h5, male_track_index_from_stem_strict,
    normalize_by_bone_length,
)

# =============================================================================
# ★  EDIT THIS before running  ★
# Each inner list = group of syllables to collapse into one.
# Example:  [[1, 3], [4, 5]]  merges {1,3} → 1  and  {4,5} → 4
# =============================================================================
SYLLABLES_TO_MERGE: list[list[int]] = [
    # [1, 3],
    # [4, 5],
]

# Override to skip reading from model_selection.json
BEST_MODEL_NAME: str | None = None


def load_coordinates_for_viz():
    """Re-load and normalise coordinates (needed by kpms visualisation calls)."""
    h5_files = collect_h5_files(DATA_ROOT)
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {DATA_ROOT.resolve()}")
    h5_files = subsample_files(h5_files)

    coordinates_dict, confidences_dict = {}, {}
    bodyparts_ref = None

    for h5_path in h5_files:
        loaded = load_sleap_h5(h5_path)
        if bodyparts_ref is None:
            bodyparts_ref = loaded.bodyparts

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
    if not SYLLABLES_TO_MERGE:
        print(
            "[MERGE] SYLLABLES_TO_MERGE is empty — nothing to do.\n"
            "  Edit the list at the top of this file and re-run."
        )
        return

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

    print(f"\n[MERGE] Model : {model_name}")
    print(f"[MERGE] Groups: {SYLLABLES_TO_MERGE}\n")

    # ------------------------------------------------------------------
    # Load & merge results
    # ------------------------------------------------------------------
    results_path = os.path.join(str(PROJECT_DIR), model_name, "results.h5")
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"results.h5 not found at {results_path}\n"
            "Run 04_visualize.py (or 02_train.py) first."
        )

    results_raw      = kpms.load_hdf5(results_path)
    syllable_mapping = kpms.generate_syllable_mapping(results_raw, SYLLABLES_TO_MERGE)
    new_results      = kpms.apply_syllable_mapping(results_raw, syllable_mapping)

    new_results_path = os.path.join(str(PROJECT_DIR), model_name, "results_merged.h5")
    kpms.save_hdf5(new_results_path, new_results)
    print(f"[MERGE] Merged results saved to: {new_results_path}")

    # ------------------------------------------------------------------
    # Re-load coordinates
    # ------------------------------------------------------------------
    coordinates, cfg = load_coordinates_for_viz()

    # ------------------------------------------------------------------
    # Trajectory plots (merged)
    # ------------------------------------------------------------------
    print("\n[MERGE] Generating trajectory plots ...")
    merged_traj_dir = os.path.join(
        str(PROJECT_DIR), model_name, "trajectory_plots_merged"
    )
    kpms.generate_trajectory_plots(
        coordinates, new_results, output_dir=merged_traj_dir, **cfg
    )
    print(f"[MERGE] Saved to: {merged_traj_dir}")

    # ------------------------------------------------------------------
    # Grid movies (merged)
    # ------------------------------------------------------------------
    print("\n[MERGE] Generating grid movies ...")
    merged_grid_dir = os.path.join(
        str(PROJECT_DIR), model_name, "grid_movies_merged"
    )
    kpms.generate_grid_movies(
        new_results, output_dir=merged_grid_dir, coordinates=coordinates, **cfg
    )
    print(f"[MERGE] Saved to: {merged_grid_dir}")

    # ------------------------------------------------------------------
    # Syllable similarity dendrogram (merged)
    # ------------------------------------------------------------------
    print("\n[MERGE] Generating syllable similarity dendrogram ...")
    merged_dendro_dir = os.path.join(
        str(PROJECT_DIR), model_name, "similarity_dendrogram_merged"
    )
    kpms.plot_similarity_dendrogram(
        coordinates, new_results, output_dir=merged_dendro_dir, **cfg
    )
    print(f"[MERGE] Saved to: {merged_dendro_dir}")

    print("\n[MERGE] Done.\n")


if __name__ == "__main__":
    main()
