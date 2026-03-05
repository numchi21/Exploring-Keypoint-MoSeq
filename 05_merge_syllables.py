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
import keypoint_moseq as kpms

from config import (
    PROJECT_DIR,
    load_json,
    load_and_preprocess, build_kpms_data,
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

# Set this to override reading from model_selection.json
BEST_MODEL_NAME: str | None = None


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
                "QA_AUDIT/model_selection.json not found.\n"
                "Run 03_select_model.py first, or set BEST_MODEL_NAME manually."
            )
        model_name = load_json(selection_path)["best_model"]

    print(f"\n[MERGE] Model : {model_name}")
    print(f"[MERGE] Groups: {SYLLABLES_TO_MERGE}\n")

    # ------------------------------------------------------------------
    # Load + merge results
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
    # Reload coordinates
    # ------------------------------------------------------------------
    coordinates_dict, confidences_dict, bodyparts, skeleton = load_and_preprocess(
        save_qa=False
    )
    _, _, coordinates, cfg = build_kpms_data(
        coordinates_dict, confidences_dict, bodyparts, skeleton,
        estimate_sigmasq=False
    )

    # ------------------------------------------------------------------
    # Trajectory plots (merged)
    # ------------------------------------------------------------------
    print("\n[MERGE] Generating trajectory plots ...")
    merged_traj_dir = os.path.join(str(PROJECT_DIR), model_name, "trajectory_plots_merged")
    kpms.generate_trajectory_plots(
        coordinates, new_results, output_dir=merged_traj_dir, **cfg
    )
    print(f"[MERGE] Saved to: {merged_traj_dir}")

    # ------------------------------------------------------------------
    # Grid movies (merged)
    # ------------------------------------------------------------------
    print("\n[MERGE] Generating grid movies ...")
    merged_grid_dir = os.path.join(str(PROJECT_DIR), model_name, "grid_movies_merged")
    kpms.generate_grid_movies(
        new_results, output_dir=merged_grid_dir, coordinates=coordinates, **cfg
    )
    print(f"[MERGE] Saved to: {merged_grid_dir}")

    # ------------------------------------------------------------------
    # Syllable similarity dendrogram (merged)
    # ------------------------------------------------------------------
    print("\n[MERGE] Generating syllable similarity dendrogram ...")
    merged_dendro_dir = os.path.join(str(PROJECT_DIR), model_name, "similarity_dendrogram_merged")
    kpms.plot_similarity_dendrogram(
        coordinates, new_results, output_dir=merged_dendro_dir, **cfg
    )
    print(f"[MERGE] Saved to: {merged_dendro_dir}")

    print("\n[MERGE] Done.\n")


if __name__ == "__main__":
    main()
