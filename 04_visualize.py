"""
Reads best_model from QA_AUDIT/model_selection.json (written by
03_select_model.py).  You can also override it by setting
BEST_MODEL_NAME below.

Outputs (all saved inside PROJECT_DIR/<model_name>/):
  trajectory_plots/         — median pose trajectory per syllable
  grid_movies/              — video clips of each syllable
  similarity_dendrogram.pdf — hierarchical clustering of syllables
"""
from __future__ import annotations

import os
import keypoint_moseq as kpms

from config import (
    PROJECT_DIR,
    ensure_dir, load_json,
    load_and_preprocess, build_kpms_data,
)

# Set this to override reading from model_selection.json
BEST_MODEL_NAME: str | None = None


def main():
    ensure_dir(PROJECT_DIR)

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

    print(f"\n[VIZ] Generating visualisations for: {model_name}\n")

    # ------------------------------------------------------------------
    # Load coordinates + config
    # ------------------------------------------------------------------
    coordinates_dict, confidences_dict, bodyparts, skeleton = load_and_preprocess(
        save_qa=False
    )
    _, _, coordinates, cfg = build_kpms_data(
        coordinates_dict, confidences_dict, bodyparts, skeleton,
        estimate_sigmasq=False
    )

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
    print(f"[VIZ] Saved to: {os.path.join(str(PROJECT_DIR), model_name, 'trajectory_plots')}")

    # ------------------------------------------------------------------
    # Grid movies
    # ------------------------------------------------------------------
    print("\n[VIZ] Generating grid movies ...")
    kpms.generate_grid_movies(
        results, str(PROJECT_DIR), model_name,
        coordinates=coordinates, **cfg
    )
    print(f"[VIZ] Saved to: {os.path.join(str(PROJECT_DIR), model_name, 'grid_movies')}")

    # ------------------------------------------------------------------
    # Syllable similarity dendrogram
    # ------------------------------------------------------------------
    print("\n[VIZ] Generating syllable similarity dendrogram ...")
    kpms.plot_similarity_dendrogram(
        coordinates, results, str(PROJECT_DIR), model_name, **cfg
    )
    print(f"[VIZ] Saved to: {os.path.join(str(PROJECT_DIR), model_name, 'similarity_dendrogram.pdf')}")

    print(
        "\n[VIZ] All visualisations complete.\n"
        "  → Review trajectory plots, grid movies, and the dendrogram.\n"
        "  → To merge similar syllables, edit SYLLABLES_TO_MERGE in\n"
        "    05_merge_syllables.py and run it.\n"
    )


if __name__ == "__main__":
    main()