"""
  1. Computes EML (Expected Marginal Likelihood) scores for all trained models.
  2. Plots EML scores and a confusion matrix between the two best models.
  3. Prints the name of the best model and saves the result to
     QA_AUDIT/model_selection.json.

After this script completes, edit BEST_MODEL_NAME in 04_visualize.py
(or just let it auto-read from model_selection.json) and run:
  python 04_visualize.py
"""
from __future__ import annotations

import numpy as np
import keypoint_moseq as kpms

from config import (
    PROJECT_DIR, MULTI_SEED_PREFIX, NUM_MODEL_FITS,
    ensure_dir, save_json,
)


def main():
    qa_dir = PROJECT_DIR / "QA_AUDIT"
    ensure_dir(qa_dir)

    model_names = [f"{MULTI_SEED_PREFIX}-{i}" for i in range(NUM_MODEL_FITS)]

    # ------------------------------------------------------------------
    # Confusion matrix: top-2 models as a quick sanity check
    # ------------------------------------------------------------------
    print("[MODEL SELECT] Plotting confusion matrix (model 0 vs model 1) ...")
    results_0 = kpms.load_results(str(PROJECT_DIR), model_names[0])
    results_1 = kpms.load_results(str(PROJECT_DIR), model_names[1])
    kpms.plot_confusion_matrix(results_0, results_1)

    # ------------------------------------------------------------------
    # EML scores
    # ------------------------------------------------------------------
    print("[MODEL SELECT] Computing EML scores ...")
    eml_scores, eml_std_errs = kpms.expected_marginal_likelihoods(
        str(PROJECT_DIR), model_names
    )

    best_idx        = int(np.argmax(eml_scores))
    best_model_name = model_names[best_idx]

    print(f"\n[MODEL SELECT] Best model: {best_model_name}  "
          f"(EML = {eml_scores[best_idx]:.4f} ± {eml_std_errs[best_idx]:.4f})")

    kpms.plot_eml_scores(eml_scores, eml_std_errs, model_names)

    save_json(qa_dir / "model_selection.json", {
        "best_model":    best_model_name,
        "model_names":   model_names,
        "eml_scores":    [float(s) for s in eml_scores],
        "eml_std_errs":  [float(s) for s in eml_std_errs],
    })

    print(
        f"\n[MODEL SELECT] Results saved to QA_AUDIT/model_selection.json\n"
        f"  → Run:  python 04_visualize.py\n"
    )


if __name__ == "__main__":
    main()
