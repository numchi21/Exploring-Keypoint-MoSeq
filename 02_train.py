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

import jax
import keypoint_moseq as kpms

from config import (
    PROJECT_DIR,
    MULTI_SEED_PREFIX, NUM_MODEL_FITS,
    AR_ONLY_KAPPA, FULL_MODEL_KAPPA,
    NUM_AR_ITERS, NUM_FULL_ITERS,
    ensure_dir, save_json,
    load_and_preprocess, build_kpms_data,
)


def main():
    ensure_dir(PROJECT_DIR)

    # ------------------------------------------------------------------
    # 1) Load + preprocess
    # ------------------------------------------------------------------
    coordinates_dict, confidences_dict, bodyparts, skeleton = load_and_preprocess()

    # ------------------------------------------------------------------
    # 2) Build kpms data structures
    # ------------------------------------------------------------------
    data, metadata, coordinates, cfg = build_kpms_data(
        coordinates_dict, confidences_dict, bodyparts, skeleton
    )

    # ------------------------------------------------------------------
    # 3) Load PCA saved by 01_explore.py
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 4) Multi-seed training
    # ------------------------------------------------------------------
    print("\n" + "="*55)
    print(f"  MULTI-SEED TRAINING  ({NUM_MODEL_FITS} models)")
    print("="*55)
    print(f"  AR-only kappa   : {AR_ONLY_KAPPA:.1e}  ({NUM_AR_ITERS} iters)")
    print(f"  Full model kappa: {FULL_MODEL_KAPPA:.1e}  ({NUM_FULL_ITERS} iters)")
    print("="*55 + "\n")

    # Keep originals so each restart uses the same input data.
    # load_checkpoint can return updated data/metadata — we propagate them
    # forward so state stays consistent across restarts.
    data_orig     = data
    metadata_orig = metadata

    qa_dir = PROJECT_DIR / "QA_AUDIT"
    ensure_dir(qa_dir)

    for restart in range(NUM_MODEL_FITS):
        print(f"\n[TRAIN] Model {restart + 1}/{NUM_MODEL_FITS}  (seed={restart})")
        model_name = f"{MULTI_SEED_PREFIX}-{restart}"

        # Log the seed used for this model
        save_json(qa_dir / f"model_seed_{model_name}.json", {"seed": restart})

        model = kpms.init_model(
            data_orig, pca=pca, **cfg, seed=jax.random.PRNGKey(restart)
        )

        # Stage 1: AR-only
        model = kpms.update_hypparams(model, kappa=AR_ONLY_KAPPA)
        model = kpms.fit_model(
            model, data_orig, metadata_orig, str(PROJECT_DIR), model_name,
            ar_only=True, num_iters=NUM_AR_ITERS
        )[0]

        # Stage 2: full model
        model = kpms.update_hypparams(model, kappa=FULL_MODEL_KAPPA)
        kpms.fit_model(
            model, data_orig, metadata_orig, str(PROJECT_DIR), model_name,
            ar_only=False, start_iter=NUM_AR_ITERS, num_iters=NUM_FULL_ITERS
        )

        kpms.reindex_syllables_in_checkpoint(str(PROJECT_DIR), model_name)
        model, data_orig, metadata_orig, _ = kpms.load_checkpoint(
            str(PROJECT_DIR), model_name
        )
        results = kpms.extract_results(model, metadata_orig, str(PROJECT_DIR), model_name)
        kpms.save_results_as_csv(results, str(PROJECT_DIR), model_name)
        print(f"[TRAIN] Saved: {model_name}")

    print(
        "\n[TRAIN] All models trained.\n"
        "  → Run:  python 03_select_model.py\n"
    )


if __name__ == "__main__":
    main()
