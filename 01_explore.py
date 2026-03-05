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
"""
01_explore.py — Run ONCE before training.

Steps:
  1. Load & preprocess data.
  2. Fit PCA on full spectrum → variance plot → ask user for num_pcs.
  3. Run kappa scan → plot results.

After this script you will know:
  - num_pcs  → already written to config.yml automatically
  - kappa    → inspect kappa scan plot, then update FULL_MODEL_KAPPA
               in config.py before running 02_train.py
"""
from __future__ import annotations

import numpy as np
import keypoint_moseq as kpms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    PROJECT_DIR,
    KAPPA_SCAN_PREFIX, KAPPA_SCAN_VALUES,
    KAPPA_DECREASE_FACTOR, NUM_AR_ITERS_SCAN, NUM_FULL_ITERS_SCAN,
    ensure_dir, save_json,
    load_and_preprocess, build_kpms_data,
)


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

    n_show   = min(max_components, len(evr))
    evr_show = evr[:n_show]
    cumvar   = np.cumsum(evr_show)
    comps    = np.arange(1, n_show + 1)

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
    # 1) Load + preprocess
    # ------------------------------------------------------------------
    coordinates_dict, confidences_dict, bodyparts, skeleton = load_and_preprocess()

    # ------------------------------------------------------------------
    # 2) Build kpms data structures
    # ------------------------------------------------------------------
    data, metadata, coordinates, cfg = build_kpms_data(
        coordinates_dict, confidences_dict, bodyparts, skeleton,
        estimate_sigmasq=True  # first run — write sigmasq_loc to config.yml
    )

    # ------------------------------------------------------------------
    # 3) PCA — full spectrum → variance plot → user picks n_pcs → re-fit
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
    cfg = kpms.load_config(str(PROJECT_DIR))
    save_json(qa_dir / "pca_n_components.json", {"num_pcs": n_pcs})
    print(f"[PCA] config.yml updated with num_pcs={n_pcs}")

    # ------------------------------------------------------------------
    # 4) Kappa scan
    # ------------------------------------------------------------------
    print("\n" + "="*55)
    print("  KAPPA SCAN")
    print("="*55)

    # Keep originals safe — fit_model may modify data/metadata in place
    data_orig     = data
    metadata_orig = metadata

    for kappa in KAPPA_SCAN_VALUES:
        print(f"\n[KAPPA SCAN] kappa={kappa:.2e}")
        model_name = f"{KAPPA_SCAN_PREFIX}-{kappa:.2e}"
        model      = kpms.init_model(data_orig, pca=pca, **cfg)

        model = kpms.update_hypparams(model, kappa=kappa)
        model = kpms.fit_model(
            model, data_orig, metadata_orig, str(PROJECT_DIR), model_name,
            ar_only=True, num_iters=NUM_AR_ITERS_SCAN, save_every_n_iters=25
        )[0]

        model = kpms.update_hypparams(model, kappa=kappa / KAPPA_DECREASE_FACTOR)
        kpms.fit_model(
            model, data_orig, metadata_orig, str(PROJECT_DIR), model_name,
            ar_only=False, start_iter=NUM_AR_ITERS_SCAN,
            num_iters=NUM_FULL_ITERS_SCAN, save_every_n_iters=25
        )

    kpms.plot_kappa_scan(KAPPA_SCAN_VALUES, str(PROJECT_DIR), KAPPA_SCAN_PREFIX)

    print(
        "\n[KAPPA SCAN] Done.\n"
        "  → Inspect the kappa scan plot.\n"
        "  → Update FULL_MODEL_KAPPA in config.py if needed.\n"
        "  → Then run:  python 02_train.py\n"
    )


if __name__ == "__main__":
    main()
