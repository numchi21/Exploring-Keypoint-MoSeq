from __future__ import annotations

from jax import config as jax_config
jax_config.update("jax_enable_x64", True) #cambiado a true

import numpy as np
import h5py
from pathlib import Path
import keypoint_moseq as kpms
import re
import json
import csv
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
DATA_ROOT = Path("data/h5")
PROJECT_DIR = Path("kpms_project")

N_RANDOM_VIDEOS = 10
RANDOM_SEED = 123

# Enmascara frames donde tracking_score es bajo (si existe)
USE_TRACKING_SCORE_MASK = True
TRACKING_SCORE_THRESHOLD = 0.50  # ajustable

# Métricas QA anti-swaps (no enmascara; solo reporta)
RUN_SWAP_QA = True

# ---- Normalización de escala ----
NORMALIZE_COORDS = False
NORM_QUANTILE = 0.90  # robusto
# Si queremos normalizar por una longitud anatómica.
NORMALIZE_BY_BONE = True
BONE_A = "base_head"
BONE_B = "base_body"

# ---- Bodyparts ----
EXCLUDE_TAIL = True  # recomendación de keypoint moseq
ANTERIOR_BPS = ["nose", "upper_head"]
POSTERIOR_BPS = ["base_body", "base_tail"]

# ---- Fitting ----
NUM_AR_ITERS = 50
NUM_FULL_ITERS = 50

# Multi-seed para estabilidad, podemos probar
RUN_MULTI_SEED = False
MODEL_SEEDS = [0, 1, 2]


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
    skeleton_edges: Optional[List[List[str]]]  # [[bp1,bp2], ...] si está disponible
    tracking_scores_RT: Optional[np.ndarray]   # (R,T) si está disponible

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
            # edge_names puede venir como bytes
            sk = []
            for a, b in edge_names:
                aa = a.decode("utf-8") if isinstance(a, (bytes, np.bytes_)) else str(a)
                bb = b.decode("utf-8") if isinstance(b, (bytes, np.bytes_)) else str(b)
                sk.append([aa, bb])
            skeleton_edges = sk

    # decode node_names
    bodyparts = []
    for x in np.array(node_names):
        if isinstance(x, (bytes, np.bytes_)):
            bodyparts.append(x.decode("utf-8"))
        else:
            try:
                bodyparts.append(bytes(x).decode("utf-8"))
            except Exception:
                bodyparts.append(str(x))

    # sanity
    if tracks.ndim != 4:
        raise ValueError(f"{h5_path}: tracks.ndim={tracks.ndim}, esperado 4")
    R, D, K, T = tracks.shape
    if D != 2:
        raise ValueError(f"{h5_path}: D={D}, esperado 2 (x,y)")
    if len(bodyparts) != K:
        raise ValueError(f"{h5_path}: len(node_names)={len(bodyparts)} != K={K}")
    if occ.shape != (T, R):
        raise ValueError(f"{h5_path}: occupancy shape {occ.shape} != (T,R)=({T},{R})")

    # coords -> (T,K,2,R)
    tracks_TK2R = np.transpose(tracks, (3, 2, 1, 0))  # (T,K,2,R)

    # confidences reales
    if point_scores is not None:
        if point_scores.shape != (R, K, T):
            raise ValueError(f"{h5_path}: point_scores {point_scores.shape} esperado (R,K,T)=({R},{K},{T})")
        conf_TKR = np.transpose(point_scores, (2, 1, 0))  # (T,K,R)
    else:
        # fallback binario (no recomendado, pero estable)
        conf_TKR = np.ones((T, K, R), dtype=np.float32)

    # occupancy mask -> NaN coords y conf=0
    for r in range(R):
        missing = (occ[:, r] == 0)
        if np.any(missing):
            tracks_TK2R[missing, :, :, r] = np.nan
            conf_TKR[missing, :, r] = 0.0

    # cualquier NaN => conf=0
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
    """
    Reporta métricas simples:
    - velocidad del centroide por track
    - jumps extremos (umbral basado en MAD)
    - cruces frecuentes (distancia inter-track baja + cambio relativo)
    """
    T, K, _, R = tracks_TK2R.shape
    # centroid aproximado: media de keypoints válidos
    cent = np.full((T, 2, R), np.nan, dtype=np.float32)
    for r in range(R):
        xy = tracks_TK2R[:, :, :, r]  # (T,K,2)
        valid = np.isfinite(xy).all(axis=2)  # (T,K)
        # media por frame
        for t in range(T):
            v = valid[t]
            if np.any(v):
                cent[t, :, r] = np.nanmean(xy[t, v, :], axis=0)

    report = {}
    # speed y jumps por track
    for r in range(R):
        c = cent[:, :, r]
        dc = np.linalg.norm(np.diff(c, axis=0), axis=1)  # (T-1,)
        mad = robust_mad(dc)
        med = np.nanmedian(dc)
        thr = med + 10.0 * mad
        jumps = np.nansum(dc > thr)
        report[f"track{r+1}_speed_median"] = float(med)
        report[f"track{r+1}_speed_mad"] = float(mad)
        report[f"track{r+1}_jumps_10MAD"] = float(jumps)

    # distancia entre tracks (si R>=2)
    if R >= 2:
        d12 = np.linalg.norm(cent[:, :, 0] - cent[:, :, 1], axis=1)  # (T,)
        report["intertrack_dist_median"] = float(np.nanmedian(d12))
        report["intertrack_dist_p05"] = float(np.nanquantile(d12[np.isfinite(d12)], 0.05)) if np.any(np.isfinite(d12)) else float("nan")

    return report


# =============================
# NORMALIZACIÓN DE ESCALA
# =============================
def normalize_coordinates_global(coordinates_dict: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
    """
    coordinates_dict[rec] = (T,K,2)
    Normaliza por escala robusta global basada en cuantiles de distancia al centro.
    """
    all_xy = []
    for c in coordinates_dict.values():
        xy = c.reshape(-1, 2)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if xy.size:
            all_xy.append(xy)
    if not all_xy:
        return coordinates_dict, 1.0

    all_xy = np.concatenate(all_xy, axis=0)
    center = np.median(all_xy, axis=0)
    dist = np.linalg.norm(all_xy - center, axis=1)
    scale = float(np.quantile(dist, NORM_QUANTILE)) + 1e-12

    out = {}
    for k, c in coordinates_dict.items():
        out[k] = c / scale
    return out, scale

def normalize_by_bone_length(coordinates_dict: Dict[str, np.ndarray], bodyparts: List[str], a: str, b: str) -> Tuple[Dict[str, np.ndarray], float]:
    if a not in bodyparts or b not in bodyparts:
        raise ValueError(f"Para NORMALIZE_BY_BONE necesitas {a} y {b} en bodyparts.")
    ia, ib = bodyparts.index(a), bodyparts.index(b)

    lengths = []
    for c in coordinates_dict.values():
        pa = c[:, ia, :]  # (T,2)
        pb = c[:, ib, :]
        v = np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1)
        if np.any(v):
            lengths.append(np.linalg.norm(pa[v] - pb[v], axis=1))
    if not lengths:
        return coordinates_dict, 1.0

    lengths = np.concatenate(lengths)
    scale = float(np.median(lengths)) + 1e-12
    out = {k: c / scale for k, c in coordinates_dict.items()}
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
# PIPELINE PRINCIPAL
# =============================
def main_one_run(run_tag: str, model_seed: Optional[int] = None, kappa_full: Optional[float] = None):
    ensure_dir(PROJECT_DIR)
    qa_dir = PROJECT_DIR / "QA_AUDIT"
    ensure_dir(qa_dir)

    h5_files = collect_h5_files(DATA_ROOT)
    if not h5_files:
        raise FileNotFoundError(f"No encontré .h5 en {DATA_ROOT.resolve()}")

    h5_files = subsample_files(h5_files)

    # Guardar lista de archivos usada (punto 0)
    file_list = [str(p.resolve()) for p in h5_files]
    save_json(qa_dir / f"used_h5_files_{run_tag}.json", file_list)

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
        tracking_scores = loaded.tracking_scores_RT  # (R,T) o None

        T, K, _, R = coords_TK2R.shape

        # Consistencia de bodyparts entre archivos
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

        # QA swaps (report)
        if RUN_SWAP_QA:
            rep = swap_qa_report(coords_TK2R)
        else:
            rep = {}

        # --- SELECCIÓN MACHO ---
        male_r, sid = male_track_index_from_stem_strict(h5_path.stem)
        if not (0 <= male_r < R):
            raise ValueError(f"{h5_path.name}: male_r={male_r} fuera de rango para R={R}")

        # Anti-swaps: enmascarar frames con tracking_score bajo (si existe)
        if USE_TRACKING_SCORE_MASK and tracking_scores is not None:
            # tracking_scores: (R,T) -> queremos track male_r
            ts = tracking_scores[male_r]  # (T,)
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
        coords_male = coords_TK2R[:, :, :, male_r]   # (T,K,2)
        conf_male = conf_TKR[:, :, male_r]           # (T,K)

        # Métricas por recording (punto 0 + 2 + 8)
        valid_xy = np.isfinite(coords_male).all(axis=2)  # (T,K)
        mask_frac = float(np.mean(valid_xy)) if valid_xy.size else 0.0

        # rangos de coords (en px antes de normalizar)
        finite = coords_male[np.isfinite(coords_male)]
        coord_min = float(np.min(finite)) if finite.size else float("nan")
        coord_max = float(np.max(finite)) if finite.size else float("nan")

        # validez de anterior/posterior
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

    save_csv(qa_dir / f"per_recording_metrics_{run_tag}.csv", per_recording_metrics)

    # Verificación final de Track2 subset (con subsampling)
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
    norm_info = {"enabled": False}
    if NORMALIZE_COORDS:
        if NORMALIZE_BY_BONE:
            coordinates_dict, scale = normalize_by_bone_length(coordinates_dict, bodyparts_ref, BONE_A, BONE_B)
            norm_info = {"enabled": True, "method": "bone_median", "bone": [BONE_A, BONE_B], "scale": scale}
        else:
            coordinates_dict, scale = normalize_coordinates_global(coordinates_dict)
            norm_info = {"enabled": True, "method": "global_quantile", "quantile": NORM_QUANTILE, "scale": scale}
        save_json(qa_dir / f"normalization_{run_tag}.json", norm_info)
        print(f"[NORM] Coordenadas normalizadas. Info: {norm_info}")

    # =============================
    # 5) skeleton + checks duros
    # =============================
    bodyparts = bodyparts_ref

    # Skeleton: usar el del h5 si existe; si no, fallback al tuyo
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

    # use_bodyparts: excluir cola si aplica
    if EXCLUDE_TAIL:
        use_bodyparts = [bp for bp in bodyparts if "tail" not in bp.lower()]
        # opcional: mantener base_tail si quieres posterior estable
        if "base_tail" in bodyparts and "base_tail" not in use_bodyparts:
            use_bodyparts.append("base_tail")
    else:
        use_bodyparts = bodyparts

    anterior_bodyparts = [bp for bp in ANTERIOR_BPS if bp in bodyparts]
    posterior_bodyparts = [bp for bp in POSTERIOR_BPS if bp in bodyparts]

    # checks
    bp_set = set(bodyparts)
    for a, b in skeleton:
        if a not in bp_set or b not in bp_set:
            raise ValueError(f"Skeleton edge ({a},{b}) not in bodyparts")
    for x in anterior_bodyparts + posterior_bodyparts:
        if x not in bp_set:
            raise ValueError(f"Bodypart {x} not in bodyparts")

    # setup project
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
    # Outlier removal + format
    # =============================
    coordinates, confidences = kpms.outlier_removal(
        coordinates_dict,
        confidences_dict,
        str(PROJECT_DIR),
        overwrite=False,
        **cfg
    )
    data, metadata = kpms.format_data(coordinates, confidences, **cfg)

    # Si quieres forzar precisión de data (tu patrón original)
    from jax_moseq.utils.debugging import convert_data_precision
    data = convert_data_precision(data)

    # =============================
    # Estimar sigmasq_loc
    # =============================
    kpms.update_config(
        str(PROJECT_DIR),
        sigmasq_loc=kpms.estimate_sigmasq_loc(
            data["Y"], data["mask"], filter_size=cfg["fps"]
        )
    )
    cfg = kpms.load_config(str(PROJECT_DIR))

    # =============================
    # PCA + init (no lo hacíamos antes)
    # =============================
    pca = kpms.fit_pca(data["Y"], data["mask"], **cfg)

    try:
        kpms.save_pca(pca, str(PROJECT_DIR))
    except Exception:
        pass

    model = kpms.init_model(data, pca=pca, **cfg)

    # Si tu versión permite semilla explícita en init, úsala aquí.
    # (No todas las versiones la exponen; por eso lo dejamos opcional.)
    if model_seed is not None:
        # guardamos seed en QA para trazabilidad
        save_json(qa_dir / f"model_seed_{run_tag}.json", {"seed": model_seed})

    # =============================
    # Fit AR-HMM
    # =============================
    model, model_name = kpms.fit_model(
        model, data, metadata, str(PROJECT_DIR),
        ar_only=True, num_iters=NUM_AR_ITERS
    )

    # =============================
    # Fit full model
    # =============================
    model, data, metadata, current_iter = kpms.load_checkpoint(
        str(PROJECT_DIR), model_name, iteration=NUM_AR_ITERS
    )

    # kappa tuning (punto 9)
    kappa_use = float(kappa_full) if kappa_full is not None else 1e4
    model = kpms.update_hypparams(model, kappa=kappa_use)
    save_json(qa_dir / f"kappa_{run_tag}.json", {"kappa_full": kappa_use})

    model = kpms.fit_model(
        model, data, metadata, str(PROJECT_DIR),
        model_name=model_name,
        ar_only=False,
        start_iter=current_iter,
        num_iters=current_iter + NUM_FULL_ITERS,
    )[0]

    # reindex + results
    kpms.reindex_syllables_in_checkpoint(str(PROJECT_DIR), model_name)
    model, data, metadata, _ = kpms.load_checkpoint(str(PROJECT_DIR), model_name)

    results = kpms.extract_results(model, metadata, str(PROJECT_DIR), model_name)
    kpms.save_results_as_csv(results, str(PROJECT_DIR), model_name)

    results = kpms.load_results(str(PROJECT_DIR), model_name)
    kpms.generate_trajectory_plots(coordinates, results, str(PROJECT_DIR), model_name, **cfg)

    print("DONE:", (PROJECT_DIR / model_name).resolve())
    return model_name


def main():
    run_tag = "default"

    main_one_run(
        run_tag=run_tag,
        model_seed=None,
        kappa_full=None
    )

if __name__ == "__main__":
    main()
