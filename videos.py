from pathlib import Path
import re
import numpy as np
import h5py
import keypoint_moseq as kpms
import random
from collections import Counter


# ======================================
# CONFIGURACIÓN
# ======================================

project_dir = Path("kpms_project")
model_name = "TU_MODELO_AQUI"

h5_root = Path("data_new/h5")
videos_root = Path("data/videos")

video_suffix = "_original"   # cambiar a "_inferencia" si corresponde
MODE = "random"              # "random" o "representative"

# ======================================
# DEFINICIÓN MACHO
# ======================================

MALE_TRACK2_IDS = {
    "50_S2", "50_S3", "51_S2", "55_S3",
    "85_S3", "86_S2", "92_S2", "92_S3", "128_S3"
}


def extract_session_id(stem: str):
    m = re.fullmatch(r"(\d+_S[23])", stem)
    if m is None:
        raise ValueError(f"Nombre inválido: {stem}")
    return m.group(1)


def male_track_index(stem: str):
    sid = extract_session_id(stem)
    return (1 if sid in MALE_TRACK2_IDS else 0), sid


# ======================================
# CARGADOR SLEAP
# ======================================

def load_sleap_h5(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        tracks = np.array(f["/tracks"], dtype=np.float32)      # (R,2,K,T)
        occ = np.array(f["/track_occupancy"], dtype=np.uint8)
        node_names = f["/node_names"][...]

    bodyparts = [
        x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in node_names
    ]

    R, D, K, T = tracks.shape
    tracks = np.transpose(tracks, (3, 2, 1, 0))  # (T,K,2,R)

    for r in range(R):
        missing = occ[:, r] == 0
        tracks[missing, :, :, r] = np.nan

    conf = np.ones((T, K, R), dtype=np.float32)
    conf[np.isnan(tracks[:, :, 0, :]) | np.isnan(tracks[:, :, 1, :])] = 0.0

    return tracks, conf, bodyparts

# ===============================================
# ALEATORIO O REPRESENTATIVO (definir en MODE)
# ===============================================
#aleatorio
num_to_plot = min(10, len(all_syllables))
selected_syllables_random = random.sample(all_syllables, num_to_plot)

print("Seleccion aleatoria (seed fija):", selected_syllables_random)

#representativo
most_common = counter.most_common(10)
selected_syllables_representative = [s for s, _ in most_common]

print("Selección representativa (top frecuencia):", selected_syllables_representative)

# ======================================
# MAIN
# ======================================

def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    # ----------------------------------
    # Cargar config
    # ----------------------------------
    cfg = kpms.load_config(str(project_dir))

    # ----------------------------------
    # Cargar results existentes
    # ----------------------------------
    results = kpms.load_results(str(project_dir), model_name)

    # Extraer todas las sílabas y contar frecuencia global
    counter = Counter()

    for rec in results:
        z = np.asarray(results[rec]["syllable"])
        counter.update(z)

    all_syllables = sorted(counter.keys())

    print("Sílabas disponibles:", all_syllables)
    print("Frecuencias:", counter)

    # ----------------------------------
    # Reconstruir coordinates (solo macho)
    # ----------------------------------
    coordinates = {}

    for rec in results.keys():

        sid = rec.split("__")[0]
        h5_path = h5_root / f"{sid}.h5"

        if not h5_path.exists():
            print(f"[WARNING] No encuentro {h5_path}")
            continue

        coords, _, _ = load_sleap_h5(h5_path)
        male_r, _ = male_track_index(sid)

        coordinates[rec] = coords[:, :, :, male_r]

    # ----------------------------------
    # Asignar rutas de vídeo
    # ----------------------------------
    video_paths = {}

    for rec in coordinates.keys():

        sid = rec.split("__")[0]
        video_path = videos_root / f"{sid}{video_suffix}.mp4"

        if not video_path.exists():
            print(f"[WARNING] Video no encontrado: {video_path}")
            continue

        video_paths[rec] = str(video_path)
    if MODE == "random":
        syllables_to_plot = selected_syllables_random
    elif MODE == "representative":
        syllables_to_plot = selected_syllables_representative
    else:
        raise ValueError("MODE debe ser 'random' o 'representative'")
    # ----------------------------------
    # Generar Grid Movies
    # ----------------------------------
    kpms.generate_grid_movies(
        results,
        str(project_dir),
        model_name,
        coordinates=coordinates,
        video_paths=video_paths,
        syllables_to_plot=syllables_to_plot,
        overlay_syllable=True,  # muestra el estado
        overlay_trajectory=True,  # 👈 ESTA ES LA CLAVE
        trajectory_length=30,  # nº de frames hacia atrás (ajustable) probar primero sin, si funciona añadir
        **cfg
    )

    # ----------------------------------
    # Otras visualizaciones para análisis
    # ----------------------------------
    kpms.generate_trajectory_plots(
        coordinates,
        results,
        str(project_dir),
        model_name,
        **cfg
    )

    kpms.plot_similarity_dendrogram(
        coordinates,
        results,
        str(project_dir),
        model_name,
        **cfg
    )

    print("\nDONE: Grid movies generados correctamente")


if __name__ == "__main__":
    main()