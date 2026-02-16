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

TARGET_SESSIONS = ["92_S3", "50_S2"]

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

# ======================================
# MAIN
# ======================================

def main():
    # ----------------------------------
    # Cargar config
    # ----------------------------------
    cfg = kpms.load_config(str(project_dir))

    # ----------------------------------
    # Cargar results existentes
    # ----------------------------------
    results = kpms.load_results(str(project_dir), model_name)

    # Filtrar results solo a sesiones deseadas
    filtered_results = {}

    for rec in results:
        m = re.fullmatch(r"(\d+_S[23])__male_track([12])", rec)
        if m is None:
            continue
        sid = m.group(1)

        if sid in TARGET_SESSIONS:
            filtered_results[rec] = results[rec]

    if not filtered_results:
        raise ValueError("Ninguna sesión encontrada en results.")

    results = filtered_results
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

        # Parseo ESTRICTO del nombre usado en entrenamiento
        m = re.fullmatch(r"(\d+_S[23])__male_track([12])", rec)
        if m is None:
            raise ValueError(f"Nombre de recording inesperado en results: {rec}")

        sid = m.group(1)
        male_r = int(m.group(2)) - 1  # 0 o 1

        h5_path = h5_root / f"{sid}.h5"

        if not h5_path.exists():
            print(f"[WARNING] No encuentro {h5_path}")
            continue

        coords, _, _ = load_sleap_h5(h5_path)

        coordinates[rec] = coords[:, :, :, male_r]


    results = {rec: results[rec] for rec in coordinates.keys()}
    results.keys() == coordinates.keys()

    # ----------------------------------
    # Asignar rutas de vídeo
    # ----------------------------------
    video_paths = {}
    valid_recs = []

    for rec in list(coordinates.keys()):  # list() para evitar problemas

        sid = rec.split("__")[0]
        video_path = videos_root / f"{sid}{video_suffix}.mp4"

        if video_path.exists():
            video_paths[rec] = str(video_path)
            valid_recs.append(rec)
        else:
            print(f"[WARNING] Video no encontrado: {video_path}")

    # Mantener solo recordings válidos
    coordinates = {rec: coordinates[rec] for rec in valid_recs}
    results = {rec: results[rec] for rec in valid_recs}

    # ----------------------------------
    # Generar Grid Movies
    # ----------------------------------
    # Mantener solo recs con coordenadas válidas
    results = {rec: results[rec] for rec in coordinates.keys()}

    # Mostrar todas las sílabas
    syllables_to_plot = None

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