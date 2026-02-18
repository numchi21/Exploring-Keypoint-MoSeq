from pathlib import Path
import re
import numpy as np
import h5py
import keypoint_moseq as kpms
from collections import Counter


# ======================================
# CONFIGURACIÓN
# ======================================

project_dir = Path("kpms_project")
model_name = "2026_02_13-16_05_06"

videos_root = Path("data/videos")

video_suffix = "_inferencia"   # cambiar a "_inferencia" si corresponde

TARGET_SESSIONS = ["128_S2", "129_S2", "51_S2", "54_S2", "81_S2", "86_S2", "92_S2", "55_S3", "85_S3", "92_S3"]

# ======================================
# CARGADOR SLEAP
# ======================================

def load_sleap_h5(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        tracks = np.array(f["/tracks"], dtype=np.float32)      # (R,2,K,T)
        occ = np.array(f["/track_occupancy"], dtype=np.uint8)  # (T,R)
        node_names = f["/node_names"][...]

        point_scores = None
        if "/point_scores" in f:
            point_scores = np.array(f["/point_scores"], dtype=np.float32)  # (R,K,T)

    bodyparts = [
        x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in node_names
    ]

    R, D, K, T = tracks.shape
    tracks = np.transpose(tracks, (3, 2, 1, 0))  # (T,K,2,R)

    # conf real si existe
    if point_scores is not None:
        conf = np.transpose(point_scores, (2, 1, 0))  # (T,K,R)
    else:
        conf = np.ones((T, K, R), dtype=np.float32)

    # occupancy -> NaNs y conf=0
    for r in range(R):
        missing = occ[:, r] == 0
        if np.any(missing):
            tracks[missing, :, :, r] = np.nan
            conf[missing, :, r] = 0.0

    # cualquier NaN => conf=0
    invalid = np.isnan(tracks[:, :, 0, :]) | np.isnan(tracks[:, :, 1, :])
    conf[invalid] = 0.0

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

    # CÓDIGO CORREGIDO:

    def get_h5_path(sid: str) -> Path:
        for subdir in ["S2", "S3"]:
            path = Path(f"data/h5/{subdir}/{sid}.h5")
            if path.exists():
                return path
        return None  # Si no existe en ningún lado

    # ----------------------------------
    # Reconstruir coordinates (solo macho)
    # ---------------------------------
    coordinates = {}

    for rec in results.keys():

        # Parseo ESTRICTO del nombre usado en entrenamiento
        m = re.fullmatch(r"(\d+_S[23])__male_track([12])", rec)
        if m is None:
            raise ValueError(f"Nombre de recording inesperado en results: {rec}")

        sid = m.group(1)
        male_r = int(m.group(2)) - 1  # 0 o 1

        h5_path = get_h5_path(sid)

        if h5_path is None or not h5_path.exists():
            print(f"[WARNING] No encuentro h5 para {sid}")
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

    kpms.generate_grid_movies(
        results,
        str(project_dir),
        model_name,
        coordinates=coordinates,
        video_paths=video_paths,
        fps=30,
        syllables_to_plot=all_syllables[:10],
        overlay_syllable=True,  # muestra el estado
        overlay_trajectory=True,  # ESTA ES LA CLAVE
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
