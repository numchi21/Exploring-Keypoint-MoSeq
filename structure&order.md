# STRUCTURE

kpms_project/
│
├── config.py                        # Parámetros globales + utilidades compartidas
│
├── 01_explore.py                    # Ejecutar UNA VEZ antes de entrenar
│   ├── Carga y preprocesado de datos
│   ├── PCA (espectro completo)
│   │   ├── Variance plot → pca_variance.png
│   │   └── Input interactivo: ¿cuántos PCs?
│   │       └── Re-fit PCA con n_pcs elegido → guarda en config.yml
│   └── Kappa scan (1e3 → 1e7)
│       ├── Por cada kappa: AR-only + full model
│       └── plot_kappa_scan() → elegir kappa para config.py
│
├── 02_train.py                      # Entrenamiento multi-seed
│   ├── Carga y preprocesado de datos
│   │   ├── Subsampling de vídeos (N_RANDOM_VIDEOS)
│   │   ├── Carga SLEAP (.h5)
│   │   │   ├── Selección track macho (MALE_TRACK2_IDS)
│   │   │   ├── Tracking score mask
│   │   │   └── QA anti-swaps
│   │   ├── Normalización por longitud de hueso
│   │   ├── Outlier removal
│   │   └── format_data()
│   └── Multi-seed fitting (NUM_MODEL_FITS modelos)
│       ├── Por cada seed:
│       │   ├── Stage 1: AR-only  (AR_ONLY_KAPPA)
│       │   ├── Stage 2: Full model (FULL_MODEL_KAPPA)
│       │   ├── reindex_syllables
│       │   └── extract_results → results.h5 + results.csv
│       └── → Todos los modelos guardados en PROJECT_DIR/
│
├── 03_select_model.py               # Selección del mejor modelo
│   ├── Confusion matrix (modelo 0 vs modelo 1)
│   ├── EML scores (todos los modelos)
│   │   └── plot_eml_scores()
│   └── → model_selection.json (best_model_name)
│
├── 04_visualize.py                  # Visualizaciones del mejor modelo
│   ├── Lee best_model de model_selection.json
│   ├── Trajectory plots → trajectory_plots/
│   ├── Grid movies      → grid_movies/
│   └── Syllable dendrogram → similarity_dendrogram.pdf
│
└── 05_merge_syllables.py            # Merge post-hoc (iterativo, opcional)
    ├── Editar SYLLABLES_TO_MERGE manualmente
    ├── generate_syllable_mapping + apply_syllable_mapping
    ├── Trajectory plots → trajectory_plots_merged/
    ├── Grid movies      → grid_movies_merged/
    └── Dendrogram       → similarity_dendrogram_merged/


# ORDER

config.py        (nunca ejecutar, solo se edita)
    ↓
01_explore.py     (solo se ejecuta una vez)
    ↓
config.py  ← actualizar FULL_MODEL_KAPPA
    ↓
02_train.py
    ↓
03_select_model.py
    ↓
04_visualize.py
    ↓
05_merge_syllables.py 
