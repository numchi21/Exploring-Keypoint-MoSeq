# STRUCTURE

kpms_project/
│
├── config.py                          # Nunca se ejecuta — solo se edita y se importa
│   │
│   ├── PARÁMETROS
│   │   ├── Paths: DATA_ROOT, PROJECT_DIR
│   │   ├── Datos: FPS, N_RANDOM_VIDEOS, RANDOM_SEED
│   │   ├── Masking: USE_TRACKING_SCORE_MASK, TRACKING_SCORE_THRESHOLD
│   │   ├── Normalización: BONE_A, BONE_B
│   │   ├── Bodyparts: EXCLUDE_TAIL, ANTERIOR_BPS, POSTERIOR_BPS
│   │   ├── Entrenamiento: AR_ONLY_KAPPA, FULL_MODEL_KAPPA, NUM_AR_ITERS, NUM_FULL_ITERS, NUM_MODEL_FITS
│   │   ├── Kappa scan: KAPPA_SCAN_VALUES, KAPPA_DECREASE_FACTOR
│   │   └── Track asignación: MALE_TRACK2_IDS
│   │
│   └── FUNCIONES COMPARTIDAS
│       ├── I/O: collect_h5_files, ensure_dir, save_json, load_json, save_csv
│       ├── Track helpers: extract_session_id_strict, male_track_index_from_stem_strict, subsample_files
│       ├── SLEAP loader: load_sleap_h5 → SleapLoaded
│       ├── QA: swap_qa_report, recording_qa_metrics (anterior/posterior)
│       ├── Validación: validate_skeleton_and_bodyparts
│       ├── Normalización: normalize_by_bone_length
│       ├── load_and_preprocess()         ← pipeline de carga completo
│       │   ├── collect + subsample h5 files
│       │   ├── load_sleap_h5 por archivo
│       │   ├── Selección track macho
│       │   ├── Tracking score mask
│       │   ├── swap_qa_report + recording_qa_metrics → per_recording_metrics.csv
│       │   ├── Validación Track2 contra MALE_TRACK2_IDS
│       │   ├── Resumen de tracks (print)
│       │   └── normalize_by_bone_length → normalization.json
│       └── build_kpms_data(estimate_sigmasq)  ← pipeline de formato kpms
│           ├── validate_skeleton_and_bodyparts (hard checks)
│           ├── kpms.update_config + kpms.load_config
│           ├── kpms.outlier_removal
│           ├── kpms.format_data
│           ├── convert_data_precision
│           └── kpms.estimate_sigmasq_loc  (solo si estimate_sigmasq=True)
│
│
├── 01_explore.py                      # Ejecutar UNA VEZ antes de entrenar
│   ├── load_and_preprocess()
│   ├── build_kpms_data(estimate_sigmasq=True)
│   ├── PCA
│   │   ├── fit_pca (espectro completo)
│   │   ├── variance plot → QA_AUDIT/pca_variance.png
│   │   ├── Input interactivo: ¿cuántos PCs?
│   │   ├── Re-fit PCA con n_pcs elegido
│   │   └── Guarda PCA + actualiza config.yml + pca_n_components.json
│   └── Kappa scan
│       ├── data_orig / metadata_orig (protegidos)
│       ├── Por cada kappa en KAPPA_SCAN_VALUES:
│       │   ├── init_model
│       │   ├── Stage 1: AR-only  (kappa)
│       │   └── Stage 2: full model  (kappa / KAPPA_DECREASE_FACTOR)
│       └── plot_kappa_scan()
│           └── → Inspeccionar plot, actualizar FULL_MODEL_KAPPA en config.py
│
│
├── 02_train.py                        # Entrenamiento multi-seed
│   ├── load_and_preprocess()
│   ├── build_kpms_data(estimate_sigmasq=True)
│   ├── Carga PCA guardado por 01_explore.py
│   │   └── Fallback: fit_pca si no existe
│   └── Multi-seed fitting (NUM_MODEL_FITS modelos)
│       ├── data_orig / metadata_orig (protegidos)
│       ├── Por cada seed (0 → NUM_MODEL_FITS-1):
│       │   ├── Guarda model_seed_<name>.json → QA_AUDIT/
│       │   ├── init_model (jax.random.PRNGKey(seed))
│       │   ├── Stage 1: AR-only  (AR_ONLY_KAPPA)
│       │   ├── Stage 2: full model  (FULL_MODEL_KAPPA)
│       │   ├── reindex_syllables_in_checkpoint
│       │   ├── load_checkpoint → actualiza data_orig / metadata_orig
│       │   ├── extract_results
│       │   └── save_results_as_csv → results.h5 + results.csv
│       └── → Todos los modelos guardados en PROJECT_DIR/
│
│
├── 03_select_model.py                 # Selección del mejor modelo
│   ├── Confusion matrix (modelo 0 vs modelo 1)
│   ├── expected_marginal_likelihoods (todos los modelos)
│   ├── plot_eml_scores()
│   └── Guarda best_model → QA_AUDIT/model_selection.json
│
│
├── 04_visualize.py                    # Visualizaciones del mejor modelo
│   ├── Lee best_model de model_selection.json
│   ├── load_and_preprocess(save_qa=False)
│   ├── build_kpms_data(estimate_sigmasq=False)
│   ├── load_results
│   ├── generate_trajectory_plots → trajectory_plots/
│   ├── generate_grid_movies      → grid_movies/
│   └── plot_similarity_dendrogram → similarity_dendrogram.pdf
│
│
└── 05_merge_syllables.py              # Merge post-hoc (opcional, iterativo)
    ├── Editar SYLLABLES_TO_MERGE manualmente
    ├── Lee best_model de model_selection.json
    ├── load_hdf5 + generate_syllable_mapping + apply_syllable_mapping
    ├── save_hdf5 → results_merged.h5
    ├── load_and_preprocess(save_qa=False)
    ├── build_kpms_data(estimate_sigmasq=False)
    ├── generate_trajectory_plots → trajectory_plots_merged/
    ├── generate_grid_movies      → grid_movies_merged/
    └── plot_similarity_dendrogram → similarity_dendrogram_merged/


# ORDER

config.py        (nunca ejecutar, solo se edita)
    ↓
01_explore.py     (solo se ejecuta una vez)
    ↓
config.py  ← actualizar FULL_MODEL_KAPPA según kappa scan
    ↓
02_train.py
    ↓
03_select_model.py
    ↓
04_visualize.py
    ↓
05_merge_syllables.py
