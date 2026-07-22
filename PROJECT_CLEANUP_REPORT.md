# Project Cleanup Report

**Branch:** `fix/project-audit-cleanup`  
**Date:** 2026-07-22  
**Based on:** `PROJECT_AUDIT_REPORT.md` findings

---

## 1. Git Security

- Safety commit created on `main` before any changes.
- New branch `fix/project-audit-cleanup` created from that safe state.
- No images in `data/raw/` were touched: `git ls-files data/raw/` returns only `data/raw/README.md`.

---

## 2. Git Status Before Changes (on `main` before branch)

```
Modified:   configs/densenet121.yaml
Modified:   data/README.md
Deleted:    notebooks/colab/{00,01,02}_*.ipynb
Modified:   reports/figures/sample_images_grid.png
Modified:   src/data/explore_dataset.py

Untracked:  PROJECT_AUDIT_REPORT.md
             data/manifests/class_mapping.json
             data/manifests/original_22_dataset_manifest.csv
             reports/figures/class_distribution.png
             scripts/
             tests/test_nuinsseg_migration.py
```

## 3. Git Status After Cleanup (on `fix/project-audit-cleanup`, pending commit)

```
Modified:   .gitignore
Modified:   README.md
Modified:   src/data/explore_dataset.py
Modified:   src/data/pipeline.py
Modified:   src/training/train.py

Deleted:    current_tree.txt / tree_after.txt
Deleted:    data/processed/README.md
Deleted:    experiments/README.md, experiment_tracking.md
Deleted:    reports/metrics/README.md, reports/predictions/.gitkeep
Deleted:    src/utils/device.py, file_utils.py, logger.py

Untracked:  data/manifests/densenet121_folds.csv
             docs/project_guide.md
             notebooks/colab/01_dataset_preparation.ipynb
             notebooks/colab/02_train_densenet121.ipynb
             tests/test_config.py
             tests/test_pipeline_paths.py
             tests/test_real_folds_manifest.py
```

---

## 4. Anomalies Corrected

| # | Problem | Fix |
|---|---|---|
| 1 | **CRITICAL** Path resolution bug in `pipeline.py` | Rewrote pipeline with `resolve_image_path()` function |
| 2 | **CRITICAL** `densenet121_folds.csv` missing | Generated via `src/data/create_folds.py` |
| 3 | **IMPORTANT** Colab notebooks deleted on disk | Recreated `01_dataset_preparation.ipynb` and `02_train_densenet121.ipynb` |
| 4 | **IMPORTANT** `explore_dataset.py` default path pointed to legacy dataset | Fixed `DEFAULT_DATA_DIR` |
| 5 | **IMPORTANT** Split recommendation said "train/val/test split" | Updated to "5-fold cross-validation manifest" |
| 6 | **MINOR** `train.py` loaded YAML directly ignoring `src/utils/config.py` | Now uses `load_yaml()` from `src/utils/config.py` |
| 7 | **MINOR** Stub files bloating the codebase | Deleted `device.py`, `file_utils.py`, `logger.py` |
| 8 | **MINOR** Temporary artifact files at root | Deleted `current_tree.txt`, `tree_after.txt` |
| 9 | **INFO** Empty placeholder directories | Deleted `data/processed/`, `experiments/`, `reports/metrics/`, `reports/predictions/` |

---

## 5. Files Modified

| File | Change |
|---|---|
| `src/data/pipeline.py` | Full rewrite: path resolution via `resolve_image_path()`, RGBA→RGB forced, no double preprocess_input |
| `src/data/explore_dataset.py` | Default `--data-dir` path and split recommendation text |
| `src/training/train.py` | Uses `load_yaml()` from `src/utils/config.py` instead of direct `yaml.safe_load` |
| `README.md` | Full rewrite: accurate status table, real structure, team info, instructions |
| `.gitignore` | Explicit allow-list for manifests, scripts, notebooks, tests; proper exclusions for raw data and model binaries |

---

## 6. Files Created

| File | Purpose |
|---|---|
| `data/manifests/densenet121_folds.csv` | 5-fold CV assignments for 432 images |
| `notebooks/colab/01_dataset_preparation.ipynb` | Colab: mount Drive, reconstruct dataset, run EDA, create folds |
| `notebooks/colab/02_train_densenet121.ipynb` | Colab: verify GPU, load config, run training |
| `docs/project_guide.md` | Full project documentation (9 sections) |
| `tests/test_config.py` | Tests for YAML config loading |
| `tests/test_pipeline_paths.py` | Tests for path resolution correctness |
| `tests/test_real_folds_manifest.py` | Tests for real 432-image fold distribution |

---

## 7. Files Deleted

| File | Reason |
|---|---|
| `src/utils/device.py` | Docstring stub, unused, not imported anywhere |
| `src/utils/file_utils.py` | Docstring stub, unused, not imported anywhere |
| `src/utils/logger.py` | Docstring stub, unused, not imported anywhere |
| `current_tree.txt` | Temporary artifact from refactor work |
| `tree_after.txt` | Temporary artifact from refactor work |

---

## 8. Directories Deleted

| Directory | Content | Reason |
|---|---|---|
| `data/processed/` | Only `README.md` placeholder | No processed data generated |
| `experiments/` | Only `README.md` and placeholder tracking file | No real experiment logs |
| `reports/metrics/` | Only `README.md` placeholder | Actual metrics go in `reports/densenet121/metrics/` |
| `reports/predictions/` | Only `.gitkeep` | No predictions generated yet |

---

## 9. Folds Generated

```
Total images: 432
Unique image IDs: 432
Fold 0: 87 images
Fold 1: 87 images
Fold 2: 86 images
Fold 3: 86 images
Fold 4: 86 images
All 22 classes are present in all 5 folds.
```

---

## 10. Pipeline Test Results

`resolve_image_path()` was verified to:
- Correctly join relative paths with project root.
- NOT duplicate the `nuinsseg_human_22_original` prefix.
- Use absolute paths directly.
- Raise `FileNotFoundError` for missing files.

---

## 11. Compilation Result

```bash
python -m compileall src scripts
```
✅ All files compiled without syntax errors.

---

## 12. Pytest Results

```
============================= test session starts =============================
collected 30 items

tests/test_config.py::test_config_loads PASSED
tests/test_config.py::test_num_classes PASSED
tests/test_config.py::test_image_size PASSED
tests/test_config.py::test_num_folds PASSED
tests/test_config.py::test_seed PASSED
tests/test_config.py::test_dataset_root_path PASSED
tests/test_config.py::test_manifest_path PASSED
tests/test_config.py::test_folds_path PASSED
tests/test_folds.py::test_folds_range PASSED
tests/test_folds.py::test_every_image_in_one_fold PASSED
tests/test_folds.py::test_classes_represented_in_folds PASSED
tests/test_manifest.py::test_manifest_columns PASSED
tests/test_manifest.py::test_image_id_unique PASSED
tests/test_manifest.py::test_class_consistency PASSED
tests/test_nuinsseg_migration.py::test_exactly_22_classes PASSED
tests/test_nuinsseg_migration.py::test_no_mouse_or_placenta PASSED
tests/test_nuinsseg_migration.py::test_no_empty_class PASSED
tests/test_nuinsseg_migration.py::test_manifest_columns PASSED
tests/test_nuinsseg_migration.py::test_unique_image_id PASSED
tests/test_nuinsseg_migration.py::test_coherent_class_mapping PASSED
tests/test_pipeline_paths.py::test_resolve_relative_path PASSED
tests/test_pipeline_paths.py::test_resolve_no_duplicate_prefix PASSED
tests/test_pipeline_paths.py::test_resolve_absolute_path PASSED
tests/test_pipeline_paths.py::test_resolve_missing_file_raises PASSED
tests/test_real_folds_manifest.py::test_total_images PASSED
tests/test_real_folds_manifest.py::test_unique_image_ids PASSED
tests/test_real_folds_manifest.py::test_fold_values PASSED
tests/test_real_folds_manifest.py::test_no_missing_values PASSED
tests/test_real_folds_manifest.py::test_all_classes_in_all_folds PASSED
tests/test_real_folds_manifest.py::test_class_name_class_id_coherence PASSED

30 passed, 1 warning in 3.38s
```

---

## 13. Files Tracked in `data/raw/`

```bash
git ls-files data/raw/
→ data/raw/README.md
```
No images are tracked by Git. ✅

---

## 14. Colab Notebooks Status

| Notebook | Status |
|---|---|
| `notebooks/colab/01_dataset_preparation.ipynb` | ✅ Created — mounts Drive, clones repo, reconstructs dataset, runs EDA, creates folds |
| `notebooks/colab/02_train_densenet121.ipynb` | ✅ Created — verifies GPU, loads config, runs 5-fold training |

Both notebooks:
- Contain no tokens, passwords, or personal paths
- Call logic in `src/` rather than duplicating it
- Are parameterized via config cells

---

## 15. Instructions for the Teammate

Your teammate can start by:

1. Cloning the repository.
2. Downloading NuInsSeg from https://doi.org/10.5281/zenodo.10518852.
3. Running `scripts/create_original_22_class_dataset.py` with her own paths.
4. Using `data/manifests/class_mapping.json` and `original_22_dataset_manifest.csv` for her model.
5. Creating her own config YAML in `configs/` and her own model in `src/models/`.
6. **Not modifying** `configs/densenet121.yaml`, `src/models/densenet121.py`, or `data/manifests/densenet121_folds.csv`.

---

## 16. Remaining Problems

| Issue | Severity | Note |
|---|---|---|
| TF not installed locally | INFO | Expected — training runs on Colab GPU |
| `src/utils/config.py` not tested with merge (`load_config`) | MINOR | `load_yaml` tested; merge path can be added later |

---

## 17. Final Verdict

# VERDICT: **READY FOR SMALL TRAINING TEST** ✅

**Dataset:** 432 images, 22 classes, verified and intact.  
**Folds:** 5 stratified folds generated, all classes present in all folds.  
**Pipeline:** Path resolution bug fixed, RGBA→RGB handled, no double preprocessing.  
**Tests:** 30/30 passed.  
**Colab Notebooks:** Ready for execution.

Recommended next step: Run `02_train_densenet121.ipynb` on Colab with `TRAIN_SINGLE_FOLD = 0` and `head_epochs=1`, `fine_tuning_epochs=1` to validate the end-to-end pipeline before launching the full 5-fold training.
