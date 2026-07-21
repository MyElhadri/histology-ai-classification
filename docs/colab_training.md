# Google Colab Training Guide

> Step-by-step instructions for running the histology classification
> pipeline on Google Colab.

## 1. Activate a GPU Runtime

1. Open Google Colab: <https://colab.research.google.com/>
2. Go to **Runtime → Change runtime type**
3. Select **GPU** (T4 is sufficient for DenseNet121)
4. Click **Save**

Verify with:
```python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
```

## 2. Mount Google Drive

Google Drive is used for persistent storage of:
- The raw dataset
- Trained model checkpoints
- Saved models
- Metrics and figures

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 3. Dataset Placement

Place the dataset in your Google Drive, for example:

```
My Drive/
└── histology_project/
    ├── dataset/
    │   └── Human_Histopathological_H_E_Stained_Nuclei_Images/
    │       ├── human_tissue_image-bladder/
    │       ├── human_tissue_image-brain/
    │       └── ... (22 class directories)
    └── outputs/
        ├── data/manifests/
        ├── models/
        └── reports/
```

> **Important:** Each team member should adjust the paths in the
> notebook's configuration cell to match their own Drive structure.

## 4. Launch the Notebook

Open `notebooks/colab/00_setup_and_dataset_audit.ipynb` in Colab.

The notebook will:
1. Clone the GitHub repository (or pull updates)
2. Install additional dependencies
3. Run the dataset audit pipeline
4. Display results inline

## 5. Where Results Are Saved

| Output | Location |
|---|---|
| Dataset manifest | `<output_root>/data/manifests/dataset_manifest.csv` |
| Class mapping | `<output_root>/data/manifests/class_mapping.json` |
| Test manifest (shared) | `<output_root>/data/manifests/test_manifest.csv` |
| DenseNet121 folds | `<output_root>/data/manifests/densenet121_folds.csv` |
| Audit report | `<output_root>/reports/metrics/dataset_audit.json` |
| Class distribution plot | `<output_root>/reports/figures/class_distribution.png` |

## 6. Why `/content/` Files Are Temporary

Google Colab runs on ephemeral virtual machines. **Any file stored
directly in `/content/` is deleted when the runtime disconnects.**

This is why we save all important outputs to Google Drive:
- Drive is persistent across sessions
- Drive is accessible from any machine
- Drive can be shared with team members

Only the cloned repository code lives in `/content/` — it is re-cloned
at the start of each session.

## 7. Resuming Work

When you reconnect to Colab:
1. Re-mount Google Drive
2. The repo will be re-cloned automatically
3. Your outputs in Drive are still there
4. Re-run only the cells you need
