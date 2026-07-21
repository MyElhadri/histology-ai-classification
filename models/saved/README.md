# Models — Saved

Final trained model files.

## Naming Convention

```
<architecture>_<dataset>_<date>_<metric>.keras
```

Example: `densenet121_histology_20260721_acc95.keras`

## Formats

| Format | Extension | Use Case |
|---|---|---|
| Keras | `.keras` | Primary format for training and evaluation |
| HDF5 | `.h5` | Legacy Keras format |
| SavedModel | directory | TensorFlow Serving deployment |

> This directory is **gitignored** for large model files. Track model metadata in `experiments/`.
