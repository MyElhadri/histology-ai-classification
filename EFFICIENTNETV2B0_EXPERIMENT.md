# EfficientNetV2B0 Experiment A — Scientific Protocol & Ensemble Preparation

This document outlines the architectural justification, scientific protocol, preprocessing specifications, two-phase training strategy, and ensemble learning preparation for **EfficientNetV2B0 Expérience A** — the second model implemented in our histology image classification project.

---

## 1. Architectural Justification & Selection

### Why EfficientNetV2B0?
To construct a highly performant and robust multi-model ensemble, individual constituent models must possess strong predictive capabilities while offering **diverse inductive biases**. 
- **DenseNet121 (Baseline - Exp D)** relies on dense feature reuse via concatenation across layers, maintaining feature maps of varying abstraction levels throughout dense blocks.
- **EfficientNetV2B0** introduces an orthogonal architectural design based on **Fused-MBConv** and **MBConv** (Mobile Inverted Bottleneck Convolution) blocks with squeeze-and-excitation optimization and compound scaling.

By combining the feature-reuse paradigm of DenseNet121 with the modern inverted bottleneck architecture of EfficientNetV2B0, we maximize representation diversity, reducing shared structural error modes and enhancing ensemble complementarity.

---

## 2. Fair Comparison Protocol

To ensure that performance differences between DenseNet121 and EfficientNetV2B0 reflect true architectural merit rather than experimental variance or hyperparameter tuning, **Expérience A enforces a strict fair comparison protocol** against the archived **DenseNet121 Exp D baseline**:

| Parameter / Protocol | DenseNet121 Exp D (Archived Baseline) | EfficientNetV2B0 Exp A (Fair Comparison) | Status / Rationale |
| :--- | :--- | :--- | :--- |
| **Dataset Images** | 432 original human histology images | 432 original human histology images | **Identical** |
| **Class Count & Mapping** | 22 classes (`class_mapping.json`) | 22 classes (`class_mapping.json`) | **Identical** |
| **CV Manifest & Folds** | Stratified 5-fold (`densenet121_folds.csv`) | Stratified 5-fold (`densenet121_folds.csv`) | **Identical** (Folds: 87, 87, 86, 86, 86) |
| **Random Seed** | 42 (Global NumPy/TF/Python seed) | 42 (Global NumPy/TF/Python seed) | **Identical** |
| **Image Resolution** | 224 × 224 × 3 | 224 × 224 × 3 | **Identical** |
| **Data Augmentation** | `rich` online training augmentation only | `rich` online training augmentation only | **Identical** (Flips, Rot ±0.04, Zoom 0.10, Brightness 0.05, Contrast 0.10, Saturation 0.9–1.1, Gaussian Noise σ=0.01, Clip [0, 255]) |
| **Validation Augmentation** | None (Raw original images) | None (Raw original images) | **Identical** |
| **Class Weights** | Balanced per-fold dictionary | Balanced per-fold dictionary | **Identical** (Verified against archived Exp D fold JSONs; aborts on mismatch) |
| **Classification Head** | `article_inspired` (512 ELU -> BN -> Drop 0.3 -> 128 ELU L2=0.01 -> 22 Softmax) | `article_inspired` (Shared architecture via `src.models.heads`, 100% parity verified) | **Identical** |
| **Phase 1 Training** | Head only, 10 epochs, LR = 0.001 (Adam) | Head only, 10 epochs, LR = 0.001 (Adam) | **Identical** |
| **Phase 2 Fine-Tuning** | Full backbone, 40 epochs, LR = 1e-5 (Adam) | Full backbone, 40 epochs, LR = 1e-5 (Adam) | **Identical** |
| **Callbacks (Phase 2)** | Monitor: `val_ce_hard` (min), EarlyStopping (patience=5, min_delta=0.002, restore_best=True), ReduceLROnPlateau (factor=0.2, patience=2, min_lr=1e-7) | Monitor: `val_ce_hard` (min), EarlyStopping (patience=5, min_delta=0.002, restore_best=True), ReduceLROnPlateau (factor=0.2, patience=2, min_lr=1e-7) | **Identical** (Replicated exactly from training audit) |
| **Backbone Architecture** | DenseNet121 | EfficientNetV2B0 | *Intentional Difference* (Model evaluation) |
| **Input Preprocessing** | External Torch/ImageNet normalization | Native `[0, 255]` float32 internal rescaling | *Intentional Difference* (Model-specific requirement) |

---

## 3. Single-Pass Preprocessing Specification

A critical technical divergence between DenseNet121 and EfficientNetV2B0 lies in input tensor scaling:
- **DenseNet121** expects external preprocessing (`preprocess_input` or standard ImageNet zero-mean/unit-variance normalization).
- **EfficientNetV2B0** is instantiated with `include_preprocessing=True`, embedding internal `rescaling` ($x / 255.0$) and `normalization` layers directly at the base of the computational graph.

### Mandatory Preprocessing Rules:
1. **Input Tensor Scale**: Images from `tf.data.Dataset` must enter the model as `float32` tensors in the native range **`[0.0, 255.0]`**.
2. **No Double Preprocessing**: No external division by 255, no external `Rescaling` layer, and no manual normalization may be applied in the pipeline or outer Keras model.
3. **Automated Verification**:
   - `verify_no_double_preprocessing(model)`: Asserts internal rescaling/normalization layers exist inside the backbone and scans outer layers to reject any external Rescaling or preprocessing layers.
   - `validate_dataset_preprocessing(dataset, config)`: Inspects real batches from the tf.data pipeline to assert float32 dtype, confirm pixel values span up to ~255.0 (max > 1.0), and reject illegal YAML config parameters (`rescale: 1/255`, `normalization: [0, 1]`).

---

## 4. Two-Phase Training & BatchNormalization Freezing

To stabilize representation learning when transferring from ImageNet to histology textures, training proceeds in two distinct phases:

### Phase 1: Classifier Head Warmup (Epochs 1–10)
- **Backbone State**: Frozen (`backbone.trainable = False`).
- **Learning Rate**: $1 \times 10^{-3}$ (Adam optimizer).
- **Objective**: Train the newly initialized `article_inspired` dense layers without distorting pre-trained ImageNet feature extractors.

### Phase 2: Representation Fine-Tuning (Epochs 11–50)
- **Backbone State**: Unfreezed (`backbone.trainable = True`, strategy: `full`).
- **BatchNormalization Freezing**: All `BatchNormalization` layers within the backbone remain strictly frozen (`layer.trainable = False`). Furthermore, the backbone is explicitly called with `x = backbone(inputs, training=False)` during model construction. This guarantees that BatchNormalization layers operate in inference mode, using frozen ImageNet moving averages rather than noisy mini-batch statistics (crucial for batch size 16).
- **Model Re-compilation**: Following Keras requirements, `model.compile()` is explicitly re-invoked after modifying layer trainability, applying the low fine-tuning learning rate ($1 \times 10^{-5}$).

---

## 5. Colab Screening Execution

In compliance with project rules, **no automated training is executed during implementation or testing**. 
The initial scientific validation (screening phase) will evaluate **Folds 0, 3, and 4** on Google Colab using the following command:

```bash
python scripts/train_efficientnetv2b0.py \
  --config configs/experiments/efficientnetv2b0_exp_a_fair_comparison.yaml \
  --output-dir /content/drive/MyDrive/histology-results/efficientnetv2b0-exp-a-fair-comparison \
  --folds 0 3 4
```

### Screening Decision Criteria
Upon completion of folds 0, 3, and 4, the script automatically generates `reports/efficientnetv2b0/efficientnetv2b0_screening_summary.json`. Screening is deemed positive if the average Macro F1 and Accuracy across these three folds meet or exceed the reference DenseNet121 D screening baseline (~0.8610 Accuracy, ~0.7889 Macro F1).

---

## 6. Ensemble Learning Preparation

To prepare for future multi-model ensembling (DenseNet121 + EfficientNetV2B0 + Partner ResNet50), `train_efficientnetv2b0.py` exports Out-Of-Fold (OOF) prediction CSVs for each fold (`reports/efficientnetv2b0/predictions/fold_{k}_oof_predictions.csv`), recording true labels, predicted labels, correctness flags, and full probability distributions (`prob_0` to `prob_21`).

We provide `scripts/compare_densenet_efficientnet_oof.py` to analyze model complementarity once OOF predictions are generated:

```bash
python scripts/compare_densenet_efficientnet_oof.py \
  --densenet-oof artifacts/models/densenet121_exp_d_v1/evaluation/oof_predictions_without_tta.csv \
  --efficientnet-oof-dir reports/efficientnetv2b0/predictions \
  --output reports/efficientnetv2b0/densenet_efficientnet_complementarity.json
```

### Complementarity Metrics Evaluated:
1. **Error Intersection**: Quantifies samples where both models succeed, both fail, or only one succeeds (revealing unique problem-solving capabilities).
2. **Agreement Rate & Disagreement Rate**: Measures the percentage of identical class predictions across the dataset.
3. **Probability Correlation**: Pearson correlation across class probability distributions, indicating whether models assign similar confidence scores.
4. **Simple 50/50 Probability Ensemble**: Computes average predicted probabilities $P_{ens} = \frac{1}{2}(P_{dense} + P_{eff})$ and reports overall Accuracy, Macro F1, and Weighted F1 gains over the best individual model.
