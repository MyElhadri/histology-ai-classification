# Experiment D: Rich Augmentation + Article-Inspired Head Report

> [!NOTE]
> Clarification méthodologique : L'expérience D est une **expérience d'interaction entre deux facteurs déjà évalués et validés séparément** (la politique d'augmentation riche de l'Expérience A et la tête de classification inspirée de l'article de l'Expérience B). Il ne s'agit pas d'une étude d'ablation à variable unique, mais d'une étude de synergie entre régularisation des données et capacité expressive de la tête.

## Objectif
Évaluer la synergie combinée de la politique d'augmentation de données enrichie ("Rich Augmentation", issue de l'Expérience A) et de la tête de classification multicouche régularisée ("Article-Inspired Head", issue de l'Expérience B) sur le modèle DenseNet121 avec fine-tuning complet.

## Hypothèse
- L'Expérience A a montré qu'une augmentation riche régularise le pipeline et augmente l'accuracy moyenne à ~0.8908 et le macro F1 à ~0.8803.
- L'Expérience B a montré qu'une tête plus profonde avec ELU, BatchNormalization et régularisation L2 permet d'atteindre une accuracy moyenne de ~0.9080.
- L'hypothèse de l'Expérience D est que la combinaison simultanée de la régularisation de données de A et de la capacité d'apprentissage non-linéaire régularisée de B permettra d'atteindre un nouveau sommet de performance (accuracy > 0.91 et macro F1 > 0.88), réduisant au minimum le fossé d'overfitting train/val.

## Composants Repris
1. **Provenance Expérience A (Rich Augmentation)** :
   - Retournements horizontal et vertical.
   - Rotation aléatoire (`rotation_factor: 0.04`, ~±14.4°).
   - Zoom aléatoire (`zoom_factor: 0.10`).
   - Variation de luminosité (`brightness_delta: 0.05` -> ±12.75 pixels).
   - Variation de contraste (`contrast_factor: 0.10`).
   - Variation de saturation (`saturation_lower: 0.90`, `saturation_upper: 1.10`).
   - Bruit gaussien (`gaussian_noise_stddev: 0.01` -> σ ≈ 2.55 pixels).
   - Écrêtage des valeurs dans l'intervalle `[0.0, 255.0]`.

2. **Provenance Expérience B (Article-Inspired Head)** :
   - `GlobalAveragePooling2D`.
   - `Dense(512)` + `ELU` + `BatchNormalization` + `Dropout(0.30)`.
   - `Dense(128, kernel_regularizer=L2(0.01))` + `ELU`.
   - `Dense(22, activation="softmax")`.
   - Nombre de paramètres du modèle : **7 632 854**.

3. **Provenance Baseline (Fine-Tuning Full)** :
   - `fine_tuning.strategy: full` (toutes les couches convolutionnelles de la base dégelées).
   - `keep_batch_normalization_frozen: true` (BatchNormalization de la base gelées).

## Raison de l'Exclusion de l'Expérience C
L'Expérience C a testé un dégel sélectif restreint au bloc `conv5_`. Afin de ne pas introduire un 3ème facteur simultané et conserver une étude d'interaction claire et interprétable (2 facteurs combinés A + B), la stratégie de fine-tuning a été maintenue à `full` (baseline).

## Paramètres Exacts
```yaml
model:
  architecture: DenseNet121
  weights: imagenet
  dropout_rate: 0.30
  classifier_head:
    type: article_inspired
    pooling: global_average
    dense_1_units: 512
    dense_1_activation: elu
    batch_normalization: true
    dropout_rate: 0.30
    dense_2_units: 128
    dense_2_activation: elu
    l2_strength: 0.01
    output_activation: softmax

augmentation:
  enabled: true
  policy: rich
  horizontal_flip: true
  vertical_flip: true
  rotation_factor: 0.04
  zoom_factor: 0.10
  brightness_delta: 0.05
  contrast_factor: 0.10
  saturation_lower: 0.90
  saturation_upper: 1.10
  gaussian_noise_stddev: 0.01
  clip_min: 0.0
  clip_max: 255.0

fine_tuning:
  strategy: full
  keep_batch_normalization_frozen: true
```

## Fichiers Modifiés & Créés
- `configs/experiments/densenet121_exp_d_rich_aug_article_head.yaml` : Fichier de configuration combiné de l'Expérience D.
- `notebooks/colab/02_train_densenet121.ipynb` : Intégration du chemin de config D et de la sortie `/content/drive/MyDrive/histology-results/densenet121-exp-d-rich-aug-article-head-screening`.
- `tests/test_exp_d.py` : Création de la suite de 15 tests automatisés.
- `EXPERIMENT_D_RICH_AUG_ARTICLE_HEAD_REPORT.md` : Rédaction du présent rapport.

## Tests Réalisés
La suite de tests automatisés sous `pytest` valide 15 points clés :
1. Chargement de la tête `article_inspired` depuis le YAML D.
2. Chargement de la politique `rich` depuis le YAML D.
3. Chargement de la stratégie `full` depuis le YAML D.
4. Maintien du gel des couches `BatchNormalization` de la base pendant le fine-tuning.
5. Absence totale d'augmentation sur le dataset de validation.
6. Présence de toutes les transformations `rich` sur le dataset d'entraînement.
7. Présence de la couche `Dense 512`.
8. Présence de la couche `BatchNormalization`.
9. Présence du `Dropout 0.30`.
10. Présence de la couche `Dense 128` avec `L2 = 0.01`.
11. Présence de la couche de sortie `Dense 22` softmax.
12. Total de paramètres exactement égal à `7 632 854`.
13. Absence de la stratégie `last_dense_block` dans le YAML D.
14. Exécution d'un forward pass sur tenseur factice sans erreur.
15. Immuabilité et intégrité des fichiers de configuration des Expériences A, B, C et Baseline.

## Procédure Colab
1. Ouvrir le notebook `notebooks/colab/02_train_densenet121.ipynb`.
2. S'assurer que les variables utilisateur pointent vers l'Expérience D :
   ```python
   CONFIG_PATH = "configs/experiments/densenet121_exp_d_rich_aug_article_head.yaml"
   RUN_MODE = "selected_folds"
   SELECTED_FOLDS = [0, 1]
   ```
3. Exécuter la cellule de pré-vérification. Elle doit afficher :
   - `Requested head: article_inspired`
   - `Actual head: article_inspired`
   - `Augmentation: rich`
   - `Requested fine-tuning: full`
   - `Actual fine-tuning: full`
4. Lancer l'entraînement. Les résultats seront enregistrés séparément dans :
   `/content/drive/MyDrive/histology-results/densenet121-exp-d-rich-aug-article-head-screening`.

## Limites
- Combiner deux régularisations fortes (augmentation riche + régularisation L2/Dropout dans la tête) pourrait théoriquement sous-ajuster si la capacité du réseau devenait trop restreinte, bien que la profondeur de la tête (512 -> 128) compense cette contrainte.

## Critères de Décision
L'expérience D sera considérée comme un succès majeur si elle surpasse individuellement l'Expérience A (macro F1 > 0.88) et l'Expérience B (accuracy > 0.908) sur les folds 0 et 1.
