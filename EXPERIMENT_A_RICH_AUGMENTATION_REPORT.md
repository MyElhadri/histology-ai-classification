# Experiment A: Rich Augmentation Report

## Objectif
Implémenter l'Expérience A ("Rich Data Augmentation") visant à évaluer l'impact d'une politique d'augmentation de données plus riche et agressive sur les performances de classification du modèle DenseNet121, en particulier pour lutter contre l'overfitting.

## Baseline
La baseline (`densenet121.yaml`) effectue une validation croisée à 5 folds avec une augmentation de données très basique :
- Retournement horizontal (`random_flip_left_right`)
- Retournement vertical (`random_flip_up_down`)
- Modification de luminosité (`random_brightness` avec `max_delta=0.1` qui, sur des images dans l'intervalle [0, 255], a un impact négligeable de 0.04%).
- L'accuracy moyenne est d'environ 0.8449 et le macro F1 d'environ 0.7728.

## Seule Variable Modifiée
La **seule variable modifiée** est la politique d'augmentation des données (`augmentation`). L'architecture du modèle, les hyperparamètres (optimiseur, loss, taux d'apprentissage, epochs, class weights), la gestion des folds et les callbacks demeurent strictement identiques à la baseline.

## Paramètres Exacts
La configuration suivante a été ajoutée dans `densenet121_exp_a_rich_augmentation.yaml` :
```yaml
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
```

## Justification des Valeurs
- **rotation_factor (0.04)** : Correspond à une rotation d'environ ±14.4 degrés, ce qui introduit une variation d'orientation réaliste pour des images histologiques sans dénaturer la structure globale.
- **zoom_factor (0.10)** : Permet un zoom aléatoire jusqu'à 10%, simulant des variations de grossissement de numérisation ou des légers décalages d'échelle.
- **brightness_delta (0.05)** : Modification maximale de ±12.75 pixels (5% de 255), introduisant une variation de l'intensité de la coloration plus prononcée que la baseline tout en restant dans des limites physiquement plausibles.
- **contrast_factor (0.10)** : Altération du contraste de ±10% pour forcer le modèle à être invariant aux variations de préparation des lames.
- **saturation (0.90 à 1.10)** : Modification de la saturation colorimétrique simulant les différences d'intensité de la coloration H&E (hématoxyline-éosine).
- **gaussian_noise_stddev (0.01)** : Bruit gaussien de faible amplitude (σ ≈ 2.55 pixels) pour régulariser le réseau (réduction de la confiance excessive).

## Fichiers Modifiés
- `src/data/pipeline.py` : Introduction de la classe `RichAugmentation` et mise à jour de `create_dataset`.
- `src/training/train.py` : Passage de la configuration d'augmentation au `train_dataset`.
- `configs/experiments/densenet121_exp_a_rich_augmentation.yaml` : Création du nouveau fichier de configuration expérimental.
- `notebooks/colab/02_train_densenet121.ipynb` : Intégration du mode `selected_folds` (avec `SELECTED_FOLDS = [0, 1]`) et de la sauvegarde détaillée dans un dossier dédié.
- `tests/test_pipeline_augmentation.py` : Ajout de tests automatisés vérifiant le bon fonctionnement du pipeline augmenté.

## Tests
- Des tests unitaires ont été implémentés sous `pytest`. Ils valident la conservation de la baseline si l'expérience n'est pas activée, la désactivation de l'augmentation sur le set de validation, et la conformité des tenseurs générés (shape `224x224x3`, type `float32`, valeurs finies et strictement contenues dans `[0, 255]`).

## Procédure Colab
Le mode `selected_folds` entraîne itérativement les sous-ensembles ciblés (folds 0 et 1).
La mémoire est libérée (`tf.keras.backend.clear_session()` et `gc.collect()`) entre les entraînements de folds.
Les résultats (modèles, historiques, métriques et le résumé global) sont enregistrés sans écraser la baseline, dans le répertoire de Google Drive : `/content/drive/MyDrive/histology-results/densenet121-exp-a-rich-augmentation-screening`.

## Limites
- Cette politique d'augmentation "riche" peut potentiellement ralentir légèrement le chargement du pipeline `tf.data`, bien que compensée par `tf.data.AUTOTUNE`.
- Les valeurs de seuils ont été déterminées de façon heuristique et pourraient nécessiter un réglage fin selon les résultats obtenus.
