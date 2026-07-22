# CLASS IMBALANCE IMPLEMENTATION REPORT

**Branch:** `feat/fold-class-weights`  
**Date:** 2026-07-22

---

## 1. État réel du pipeline avant modification

L'audit du pipeline (`src/data/pipeline.py`) a révélé les éléments suivants :
- Lecture des images avec `tf.io.read_file` et `tf.image.decode_image(channels=3)`.
- Prétraitement de DenseNet121 (`preprocess_input`) délégué au modèle, et non effectué dans le pipeline (pas de double prétraitement).
- Labels encodés en **One-hot** avec `categorical_crossentropy`.

## 2. Liste exacte des augmentations existantes

Les augmentations, appliquées *uniquement* lorsque `is_training=True`, sont au nombre de trois :
1. **Flip horizontal :** `tf.image.random_flip_left_right` (50% implicite)
2. **Flip vertical :** `tf.image.random_flip_up_down` (50% implicite)
3. **Luminosité aléatoire :** `tf.image.random_brightness` (delta max = 0.1)

Le fichier original sur disque n'est jamais modifié.

## 3. Class weights initiaux

- **Présence initiale :** AUCUNE.
- La méthode `model.fit()` était appelée sans aucun argument `class_weight`. Le déséquilibre important (muscle=9 vs oesophagus=47) était ignoré.

## 4. Oversampling initial

- **Présence initiale :** AUCUNE.
- Aucun mécanisme de sous-échantillonnage (undersampling) ou de suréchantillonnage (oversampling) n'était présent.

## 5. Problèmes découverts

1. L'absence de pondération pénalisait fortement les petites classes (muscle, jejunum, kidney), conduisant à un biais d'apprentissage en faveur des grandes classes (oesophagus, pancreas).
2. La validation croisée à 5 folds modifie légèrement la distribution du train à chaque fold. Des class weights globaux calculés sur la totalité du dataset introduiraient une fuite de données de la validation vers l'entraînement.

## 6. Fichiers modifiés

- `configs/densenet121.yaml` : Ajout de la configuration pour activer l'équilibrage (`use_class_weights: true`, `class_weight_method: balanced`).
- `src/training/train.py` : Calcul des poids avant l'entraînement de chaque fold, en ignorant le fold de validation courant. Application des poids via l'argument `class_weight` de `model.fit()` lors des phases 1 (head) et 2 (fine-tuning). Sauvegarde des poids dans un fichier JSON.
- `docs/project_guide.md` : Ajout d'une section "Class Imbalance Strategy" pour expliquer l'approche.
- `notebooks/colab/02_train_densenet121.ipynb` : Ajout d'une section "Smoke Test" pour comparer l'entraînement avec et sans class weights.

## 7. Fichiers créés

- `src/training/class_weights.py` : Module en Python pur, indépendant de TensorFlow, contenant la fonction `compute_balanced_class_weights`.
- `tests/test_class_weights.py` : Tests unitaires (données synthétiques) et d'intégration (utilisant le vrai manifest `densenet121_folds.csv`) pour valider la robustesse de l'implémentation.

## 8. Méthode de calcul des poids

Les poids sont calculés par la fonction `compute_balanced_class_weights` selon la formule standard :

`poids_classe = nombre_total_images_train / (nombre_classes × nombre_images_de_la_classe)`

Exigences respectées :
- Indépendance vis-à-vis de TensorFlow.
- Clés entières (`int`), valeurs flottantes (`float`).
- Rejet immédiat (erreur) si une classe est absente des données d'entraînement.

## 9. Poids obtenus pour chaque classe et chaque fold

| Classe | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| bladder | 1.7424 | 1.7424 | 1.5727 | 1.5727 | 1.5727 |
| brain | 1.5682 | 1.5682 | 1.7475 | 1.7475 | 1.5727 |
| cardia | 1.7424 | 1.5682 | 1.5727 | 1.5727 | 1.7475 |
| cerebellum | 1.5682 | 1.7424 | 1.7475 | 1.5727 | 1.5727 |
| epiglottis | 1.7424 | 1.7424 | 1.7475 | 1.9659 | 1.7475 |
| gland | 0.4481 | 0.4481 | 0.4494 | 0.4369 | 0.4494 |
| jejunum | 1.9602 | 1.9602 | 1.9659 | 1.9659 | 1.9659 |
| kidney | 1.7424 | 1.7424 | 1.7475 | 1.9659 | 1.7475 |
| liver | 0.4901 | 0.4901 | 0.4915 | 0.4915 | 0.4915 |
| lung | 1.7424 | 1.7424 | 1.7475 | 1.7475 | 1.9659 |
| melanoma | 1.7424 | 1.7424 | 1.5727 | 1.5727 | 1.5727 |
| muscle | **2.2403** | **1.9602** | **2.2468** | **2.2468** | **2.2468** |
| oesophagus | **0.4127** | **0.4238** | **0.4251** | **0.4139** | **0.4139** |
| pancreas | 0.4481 | 0.4481 | 0.4369 | 0.4494 | 0.4494 |
| peritoneum | 1.5682 | 1.5682 | 1.7475 | 1.7475 | 1.5727 |
| pylorus | 1.7424 | 1.5682 | 1.5727 | 1.5727 | 1.7475 |
| rectum | 1.5682 | 1.7424 | 1.7475 | 1.5727 | 1.5727 |
| spleen | 0.5808 | 0.5808 | 0.5617 | 0.5825 | 0.5825 |
| testis | 1.5682 | 1.5682 | 1.7475 | 1.7475 | 1.5727 |
| tongue | 0.4901 | 0.4901 | 0.4915 | 0.4915 | 0.4915 |
| tonsile | 1.7424 | 1.5682 | 1.5727 | 1.5727 | 1.7475 |
| umbilical-cord | 1.7424 | 1.9602 | 1.7475 | 1.7475 | 1.7475 |

*(Les classes minoritaires comme `muscle` reçoivent systématiquement un poids plus important que les classes majoritaires comme `oesophagus`)*

## 10. Preuve de l'utilisation exclusive du Train

Le test `test_uses_only_train_labels` (dans `test_class_weights.py`) vérifie explicitement que l'ensemble des identifiants des images (`image_id`) du sous-ensemble d'entraînement et de validation sont disjoints.
Dans `train.py`, les labels sont extraits par `train_df = df[df["fold"] != fold]`. L'ensemble de validation n'est donc jamais inclus.

## 11. Tests exécutés

- Compilation complète réussie (`python -m compileall src scripts`).
- Tests unitaires et d'intégration validés (`python -m pytest tests/ -v`).
- 66 tests passés avec succès.

Parmi les tests spécifiques aux class weights ajoutés :
- Les petites classes reçoivent un poids plus élevé.
- Les classes équilibrées reçoivent le même poids.
- Une erreur est générée si une classe est manquante ou num_classes = 0.
- La reproductibilité est garantie.
- Sur les folds réels, 22 classes sont calculées pour chaque fold.
- Aucune donnée NaN ou Inf n'est retournée.
- La classe `muscle` est systématiquement pondérée plus fort que `oesophagus`.

## 12. Tests non exécutés (faute de TensorFlow local)

Les tests vérifiant que le dataset TensorFlow applique ou non les augmentations (`tf.image.*`) en fonction du flag `is_training` ne sont pas testables de manière native sur un environnement local sans TensorFlow installé. Un test sur Colab sera nécessaire pour valider ces transformations.

## 13. Instructions exactes pour le test Colab

Le notebook `02_train_densenet121.ipynb` intègre désormais l'étape 9 (Smoke Test).

1. Ouvrir `02_train_densenet121.ipynb` dans Google Colab.
2. Exécuter l'ensemble des cellules jusqu'à l'étape 9.
3. Exécuter la cellule "Step 9 — Smoke Test: Class Weights Comparison".
4. Observer les journaux d'entraînement. Deux expériences rapides sur un fold seront menées :
   - L'une désactivant explicitement `use_class_weights`.
   - L'autre activant l'option avec calcul sur-le-champ.
5. Vérifier que dans le répertoire Drive de sortie, le fichier JSON de statistiques de distribution est correctement généré et loggé dans l'Expérience B.

## 14. Limites de cette stratégie

1. Les class weights pénalisent plus fortement les classes minoritaires, ce qui peut potentiellement accroître l'influence du bruit ou des labels erronés dans ces mêmes classes.
2. Un fort déséquilibre, bien que temporisé par ces poids, n'entraîne pas le réseau sur autant de variations morphologiques pour les petites classes, contrairement à de l'oversampling ou de l'augmentation synthétique plus agressive.
3. Le paramétrage d'un taux d'apprentissage optimisé globalement peut être déstabilisé si certaines mini-batchs ne contiennent que des images extrêmement pondérées.

## 15. Verdict final

**READY FOR COLAB SMOKE TEST**
L'implémentation a été testée et se limite strictement au périmètre demandé (aucune augmentation ajoutée, aucun resampling, aucun touché aux configurations non-nécessaires). L'étape suivante requiert un run court sur Colab pour s'assurer de l'adéquation au flux TensorFlow en GPU.
