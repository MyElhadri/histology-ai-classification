# Experiment B: Article-Inspired Classification Head Report

## Objectif
Implémenter et évaluer une tête de classification DenseNet121 inspirée de la littérature, sans perturber le comportement fondamental de la baseline et sans ajouter d'augmentation de données complexe, afin d'isoler l'impact exclusif de cette architecture sur les performances (accuracy et F1 score macro).

## Cause Exacte du Bug & Diagnostic
- **Cause du Bug** : Lors du chargement de la configuration dans `train.py` et le notebook Colab, la clé `classifier_head` était lue directement à la racine du dictionnaire `config.get("classifier_head")` au lieu de `config.get("model", {}).get("classifier_head")`. Or, dans les fichiers YAML, `classifier_head` est imbriqué sous la section `model:`.
- **Preuve de l'ancien comportement** : La fonction `config.get("classifier_head")` renvoyait `None`, déclenchant le fallback par défaut `{"type": "baseline"}`. Par conséquent, l'entraînement s'exécutait avec la tête baseline et enregistrait dans `selected_folds_summary.json` :
  - `head_type: baseline`
  - `total_params: 7060054`
- **Nombre de paramètres de l'ancien modèle** : `7 060 054` (modèle baseline avec 22 550 paramètres dans la tête).

## Architecture Réellement Construite Après Correction
Après correction de l'extraction de configuration et ajout de la séparation explicite des fonctions d'activation ELU :
- `global_average_pooling` (GlobalAveragePooling2D)
- `classifier_dense_512` (Dense 512 unités)
- `classifier_elu_512` (Activation ELU)
- `classifier_batch_norm` (BatchNormalization)
- `classifier_dropout` (Dropout 0.30)
- `classifier_dense_128` (Dense 128 unités avec L2 = 0.01)
- `classifier_elu_128` (Activation ELU)
- `predictions` (Dense 22 unités, Softmax)

- **Nombre de paramètres après correction** : `7 632 854` (DenseNet121 base : 7 037 504 + Tête inspirée de l'article : 595 350).

## Validation Runtime Obligatoire
Une fonction `validate_model_matches_config(model, config)` a été intégrée dans `src/models/densenet121.py` et appelée systématiquement avant tout `model.fit()` :
- Elle vérifie la présence explicite des couches `classifier_dense_512`, `classifier_batch_norm`, `classifier_dropout` (rate 0.30), `classifier_dense_128` (L2 = 0.01) et `predictions` (22 neurones softmax).
- Elle garantit l'absence totale de couche `Flatten`.
- En cas de non-correspondance entre le YAML et le modèle instancié, elle stoppe immédiatement l'exécution avec une erreur explicite.

## Tests Ajoutés
Une suite de 13 tests de non-régression a été intégrée dans `tests/test_model_head.py` :
1. Construction effective de la tête baseline depuis le YAML baseline.
2. Construction effective de la tête article_inspired depuis le YAML B.
3. Présence de la couche Dense 512.
4. Présence de la couche BatchNormalization.
5. Présence du Dropout 0.30.
6. Présence de la couche Dense 128 avec régularisation L2 = 0.01.
7. Présence de la couche de sortie avec 22 neurones et activation softmax.
8. Absence totale de couche Flatten.
9. Nombre de paramètres du modèle B supérieur à la baseline (7 632 854 vs 7 060 054).
10. Acceptation des modèles valides par `validate_model_matches_config`.
11. Rejet par `validate_model_matches_config` si un modèle baseline est construit alors que `article_inspired` est demandé.
12. Immuabilité de `CONFIG_PATH` dans le notebook Colab.
13. Utilisation de la politique d'augmentation `baseline` par le YAML B.

## Procédure Colab de Relance
Pour exécuter l'entraînement corrigé sur Google Colab :
1. Ouvrir le notebook `notebooks/colab/02_train_densenet121.ipynb`.
2. S'assurer que les variables dans la cellule utilisateur / chargement YAML contiennent :
   ```python
   CONFIG_PATH = "configs/experiments/densenet121_exp_b_article_head.yaml"
   RUN_MODE = "selected_folds"
   SELECTED_FOLDS = [0, 1]
   ```
3. Exécuter l'ensemble des cellules du notebook. La cellule de pré-vérification validera l'architecture (7 632 854 paramètres) avant le lancement des folds.
4. Les résultats seront enregistrés séparément dans `/content/drive/MyDrive/histology-results/densenet121-exp-b-article-head-screening`.
