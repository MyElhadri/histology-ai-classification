# Experiment C: Selective Unfreezing of the Last DenseNet121 Block Report

> [!NOTE]
> Clarification méthodologique : L'article scientifique de référence évoque le principe du dégel sélectif de blocs profonds mais ne précise pas les couches exactes ni les identifiants Keras précis. Cette expérience est donc qualifiée d'**article-inspired** et ne constitue pas une reproduction stricte au bit près de la publication.

## Objectif
Évaluer l'impact du dégel sélectif restreint au dernier bloc dense (`conv5_*`) de DenseNet121 durant la phase de fine-tuning. L'objectif est d'isoler cette stratégie de fine-tuning du backbone comme unique variable expérimentale, en maintenant la tête baseline et l'augmentation baseline.

## Hypothèse
Dégeler uniquement le dernier bloc dense (`conv5_*`) tout en conservant les premiers blocs (`conv1_*` à `conv4_*`) gelés permet d'adapter les caractéristiques abstraites de haut niveau aux spécificités du tissu histologique, sans risquer de détruire les filtres génériques de bas niveau appris sur ImageNet ni de provoquer un sur-apprentissage catastrophique sur un dataset de taille modérée (432 images).

## Comportement Baseline
- **Phase Tête** : Base DenseNet121 entièrement gelée (`trainable = False`), tête baseline seule entraînable.
- **Phase Fine-Tuning Baseline (`strategy: full`)** : Toutes les couches de la base (blocs 1 à 5) sont dégelées (`trainable = True`), à l'exception des couches `BatchNormalization` de la base qui demeurent gelées.

## Stratégie C (`strategy: last_dense_block`)
- **Phase Tête** : Identique à la baseline (base entièrement gelée).
- **Phase Fine-Tuning C** :
  - Seules les couches dont le nom commence par le préfixe `conv5_` sont entraînables.
  - Les couches des blocs `conv1_`, `conv2_`, `conv3_` et `conv4_` restent strictement gelées.
  - Toutes les couches `BatchNormalization` de la base (y compris celles situées dans `conv5_`) restent gelées (`keep_batch_normalization_frozen: true`).
  - La tête de classification baseline reste entraînable.

## Raison du Gel des BatchNormalization
Conserver les couches `BatchNormalization` gelées pendant le fine-tuning évite de destabiliser les moyennes et variances glissantes calculées sur ImageNet lorsque l'entraînement s'effectue avec des mini-batchs de petite taille (batch size 16).

## Nombre de Couches et Paramètres Entraînables
- **Phase Tête** :
  - Couches backbone entraînables : 0
  - Paramètres entraînables (tête baseline) : 22 550
- **Phase Fine-Tuning C (`last_dense_block`)** :
  - Couches backbone entraînables : 128 (toutes sous le préfixe `conv5_`)
  - Couches backbone gelées : 299 (`conv1_` à `conv4_` + toutes les BatchNormalization)
  - Paramètres entraînables backbone : 4 485 120
  - Paramètres entraînables totaux : 4 507 670 (vs 7 060 054 pour un fine-tuning complet)

## Fichiers Modifiés
- `configs/densenet121.yaml` : Ajout de la section `fine_tuning: { strategy: full }` pour préserver le comportement historique.
- `configs/experiments/densenet121_exp_c_selective_unfreezing.yaml` : Fichier de configuration de l'expérience C (`strategy: last_dense_block`).
- `src/models/densenet121.py` : Implémentation des fonctions `apply_fine_tuning_strategy` et `validate_fine_tuning_strategy`.
- `src/training/train.py` : Intégration de l'application et de la validation runtime de la stratégie de fine-tuning, avec journalisation exhaustive des couches et paramètres.
- `notebooks/colab/02_train_densenet121.ipynb` : Intégration des pré-vérifications et sauvegarde des métadonnées de fine-tuning dans `selected_folds_summary.json`.
- `tests/test_fine_tuning.py` : Création de la suite de 14 tests de non-régression.

## Tests Réalisés
La suite de tests automatisés sous `pytest` valide 14 points clés :
1. Conservation du comportement baseline avec `strategy: full`.
2. Dégel effectif des couches `conv5_*` non-BN sous `last_dense_block`.
3. Maintien strict du gel des blocs `conv1_` à `conv4_`.
4. Maintien du gel des couches `BatchNormalization` de la base.
5. Maintien de l'entraînabilité de la tête baseline.
6. Absence de la tête article-inspired (Expérience B).
7. Recompilation du modèle après changement de trainabilité.
8. Levée d'erreur pour une stratégie inconnue.
9. Levée d'erreur pour une liste de préfixes vide.
10. Validation runtime acceptant la configuration C correcte.
11. Rejet par la validation runtime si un dégel complet est appliqué à la place de `last_dense_block`.
12. Utilisation de la politique d'augmentation baseline par la config C.
13. Utilisation de la tête baseline par la config C.
14. Compatibilité de la sérialisation JSON avec les types NumPy int64/float64.

## Procédure Colab
1. Ouvrir le notebook `notebooks/colab/02_train_densenet121.ipynb`.
2. Vérifier les variables de configuration utilisateur :
   ```python
   CONFIG_PATH = "configs/experiments/densenet121_exp_c_selective_unfreezing.yaml"
   RUN_MODE = "selected_folds"
   SELECTED_FOLDS = [0, 1]
   ```
3. La cellule de pré-vérification doit afficher :
   - `Requested head: baseline`
   - `Actual head: baseline`
   - `Augmentation: baseline`
   - `Requested fine-tuning: last_dense_block`
   - `Actual fine-tuning: last_dense_block`
4. Exécuter l'entraînement. Les métriques et récapitulatifs seront enregistrés dans :
   `/content/drive/MyDrive/histology-results/densenet121-exp-c-selective-unfreezing-screening`.

## Limites
- Seul le bloc `conv5_` est dégelé. Les blocs intermédiaires (`conv4_`) ne sont pas testés dans cette expérience afin de respecter le principe de variable scientifique unique.

## Critères de Décision
L'expérience C sera considérée comme supérieure à la baseline si l'accuracy ou le macro F1 moyen sur les 2 folds s'améliore, tout en réduisant l'écart d'overfitting entre l'entraînement et la validation.
