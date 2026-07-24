# Experiment E1: Regularized Representation Learning + Balanced Classifier Retraining (cRT) Report

> [!NOTE]
> Clarification méthodologique : L'Expérience E1 combine plusieurs techniques de régularisation et de rééquilibrage découplé (Label Smoothing 0.05, Apprentissage de représentation en distribution naturelle, et Classifier Retraining à échantillonnage Square-Root). Il ne s'agit donc pas d'une étude d'ablation à variable unique, mais d'un pipeline complet en 3 phases pour contrer l'overfitting et l'instabilité sur les classes minoritaires. De plus, cette évaluation est exploratoire puisque les 5 folds ont déjà participé aux choix d'hyperparamètres antérieurs.

## 1. Diagnostic de l'Expérience D
Bien que l'Expérience D (Rich Augmentation + Article-Inspired Head) ait permis d'obtenir une accuracy moyenne de ~0.8841 et un macro F1 de ~0.8280, les observations montrent :
1. **Surapprentissage & Surconfiance** : L'accuracy d'entraînement atteint fréquemment 98 % à 100 %, alors que l'accuracy de validation plafonne aux alentours de 88 %.
2. **Instabilité des classes minoritaires** : Certaines classes sous-représentées (notamment `bladder`, `cardia`, `lung`, `muscle`, `rectum`) présentent un F1-score fragile ou nul selon les folds.

## 2. Objectif de l'Expérience E1
L'objectif est double :
1. **Régulariser les représentations** en éliminant les pénalités de fréquences excessives pendant le fine-tuning du backbone (aucun class weight, distribution naturelle, label smoothing = 0.05).
2. **Rééquilibrer le classifieur (cRT)** en 3ème phase par un re-sampleur square-root modéré sur la couche de sortie uniquement, sans modifier les caractéristiques apprises par le backbone.

## 3. Pipeline d'Entraînement en 3 Phases

| Phase | Description | Sampling | Class Weights | Label Smoothing | Layers Entraînables | Params Entraînables |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Head Training | Naturel | Désactivés | 0.05 | Tête `article_inspired` | **594 326** |
| **Phase 2** | Fine-Tuning Représentation | Naturel | Désactivés | 0.05 | Full backbone (BN gelées) + Tête | **7 632 854** |
| **Phase 3** | Classifier Retraining (cRT) | Square-Root | Désactivés | 0.05 | Couche `predictions` uniquement | **2 838** |

## 4. Différence entre `val_loss` et `val_ce_hard`
- `val_loss` : Calculée avec `CategoricalCrossentropy(label_smoothing=0.05)` et inclut les pertes de régularisation L2 issues des couches de la tête.
- `val_ce_hard` : Calculée de façon pure avec `CategoricalCrossentropy(label_smoothing=0.0)` sur des étiquettes dures (0/1), sans aucune pénalité L2 ni lissage.
- Ainsi, `val_loss != val_ce_hard`. Les callbacks de l'Expérience E1 surveillent strictement `val_ce_hard` pour sélectionner les meilleurs poids.

## 5. Principe du Square-Root Sampling (Phase 3)
- **Formule** : Chaque classe $c$ contenant $n_c$ images a une probabilité d'échantillonnage proportionnelle à $\sqrt{n_c}$ :
  $$P(c) = \frac{\sqrt{n_c}}{\sum_j \sqrt{n_j}}$$
- **Modération** : Pour un ratio de classe majoritaire/minoritaire de $38 / 7 = 5.43$, le ratio square-root devient $\sqrt{38}/\sqrt{7} \approx 2.33$. Cela évite le sur-échantillonnage agressif qui dégraderait la représentation.
- **Dynamic tf.data** : Échantillonnage avec remplacement via `tf.data.Dataset.sample_from_datasets` sans aucune duplication physique de fichiers sur disque.
- **Taille de l'époque** : Égale au nombre d'images d'entraînement original du fold ($N$).
- **Garantie Validation** : Le dataset de validation demeure strictement en distribution naturelle, non augmenté.

## 6. Réinitialisation de la Couche `predictions`
Avant la Phase 3 :
- Le backbone et la tête cachée (512, BN, Dropout, 128) sont totalement gelés.
- Les poids (`kernel` et `bias`) de la couche `predictions` ($128 \times 22 + 22 = 2838$ paramètres) sont réinitialisés avec leurs initialiseurs d'origine.
- L'optimiseur Adam est instancié à neuf avec un learning rate de 0.0003.

## 7. Tests & Validation
Une suite complète de tests validés avec `pytest` garantit 30 points clés, notamment :
- Conformité YAML E1 (`article_inspired`, `rich`, `full`, `label_smoothing=0.05`).
- Calcul exact du ratio square-root (38 vs 7 $\rightarrow$ 2.33).
- Strictement 2838 paramètres entraînables en Phase 3.
- Immuabilité des configurations des expériences Baseline, A, B, C et D.
- Sérialisabilité native JSON de tous les types NumPy/TF.

## 8. Procédure Colab
1. Ouvrir `notebooks/colab/02_train_densenet121.ipynb`.
2. Définir :
   ```python
   CONFIG_PATH = "configs/experiments/densenet121_exp_e1_regularized_crt.yaml"
   RUN_MODE = "selected_folds"
   SELECTED_FOLDS = [0, 3, 4]
   SEED_OVERRIDE = 42
   ```
3. Les résultats seront sauvegardés séparément dans :
   `/content/drive/MyDrive/histology-results/densenet121-exp-e1-regularized-crt-screening-seed42`

## 9. Limites & Critères de Décision
- **Limites** : Évaluation exploratoire réalisée sur les mêmes 5 folds. Le réentraînement cRT ajuste uniquement la frontière linéaire finale.
- **Succès** : L'expérience sera jugée fructueuse si elle améliore la métrique `minority_macro_f1` et réduit `zero_f1_class_count` tout en maintenant ou améliorant le `macro_f1` global par rapport à l'Expérience D.
