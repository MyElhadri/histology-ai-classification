# InceptionResNetV2 Experiment A — High Performance Histology

> **Avertissement éthique** : Ce système est à visée pédagogique et expérimentale uniquement.
> Il ne constitue en aucun cas un outil de diagnostic clinique validé.
> Les résultats ne doivent pas être utilisés pour des décisions médicales.

---

## 1. Justification du choix InceptionResNetV2

InceptionResNetV2 est choisi comme troisième architecture candidate (après DenseNet121 et InceptionV3)
pour les raisons suivantes :

- **Combinaison Inception + Résiduel** : associe la puissance des blocs Inception multi-échelle
  à l'optimisation par connexions résiduelles de ResNet
- **Capacité de représentation supérieure** : plus profond qu'InceptionV3, potentiellement meilleur
  pour discriminer les 22 classes histologiques fines
- **Transfer learning éprouvé** : pré-entraîné sur ImageNet-1k (~1,2 million d'images)
- **Complémentarité attendue** : erreurs différentes de DenseNet121, potentiel d'ensemble

---

## 2. Différences entre InceptionV3 et InceptionResNetV2

| Propriété | InceptionV3 | InceptionResNetV2 |
|---|---|---|
| Architecture | Blocs Inception empilés | Blocs Inception + connexions résiduelles |
| Profondeur | ~159 couches | ~572 couches |
| Paramètres | ~24M | ~56M |
| Résolution native | 299×299 | 299×299 |
| Preprocessing | [-1, 1] via Rescaling | [-1, 1] via Rescaling |
| Précision ImageNet top-1 | ~78% | ~80% |

---

## 3. Architecture Inception + Connexions Résiduelles

InceptionResNetV2 ajoute des connexions résiduelles (skip connections) à chaque bloc Inception :

```
Input
  ↓
Stem (conv layers)
  ↓
Inception-ResNet-A blocks (×5) — petite échelle
  ↓
Reduction-A
  ↓
Inception-ResNet-B blocks (×10) — échelle moyenne
  ↓
Reduction-B
  ↓
Inception-ResNet-C blocks (×5) — grande échelle
  ↓
GlobalAveragePooling
  ↓
[Tête article_inspired DenseNet Exp D]
```

**Connexion résiduelle** : `output = block(x) + shortcut(x)`
Cela accélère la convergence et permet d'entraîner des réseaux très profonds.

---

## 4. Résolution 299 × 299

InceptionResNetV2 a été conçu pour une entrée de 299×299×3.
Utiliser cette résolution native :
- Preserve la capacité représentationnelle maximale du réseau
- Évite l'upsampling ou downsampling non nécessaires
- Correspond aux poids ImageNet appris à cette résolution

**Impact** : Plus grande résolution que DenseNet121 (224×224) → plus d'information spatiale
disponible, potentiellement utile pour les textures histologiques fines.

---

## 5. Preprocessing [-1, 1]

InceptionResNetV2 attend des entrées dans [-1, 1] (identique à InceptionV3).

**Implémentation** :
- Une seule couche sérialisable dans le modèle :
  ```python
  tf.keras.layers.Rescaling(scale=1.0/127.5, offset=-1.0, name="inception_resnet_v2_preprocessing")
  ```
- Transformation : `output = input * (1/127.5) - 1.0`
- Valeurs clés : 0 → -1, 127.5 → 0, 255 → +1
- Numériquement équivalente à `tf.keras.applications.inception_resnet_v2.preprocess_input`

**Contrat strict** :
- Pipeline externe → `float32`, valeurs dans `[0, 255]`
- Aucune division par 255 en externe
- Aucun `preprocess_input` externe
- Aucune double normalisation

---

## 6. Tête de Classification Réutilisée

La tête `article_inspired` de DenseNet121 Exp D est réutilisée à l'identique :

```
GlobalAveragePooling2D
  ↓
Dense(512)
  ↓
ELU
  ↓
BatchNormalization
  ↓
Dropout(0.30)
  ↓
Dense(128, L2=0.01)
  ↓
ELU
  ↓
Dense(22, softmax, dtype=float32)
```

Paramètres exacts :
- `dense_1_units = 512`
- `dense_1_activation = elu`
- `batch_normalization = True`
- `dropout_rate = 0.30`
- `dense_2_units = 128`
- `dense_2_activation = elu`
- `l2_strength = 0.01`
- `output_activation = softmax`

**Justification** : réutiliser la même tête minimise les variables changeantes entre expériences
et permet une comparaison équitable centrée sur le backbone.

---

## 7. Transfer Learning

Le backbone InceptionResNetV2 est initialisé avec les poids ImageNet.
Le backbone est appelé explicitement avec `training=False` à toutes les phases :

```python
features = backbone(preprocessed_inputs, training=False)
```

Cela garantit que les couches BatchNormalization du backbone restent en mode inférence
même quand elles sont dé-gelées, évitant la contamination des statistiques de batch.

---

## 8. Entraînement en Trois Phases

### Phase 1 — Head Training (backbone gelé)
- Backbone `trainable=False`
- Seule la tête est entraînable
- LR = 3e-4 (Adam)
- 10 epochs
- Class weights activés

**Objectif** : apprendre des projections adaptées aux 22 classes histologiques
avant de commencer à ajuster le backbone.

### Phase 2 — Dégel Partiel (30% du backbone)
- Les 70% premiers layers du backbone restent gelés
- Les 30% derniers layers sont dé-gelés
- **Toutes les BatchNormalization restent gelées**
- Recompilation obligatoire
- LR = 1e-5 (Adam)
- 30 epochs avec EarlyStopping, ReduceLROnPlateau

**Objectif** : adapter les représentations de haut niveau du backbone aux textures
histologiques tout en préservant les features de bas niveau.

### Phase 3 — Dégel Complet
- Tous les layers compatibles du backbone sont dé-gelés
- **Toutes les BatchNormalization restent gelées**
- Recompilation obligatoire
- LR = 3e-6 (Adam) — très faible pour éviter la catastrophic forgetting
- 15 epochs avec EarlyStopping, ReduceLROnPlateau

**Objectif** : fine-tuning global à très faible LR pour maximiser la représentation.

---

## 9. Dégel Partiel — Implémentation

Le calcul des 30% est **déterministe** :

```python
cutoff = int(total_backbone_layers * (1.0 - 0.30))
for i, layer in enumerate(backbone.layers):
    if isinstance(layer, BatchNormalization):
        layer.trainable = False  # TOUJOURS gelé
    elif i < cutoff:
        layer.trainable = False  # Premiers 70%
    else:
        layer.trainable = True   # Derniers 30%
```

La fraction est calculée sur **toutes** les couches du backbone (incluant BN),
puis les BN sont forcément re-gelées. Cela donne un dégel déterministe
sans compter naïvement les BN dans la fraction.

---

## 10. BatchNormalization — Stratégie Frozen

Les couches BatchNormalization du backbone sont gelées dans les phases 2 et 3.

**Pourquoi** :
- Avec un petit dataset (432 images), les statistiques de batch estimées
  sur les mini-batches peuvent être bruitées
- Les statistiques appris sur ImageNet (1.2M images) sont plus fiables
- Maintenir `training=False` dans l'appel backbone garantit que les BN
  utilisent les statistiques globales (running mean/variance)

**Vérification automatique** : après chaque changement de trainabilité,
une validation lève une exception si une BN du backbone est entraînable.

---

## 11. Mixed Precision

`tf.keras.mixed_precision.set_global_policy("mixed_float16")` est activé.

- Les calculs intermédiaires se font en `float16` (2× plus rapide sur GPU Colab)
- La couche `predictions` utilise explicitement `dtype="float32"` pour la stabilité numérique
- La sortie du modèle est toujours `float32`

---

## 12. Augmentation

Identique à DenseNet121 Exp D (augmentation riche validée) :

| Transformation | Paramètre |
|---|---|
| Flip horizontal | Activé |
| Flip vertical | Activé |
| Rotation | factor=0.04 (~14°) |
| Zoom | factor=0.10 (±10%) |
| Brightness | delta=0.05×255 |
| Contrast | factor=0.10 |
| Saturation | lower=0.90, upper=1.10 |
| Gaussian noise | stddev=0.01×255 |
| Clip | [0, 255] |

**Important** :
- Augmentation uniquement sur le jeu d'**entraînement**
- Validation, prédictions OOF, matrices de confusion : images **originales uniquement**
- Aucune image augmentée n'est écrite comme nouvelle image originale

---

## 13. Class Weights

Poids balancés calculés par fold sur l'ensemble d'entraînement :

```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight("balanced", classes=unique_classes, y=train_labels)
```

- Activés uniquement pendant `model.fit()`
- **Jamais** pendant l'évaluation ou les prédictions OOF

---

## 14. Validation Croisée

Même manifeste que DenseNet121 Exp D : `data/manifests/densenet121_folds.csv`

- 432 images originales (original_22_dataset_manifest.csv)
- 5 folds stratifiés :
  - Folds 0, 1 : 87 images
  - Folds 2, 3, 4 : 86 images
- Aucune image augmentée dans le manifeste

---

## 15. OOF (Out-Of-Fold) Predictions

Pour chaque fold, le modèle prédit sur les images de validation
(images jamais vues pendant l'entraînement de ce fold).

**Schéma CSV** :
```
image_path, image_id, fold, true_label, true_class,
predicted_label, predicted_class, correct,
prob_0, prob_1, ..., prob_21
```

**Vérifications** :
- `predicted_label == argmax(prob_0...prob_21)` ✓
- `correct == (true_label == predicted_label)` ✓
- Somme des probabilités ≈ 1.0 ✓
- Aucun doublon ✓

---

## 16. Reprise Après Interruption

Si l'entraînement est interrompu pendant un fold :

1. **Détection** : vérification de tous les fichiers requis + marker `fold_X_complete.json`
2. **Backup** (si `--backup-partial`) : renommage avec timestamp `_interrupted_YYYYMMDD_HHMMSS`
3. **Jamais de suppression automatique** des fichiers partiels
4. **Reprise propre** dans un nouveau dossier

Les folds déjà complétés **ne sont pas ré-entraînés** grâce à `--skip-completed`.

---

## 17. Métriques

Par fold et pour l'OOF combiné :

| Métrique | Description |
|---|---|
| Accuracy | Proportion de prédictions correctes |
| Balanced Accuracy | Moyenne de recall par classe |
| Macro Precision | Précision non pondérée |
| Macro Recall | Rappel non pondéré |
| Macro F1 | F1 non pondéré (métrique principale) |
| Weighted F1 | F1 pondéré par support |
| Cohen's Kappa | Accord au-delà du hasard |
| Cross-entropy | val_ce_hard (monitor des callbacks) |
| Top-3 Accuracy | Proportion où la vraie classe est dans le top 3 |

---

## 18. Critères de Décision (Screening)

### Qualification individuelle minimale
- `mean_accuracy >= 0.84`
- `mean_macro_f1 >= 0.76`
- `mean_weighted_f1 >= 0.83`

### Qualification forte
- `global_accuracy >= 0.87`
- `global_macro_f1 >= 0.81`
- `global_weighted_f1 >= 0.86`

### Qualification par complémentarité
- Macro F1 ensemble 50/50 > DenseNet121
- Accuracy ensemble ≥ DenseNet121

### Statuts possibles
- `screening_qualified_individual`
- `screening_qualified_ensemble_only`
- `screening_rejected`
- `screening_incomplete`

**⚠ NE JAMAIS lancer les folds 1 et 2 automatiquement.**

---

## 19. Comparaison avec DenseNet121 Exp D

### Références scientifiques

| Référence | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Baseline historique | ~0.8449 | ~0.7728 | ~0.8403 |
| DenseNet121 Exp D OOF complet (432 img) | ~0.8843 | ~0.8280 | ~0.8794 |
| DenseNet121 Exp D screening (259 img, folds 0,3,4) | ~0.8610 | ~0.7889 | ~0.8538 |

### Périmètres de comparaison

**Ne jamais comparer** une métrique globale sur 259 images à une métrique sur 432 images
sans indiquer les périmètres respectifs.

---

## 20. Ensemble 50/50

Formule :
```
p_ensemble = (p_densenet121 + p_inception_resnet_v2) / 2
```

- Poids fixes (0.5 / 0.5) — **non optimisés** sur les 259 images de screening
- L'optimisation des poids sur le screening introduirait un biais d'overfitting

---

## 21. Limites liées aux 432 Images

Ce projet utilise 432 images histologiques réparties en 22 classes (~20 images/classe).

**Limites importantes** :
- Faible nombre d'images par classe → variance élevée des métriques par fold
- Les métriques de screening (259 images) ont des intervalles de confiance larges
- Les décisions d'architecture basées sur ce screening sont indicatives, pas définitives
- Le screening couvre seulement 3 des 5 folds

---

## 22. Absence de Valeur Diagnostique Clinique

Ce projet est développé dans un **contexte éducatif et de recherche**.

- Les modèles ne sont pas validés cliniquement
- Ils ne sont pas destinés à être utilisés pour des diagnostics médicaux
- Les résultats ne remplacent pas l'expertise d'un pathologiste qualifié
- Aucune décision médicale ne doit être basée sur ces prédictions

---

## Fichiers du Projet

| Fichier | Description |
|---|---|
| `src/models/inception_resnet_v2.py` | Modèle, preprocessing, fine-tuning |
| `configs/experiments/inception_resnet_v2_exp_a_high_performance.yaml` | Configuration complète |
| `scripts/train_inception_resnet_v2.py` | Script d'entraînement CLI |
| `scripts/generate_inception_resnet_v2_screening_report.py` | Rapport de screening |
| `scripts/compare_densenet_inception_resnet_v2_oof.py` | Comparaison DenseNet vs InceptionResNetV2 |
| `notebooks/colab/inception_resnet_v2_exp_a_complete_training.ipynb` | Notebook Colab autonome |
| `tests/test_inception_resnet_v2.py` | Tests unitaires modèle |
| `tests/test_inception_resnet_v2_report.py` | Tests rapport screening |
| `tests/test_compare_densenet_inception_resnet_v2_oof.py` | Tests comparaison |

---

*Document créé le 28 juillet 2026 — Expérience A — InceptionResNetV2*
