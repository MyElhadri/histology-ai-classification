# InceptionV3 Expérience A — Documentation Scientifique & Protocole

## 1. Objectif & Justification Scientifique

L'objectif de cette expérience est d'évaluer le modèle **InceptionV3** (pré-entraîné sur ImageNet) comme second modèle candidat pour la classification histologique sur les 22 classes humaines NuInsSeg.

InceptionV3 apporte une architecture multi-échelle ("Inception modules") complémentaire aux connexions denses de DenseNet121, ce qui en fait un candidat potentiel pour un futur apprentissage par ensemble (**DenseNet121 + InceptionV3 + ResNet50**).

---

## 2. Protocole Équitable (Fair Comparison)

Pour garantir une comparaison rigoureuse avec notre baseline **DenseNet121 Exp D**, le protocole suivant est appliqué à l'identique :

| Paramètre | DenseNet121 Exp D | InceptionV3 Exp A |
| :--- | :--- | :--- |
| **Dataset** | 432 images originales (22 classes) | 432 images originales (22 classes) |
| **Manifeste** | `data/manifests/densenet121_folds.csv` | `data/manifests/densenet121_folds.csv` |
| **Seed** | `42` | `42` |
| **Folds de Screening** | `[0, 3, 4]` (259 images total) | `[0, 3, 4]` (259 images total) |
| **Dimensions Entrée** | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ |
| **Poids d'origine** | ImageNet | ImageNet |
| **Preprocessing** | Normalisation DenseNet | `Rescaling(scale=1.0/127.5, offset=-1.0)` |
| **Tête** | `article_inspired` | `article_inspired` |
| **Augmentation** | Augmentation riche (train) | Augmentation riche (train) |
| **Class Weights** | Balanced par fold | Balanced par fold |
| **Phases** | Phase 1 (10 époques), Phase 2 (40 époques) | Phase 1 (10 époques), Phase 2 (40 époques) |
| **Backbone BN** | Non entraînables en Phase 2 | Non entraînables en Phase 2 |

---

## 3. Preprocessing [-1, 1]

InceptionV3 nécessite une plage d'entrée dans $[-1, 1]$. Le pipeline transmet des images en `float32` nativement dans $[0, 255]$.
Le modèle intègre une couche sérialisable :

```python
tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="inceptionv3_preprocessing")
```

Cette couche est vérifiée numériquement comme étant exactement équivalente à `tf.keras.applications.inception_v3.preprocess_input`.

---

## 4. Tête de Classification `article_inspired`

La même tête autoritaire est réutilisée via `src/models/heads.py` :
- `GlobalAveragePooling2D`
- `Dense(512, activation='elu')`
- `BatchNormalization`
- `Dropout(0.30)`
- `Dense(128, activation='elu', kernel_regularizer=l2(0.01))`
- `Dense(22, activation='softmax')`

---

## 5. Critères de Décision pour le Screening (Folds 0, 3, 4)

### Références DenseNet121 Exp D sur Folds 0, 3, 4 :
- **Mean Accuracy** : $\approx 0.8610$
- **Mean Macro F1** : $\approx 0.7889$
- **Mean Weighted F1** : $\approx 0.8538$

### Règles d'Évaluation InceptionV3 :
Le modèle est qualifié pour les Folds 1 et 2 (évaluation complète 432 images) si au moins une condition est remplie :
1. **Performance individuelle prometteuse** : Accuracy $\ge 0.84$, Macro F1 $\ge 0.76$, Weighted F1 $\ge 0.83$.
2. **Complémentarité d'Ensemble** : L'ensemble 50/50 DenseNet121 + InceptionV3 améliore le Macro F1 global sans dégrader l'Accuracy.

Dans le cas contraire, InceptionV3 recevra le statut `screening_rejected` sans réentraîner les folds restants.

---

## 6. Structure des Fichiers et Sauvegarde Drive

Pour chaque fold, les fichiers suivants sont générés dans `OUTPUT_DIR` :
- `models/inceptionv3/checkpoints/fold_X/best_model.keras`
- `reports/inceptionv3/metrics/fold_X.json`
- `reports/inceptionv3/predictions/fold_X_oof_predictions.csv`
- `reports/inceptionv3/history/fold_X_history.json`
- `reports/inceptionv3/history/fold_X_history.csv`
- `reports/inceptionv3/class_weights/fold_X_class_weights.json`
- `reports/inceptionv3/inceptionv3_screening_summary.json`
