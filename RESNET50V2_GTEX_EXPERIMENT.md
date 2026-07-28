# ResNet50V2 — GTEx Standardisé — 11 Classes Histologiques — Expérience A

## Contexte
Cette expérience vise à évaluer la capacité du modèle pré-entraîné ImageNet `ResNet50V2` à classifier 11 tissus histologiques à partir du dataset standardisé GTEx.

## Protocole Technique

### Architecture du Modèle
- **Backbone** : ResNet50V2 (Poids : ImageNet)
- **Pré-traitement** : Intégré dans le modèle via Keras `Rescaling(scale=1.0/127.5, offset=-1.0)`.
- **Tête de Classification (Article-inspired)** :
  - GlobalAveragePooling2D
  - Dense 512 (ELU) + L2(0.01) + BatchNorm
  - Dropout(0.3)
  - Dense 128 (ELU) + L2(0.01) + BatchNorm
  - Dropout(0.3)
  - Sortie Dense(11, Softmax)

### Dataset
- **Classes** (11) : Bladder, Brain, Cerebellum, Kidney, Liver, Lung, Muscle, Oesophagus, Pancreas, Spleen, Testis.
- **Répartition (Total 56,742 patches)** :
  - Train : 40,424
  - Validation : 8,114
  - Test : 8,204
- **Isolement des donneurs** : Strict. Aucun croisement de donneurs (patients) entre les sets d'entraînement, de validation et de test n'est toléré. L'audit d'intégrité échoue si une fuite est détectée.

### Pipeline d'entraînement
Le Fine-Tuning s'effectue en trois phases strictes pour éviter l'oubli catastrophique :
1. **Phase 1** : Entraînement exclusif de la tête de classification (Backbone totalement figé). BatchNormalization en mode Inférence.
2. **Phase 2** : Fine-Tuning partiel. Les 30% derniers layers du backbone sont débloqués. La BatchNormalization reste **toujours** figée.
3. **Phase 3** : Full Fine-Tuning. Tous les layers non-BN du backbone sont débloqués.

**Hyperparamètres communs :**
- Batch size : 32
- Optimiseur : Adam
- Mixed Precision : Activée (`mixed_float16`)
- Augmentation de données : Flips, Rotation(10%), Zoom(10%), Translation(10%), Contraste.
- Callbacks : EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.

### Notebook Kaggle
Un notebook autonome `resnet50v2_gtex_11_exp_a_complete_training.ipynb` a été généré.
Il intègre les vérifications d'environnement, l'audit du dataset, la compilation des tests, le smoke test (dry-run), l'entraînement en trois phases et la génération d'un rapport avec métriques par patch et par donneur.

## Exécution
Lancez le notebook sur Kaggle avec le dataset `GTEx_11_classes` attaché à `/kaggle/input`.
L'exécution est sécurisée : les tests unitaires et l'audit d'intégrité valident les prémisses du protocole avant tout apprentissage.
