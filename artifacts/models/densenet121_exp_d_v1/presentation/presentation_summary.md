# Résumé de la Présentation Scientifique - DenseNet121 (Expérience D)

Ce document résume les caractéristiques et les performances du meilleur modèle d'apprentissage profond DenseNet121 obtenu au cours de l'évaluation finale.

## Caractéristiques du Modèle & Protocole
- **Architecture de base** : DenseNet121 pré-entraîné sur ImageNet.
- **Taille d'entrée** : 224x224 RGB.
- **Nombre de classes** : 22 classes de tissus histologiques.
- **Dataset d'évaluation** : 432 images originales uniquement (aucune image augmentée ou dupliquée n'a été insérée dans l'évaluation).
- **Protocole de validation** : Validation croisée stratifiée en 5 folds.
- **Régulation de l'apprentissage** :
  - **Augmentation** : Augmentation riche uniquement en ligne (online) pendant l'entraînement.
  - **Tête de classification (Classifier Head)** : Spécifique *article-inspired* comprenant une couche dense de 512 unités (activation ELU), une normalisation par batch, un dropout (rate = 0.30) et une seconde couche dense de 128 unités.
  - **Fine-tuning** : Un fine-tuning complet de toutes les couches du backbone a été réalisé, tout en conservant les couches BatchNormalization du backbone gelées.
- **Inférence** : Inférence classique **sans TTA** (Test-Time Augmentation rejetée car n'apportant pas d'amélioration).

## Performances Globales Out-of-Fold (OOF)
- **Accuracy OOF** : **88,43 %** (382 prédictions correctes sur 432 images).
- **Macro F1 moyen** : **82,80 %**.
- **Weighted F1 moyen** : **87,94 %**.
