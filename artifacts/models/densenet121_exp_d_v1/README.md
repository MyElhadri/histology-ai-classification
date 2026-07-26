# DenseNet121 - Expérience D Final Model Registry

Il s'agit du meilleur modèle DenseNet121 obtenu (Expérience D).

## Organisation
- Les cinq checkpoints correspondent aux cinq folds de la validation croisée (stratified 5-fold) et sont stockés sous `checkpoints/fold_X/best_model.keras`.
- Ils peuvent être combinés en ensemble learning en faisant la moyenne de leurs prédictions probabilistes.
- **Attention** : Pour des raisons scientifiques, il ne faut pas choisir seulement le fold 1 comme unique modèle final.
- Pour une utilisation en production dans l'application, un futur modèle entraîné sur l'intégralité des 432 images pourra être créé.

## Protocole d'évaluation
- Les images évaluées sont les 432 images originales du dataset.
- Les augmentations (rich online augmentation) ont été créées à la volée (en ligne) uniquement pendant l'entraînement.
- Aucune image augmentée pré-générée ou dupliquée n'a été utilisée pour le calcul des métriques d'évaluation.
- La méthode TTA (Test-Time Augmentation) a été testée mais rejetée. L'inférence finale retenue est **sans TTA** (D WITHOUT TTA SELECTED).

## Résultats globaux (Out-of-Fold)
- **Accuracy OOF** : 88,43 % (382 prédictions correctes sur 432)
- **Macro F1 moyen** : 82,80 %
- **Weighted F1 moyen** : 87,94 %
