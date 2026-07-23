# Experiment B: Article-Inspired Classification Head Report

## Objectif
Implémenter et évaluer une tête de classification DenseNet121 inspirée de la littérature, sans perturber le comportement fondamental de la baseline et sans ajouter d'augmentation de données complexe, afin d'isoler l'impact exclusif de cette architecture sur les performances (accuracy et F1 score macro).

## Hypothèse
L'ajout de couches denses intermédiaires (512 puis 128 neurones), associées à une normalisation par lot (BatchNormalization) et à une régularisation L2 forte (0.01), ainsi qu'à une activation ELU plus fluide, permettra au réseau d'apprendre des représentations non-linéaires plus robustes et de réduire l'overfitting. Le Global Average Pooling (au lieu du Flatten souvent utilisé) réduira la dimensionnalité sans explosion du nombre de paramètres.

## Architecture Baseline
- Extracteur de caractéristiques : DenseNet121 (sans fully connected layers, ImageNet).
- Réduction spatiale : GlobalAveragePooling2D.
- Régularisation : Dropout(0.30).
- Classification finale : Dense(22, activation="softmax").

## Architecture B (Article-Inspired)
- Extracteur de caractéristiques : DenseNet121.
- Réduction spatiale : GlobalAveragePooling2D.
- **Intermédiaire 1** : Dense(512, activation="elu").
- **Normalisation** : BatchNormalization.
- **Régularisation 1** : Dropout(0.30).
- **Intermédiaire 2** : Dense(128, activation="elu", kernel_regularizer=L2(0.01)).
- Classification finale : Dense(22, activation="softmax").

## Seule Variable Modifiée
Seule la tête de classification a été modifiée par rapport à la baseline. Le taux d'apprentissage, la politique d'augmentation d'images (maintenue volontairement à "baseline"), les "class weights", l'optimiseur, ainsi que la gestion de l'entraînement et du fine-tuning restent constants.

## Justification du GlobalAveragePooling
Le `GlobalAveragePooling2D` produit un vecteur de dimension 1024 pour chaque image, indépendamment de la taille initiale. Contrairement au `Flatten`, qui concatène toutes les sorties spatiales (entraînant un nombre colossal de paramètres pour les couches denses suivantes et favorisant un sur-apprentissage massif), le pooling réduit les dimensions tout en conservant l'invariance spatiale globale des structures histologiques.

## Paramètres Exacts
Expérience B :
- `pooling`: `global_average`
- `dense_1_units`: 512
- `dense_1_activation`: `elu`
- `batch_normalization`: `true`
- `dropout_rate`: 0.30
- `dense_2_units`: 128
- `dense_2_activation`: `elu`
- `l2_strength`: 0.01
- `output_activation`: `softmax`
- Augmentation maintenue à `baseline`
- Batch size : 16
- Learning rates : 0.001 (head), 0.00001 (fine-tuning)

## Tests Réalisés
L'implémentation a fait l'objet de tests validant la sécurité et l'exactitude de l'architecture :
1. **Intégrité de la Baseline** : Vérification que l'absence de paramètres spécifiques restaure strictement l'ancienne tête.
2. **Structure de l'Architecture B** : Validation de la présence de `GlobalAveragePooling2D`, absence de `Flatten`, présence et configuration correcte des couches `Dense` (512, 128), `BatchNormalization`, `Dropout` et des fonctions `ELU`.
3. **Robustesse de Configuration** : Les configurations erronées (taux de dropout impossibles, nombres de neurones négatifs, mauvais identifiant de tête) déclenchent immédiatement des erreurs explicites.
4. **Viabilité (Forward Pass)** : Un batch de dimensions simulées (2, 224, 224, 3) a été injecté dans le modèle avec succès, et la sortie (2, 22) somme bien à 1, validant ainsi la logistique des tenseurs.

## Procédure Colab
Le notebook `02_train_densenet121.ipynb` a été configuré avec le path `configs/experiments/densenet121_exp_b_article_head.yaml` sous le mode `selected_folds` (folds 0 et 1). La mémoire TensorFlow et l'objet Keras Backend sont remis à zéro entre chaque fold. 
Les sauvegardes de l'expérience (checkpoints, metrics, poids, récapitulatif avec totalisation du nombre de paramètres de l'architecture) sont stockées dans `/content/drive/MyDrive/histology-results/densenet121-exp-b-article-head-screening`, sans impact sur les résultats de la baseline ni sur l'expérience A.

## Limites
L'ajout de couches intermédiaires complexifie modérément le modèle (plus de paramètres dans la tête). Le BatchNormalization pourrait parfois se montrer instable sur de tout petits batch sizes en fine-tuning, mais un batch de 16 reste convenable.

## Critères de Décision
L'expérience B sera jugée fructueuse si la précision moyenne de validation ou le macro F1 score sur les 2 folds testés dépassent significativement ceux de la baseline entraînée dans les mêmes conditions exactes, tout en réduisant l'écart train/val (démontrant une diminution de l'overfitting).
