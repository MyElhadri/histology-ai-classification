# Benchmark Local CPU — DenseNet121 (Expérience D)

Ce document explique comment exécuter et interpréter le benchmark local CPU pour mesurer les performances d'entraînement de la première fold (Fold 0) du modèle DenseNet121 sur CPU sous Windows.

---

## 1. Installation des dépendances

Activez votre environnement virtuel local Python et installez les dépendances nécessaires :

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

---

## 2. Emplacement du Dataset

Le dataset reconstruit (22 classes NuInsSeg) doit se trouver à l'emplacement exact suivant :

```text
data/raw/nuinsseg_human_22_original
```

Il doit contenir exactement **432 images** réparties dans les sous-dossiers de classes.

---

## 3. Lancement du Benchmark

Exécutez le script sans argument :

```powershell
python scripts/run_local_cpu_benchmark.py
```

Le script va :
1. Forcer l'exécution sur CPU (`CUDA_VISIBLE_DEVICES = "-1"`).
2. Vérifier la présence des 432 images du dataset.
3. Entraîner le Fold 0 pendant 2 époques de tête (Phase 1) et 3 époques de fine-tuning (Phase 2) avec un `batch_size` de 4.

---

## 4. Emplacement des Résultats

À la fin du benchmark, les résultats sont sauvegardés dans :

```text
results/local_cpu_benchmark/benchmark_summary.json
```

---

## 5. Interprétation du Temps Estimé

Le fichier `benchmark_summary.json` contient les durées moyennes par époque et des estimations pour un entraînement complet d'une fold (10 époques de tête + N époques de fine-tuning) :

- `average_head_epoch_seconds` : durée moyenne d'une époque de tête (Phase 1).
- `average_finetuning_epoch_seconds` : durée moyenne d'une époque de fine-tuning (Phase 2).
- `estimated_full_fold_20_epochs` : estimation pour 10 époques head + 20 époques fine-tuning.
- `estimated_full_fold_30_epochs` : estimation pour 10 époques head + 30 époques fine-tuning.
- `estimated_full_fold_40_epochs` : estimation pour 10 époques head + 40 époques fine-tuning.

### Formule de Calcul
$$\text{Temps Estimé} = 10 \times \text{average\_head\_epoch\_seconds} + N \times \text{average\_finetuning\_epoch\_seconds}$$
