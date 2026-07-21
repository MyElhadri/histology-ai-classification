# Experiments

Experiment tracking, logs, and run history.

## Structure

```
experiments/
├── runs/                  # TensorBoard logs and training run outputs
├── experiment_tracking.md # Manual experiment log
└── README.md
```

## Tracking

Each experiment should record:

1. **Date and time**
2. **Model architecture** and hyperparameters
3. **Dataset version** (preprocessing applied)
4. **Training metrics** (loss, accuracy per epoch)
5. **Evaluation results** (confusion matrix, classification report)
6. **Observations and next steps**

> Use `experiment_tracking.md` to log experiment results in a structured format.
