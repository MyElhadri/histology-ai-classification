# System Architecture

> Technical architecture of the Histology AI Classification system.

## High-Level Architecture

```
┌─────────────────────────────────────────────────┐
│                 Mobile Application              │
│            (Flutter / React Native)             │
└────────────────────┬────────────────────────────┘
                     │ REST API
┌────────────────────▼────────────────────────────┐
│                  Backend API                    │
│              (FastAPI / Flask)                  │
└────────────────────┬────────────────────────────┘
                     │ Inference
┌────────────────────▼────────────────────────────┐
│              Deep Learning Models               │
│                                                 │
│  ┌─────────────┐  ┌─────────────┐              │
│  │ DenseNet121 │  │ ResNet50V2  │              │
│  └──────┬──────┘  └──────┬──────┘              │
│         └────────┬───────┘                      │
│           ┌──────▼──────┐                       │
│           │  Ensemble   │                       │
│           └─────────────┘                       │
└─────────────────────────────────────────────────┘
```

## Deep Learning Pipeline

```
Raw Images → Preprocessing → Augmentation → Training → Evaluation → Export
```

## Design Decisions

*Document key architectural decisions here as the project progresses.*
