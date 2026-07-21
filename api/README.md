# API — Backend Service

> **Status:** Planned (Phase 2)

This directory will contain the REST API backend for serving model predictions.

## Planned Stack

- **Framework:** FastAPI or Flask
- **Model Serving:** TensorFlow Serving or direct inference
- **Endpoints:**
  - `POST /predict` — Accepts microscope slide image, returns tissue classification and confidence score
  - `GET /classes` — Returns available tissue class labels and descriptions
  - `GET /health` — Health check endpoint

## Integration

The API will load the exported model from `models/exports/` and provide inference as a service to the mobile application.
