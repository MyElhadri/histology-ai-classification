"""Ensemble learning model.

This module handles:
- Combining DenseNet121 and ResNet50V2 predictions
- Ensemble strategies: averaging, weighted averaging, stacking
- Confidence calibration across models
- Final prediction aggregation
"""
