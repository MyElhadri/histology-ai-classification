# Dataset

> Human Histopathological H&E Stained Nuclei Images — Kaggle.

## Source

The dataset contains microscopic images of human tissue stained with
Hematoxylin & Eosin (H&E), organized into 22 tissue-type classes.

## Organization

```
dataset_root/
├── human_tissue_image-bladder/
├── human_tissue_image-brain/
├── human_tissue_image-cardia/
├── human_tissue_image-cerebellum/
├── human_tissue_image-epiglottis/
├── human_tissue_image-gland/
├── human_tissue_image-jejunum/
├── human_tissue_image-kidney/
├── human_tissue_image-liver/
├── human_tissue_image-lung/
├── human_tissue_image-melanoma/
├── human_tissue_image-muscle/
├── human_tissue_image-oesophagus/
├── human_tissue_image-pancreas/
├── human_tissue_image-peritoneum/
├── human_tissue_image-pylorus/
├── human_tissue_image-rectum/
├── human_tissue_image-spleen/
├── human_tissue_image-testis/
├── human_tissue_image-tongue/
├── human_tissue_image-tonsile/
└── human_tissue_image-umbilical-cord/
```

Each subdirectory name is used as the class label.

## Class Mapping

A deterministic class mapping (`class_mapping.json`) is generated
alphabetically. Both DenseNet121 and ResNet50V2 use the **same
mapping** to ensure comparable predictions.

## Accepted Image Formats

`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`

## Data Integrity

The `build_manifest` module records an MD5 hash for each image,
enabling exact duplicate detection.  Original images are **never
modified** by any pipeline step.
