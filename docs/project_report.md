# Binary Melanoma Screening from Dermatoscopic Images
### Artificial Intelligence in Medicine and Healthcare — Project Report

**Gabriel Fernando Missaka Mendes | Eduardo Takei Yaginuma**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Proposal](#project-proposal)
3. [Dataset](#dataset)
4. [Proposed Approach](#proposed-approach)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Expected Outcome](#expected-outcome)
7. [Project Organization and Authors' Contributions](#project-organization-and-authors-contributions)
8. [Development](#development)
9. [References](#references)

---

## Introduction

### Problem Background

Cutaneous melanoma is widely recognized as the most aggressive and life-threatening form of skin cancer. Although it accounts for only approximately 1%–4% of all skin cancer cases, it is responsible for over 80% of skin cancer-related deaths (American Cancer Society, 2024; Vieira & Brandão, 2022). Its high lethality is primarily associated with its strong metastatic potential — unlike other skin cancers, melanoma can rapidly spread to distant organs once it invades deeper skin layers, particularly after the tumor transitions from the radial (superficial) growth phase to the vertical growth phase (Caraviello et al., 2025).

Early detection is critical for improving patient outcomes. When diagnosed at a localized stage, melanoma presents a 5-year relative survival rate above 99%. However, if the disease progresses to distant metastasis, this rate drops to approximately 35% (American Cancer Society, 2024). This sharp contrast highlights the importance of timely and accurate diagnosis.

In clinical practice, accurate diagnosis remains challenging even for experienced dermatologists due to the visual similarity between malignant and benign lesions. Computational tools based on artificial intelligence can support clinical decision-making, particularly by reducing false negatives — the most critical diagnostic error in melanoma screening.

---

## Project Proposal

This project proposes the development of a deep learning-based system for binary classification of dermatoscopic images, distinguishing melanoma from non-melanoma lesions.

Rather than addressing a multi-class classification problem, the task is reformulated into a binary setting: all non-melanoma categories (basal cell carcinoma, actinic keratosis, benign keratosis, dermatofibroma, vascular lesions, and melanocytic nevi) are grouped into a single class. This approach aligns with clinical priorities, focusing specifically on detecting melanoma due to its high lethality.

Special emphasis will be placed on challenging negative samples — lesions that visually resemble melanoma — to improve model robustness and reduce false negatives. In particular, melanocytic nevi (NV), which constitute approximately 67% of the dataset and are visually very similar to melanoma, represent the primary hard negatives.

To align model predictions with clinical use, the classification threshold will be selected based on maximizing sensitivity while maintaining a minimum acceptable level of specificity, ensuring a balance between early detection and false positive control.

**Impact:** The tool would work as a pre-screening step during routine skin exams. After capturing a dermatoscopic image, the system would instantly flag suspicious lesions, helping the dermatologist decide whether to proceed with a biopsy. It could function as a plugin within existing image capture software, adding a real-time alert without disrupting the examination flow.

---

## Dataset

The dataset used in this project is the **HAM10000 (Human Against Machine with 10,000 training images)** dataset, available on Kaggle:

> Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions*. Scientific Data.  
> Original source: Harvard Dataverse (doi:10.7910/DVN/DBW86T)

### Overview

| Attribute | Value |
|-----------|-------|
| Total images | 10,015 |
| Number of original classes | 7 |
| Binary classes | Melanoma (1) vs. Non-Melanoma (0) |
| Image format | JPG |
| Image resolution | 600×450 px (uniform) |
| Masks available | Yes (PNG segmentation masks) |
| Dataset size | ~2.77 GB |
| License | CC BY-NC-SA 4.0 |

### Class Distribution

| Class | Abbreviation | Binary Label | Count | % |
|-------|-------------|--------------|-------|---|
| Melanocytic Nevi | NV | Non-Melanoma | 6,705 | 66.9% |
| Melanoma | MEL | Melanoma | 1,113 | 11.1% |
| Benign Keratosis | BKL | Non-Melanoma | 1,099 | 11.0% |
| Basal Cell Carcinoma | BCC | Non-Melanoma | 514 | 5.1% |
| Actinic Keratosis | AKIEC | Non-Melanoma | 327 | 3.3% |
| Vascular Lesions | VASC | Non-Melanoma | 142 | 1.4% |
| Dermatofibroma | DF | Non-Melanoma | 115 | 1.1% |
| **Total** | | | **10,015** | **100%** |

**Class imbalance:** ~8:1 (non-melanoma : melanoma). This is severe and directly impacts the choice of loss function and evaluation metrics.

### Ground Truth Reliability

More than 50% of the lesions are confirmed through histopathological examination (gold standard for skin cancer diagnosis). The remaining cases are validated through follow-up examinations, expert consensus, or in vivo confocal microscopy.

### Segmentation Masks

The dataset provides lesion segmentation masks for all images. These masks can be used as an optional preprocessing step to isolate the lesion region and reduce background noise, potentially improving feature extraction without introducing a separate segmentation model.

---

## Proposed Approach

### Architecture

Transfer learning will be applied using pre-trained architectures:

- **ResNet-50** — strong baseline with residual connections
- **EfficientNet** — more parameter-efficient alternative

The final layer will be adapted to output a single probability score using a sigmoid activation function for binary classification.

### Training Strategy

Training follows a two-stage approach:
1. **Initial training** with frozen convolutional layers — only the classification head is trained
2. **Fine-tuning** of deeper layers to capture domain-specific dermatoscopic features

### Handling Class Imbalance

Given the severe 8:1 imbalance, the following strategies will be employed:
- Class weighting in the loss function
- Focal loss (to emphasize hard negatives)
- Targeted data augmentation for the minority class

### Preprocessing Pipeline

- Resize to 224×224 px
- Normalize with ImageNet mean and std
- Optional: apply segmentation masks to isolate lesion regions

### Data Augmentation (training only)

- Random horizontal and vertical flip
- Random rotation
- Color jitter (brightness, contrast, saturation)
- Random resized crop

### Threshold Selection

The decision threshold will not be fixed at 0.5. Instead, it will be selected on the validation set to maximize sensitivity while maintaining a minimum acceptable specificity — aligned with the clinical priority of avoiding missed diagnoses.

---

## Evaluation Metrics

Given the clinical implications of missed diagnoses, evaluation goes beyond simple accuracy:

| Metric | Role | Description |
|--------|------|-------------|
| **Sensitivity (Recall)** | Primary | Fraction of melanoma cases correctly identified — minimizes false negatives |
| **Specificity** | Secondary | Fraction of non-melanoma cases correctly identified |
| **AUC-ROC** | Overall | Model performance across all classification thresholds |

Accuracy alone is not a sufficient metric due to the severe class imbalance — a model that always predicts "non-melanoma" would achieve ~89% accuracy while being clinically useless.

---

## Expected Outcome

A decision-support tool capable of assisting dermatologists during routine examinations. By analyzing dermatoscopic images, the system can highlight suspicious lesions and support decisions regarding further diagnostic procedures, such as biopsy. The solution is designed to be lightweight and easily integrable into existing clinical workflows.

Optional future extension: integration into a web application (Flask or Streamlit) allowing real-time image upload and classification.

---

## Project Organization and Authors' Contributions

The project is organized into five development sprints to ensure a structured and iterative workflow.

### Sprint 1 — Data Exploration
- Literature review on melanoma detection and clinical context
- Repository setup and project structure organization
- Dataset acquisition and organization
- Exploratory Data Analysis (EDA)
- Analysis of class distribution and binary imbalance
- Inspection of image dimensions and pixel statistics
- Exploration of segmentation masks and lesion coverage
- Image quality check

### Sprint 2 — Feature Engineering and Preprocessing
- Image resizing and normalization
- Data augmentation pipeline (rotation, flipping, color jitter)
- Optional segmentation mask application
- Train/validation/test split strategy and CSV generation
- Batch sanity check and sample weight computation

### Sprint 3 — Modeling
- Selection of baseline architecture (ResNet-50 or EfficientNet)
- Implementation of transfer learning
- Training with frozen layers followed by fine-tuning
- Handling class imbalance (class weights or focal loss)
- Initial model training and baseline performance assessment

### Sprint 4 — Validation and Optimization
- Hyperparameter tuning
- Threshold selection prioritizing sensitivity with minimum specificity
- Performance evaluation (Recall, Specificity, AUC-ROC)
- Error analysis and model refinement
- Regularization strategies (dropout, early stopping)

### Sprint 5 — Testing and Finalization
- Final evaluation on held-out test set
- Analysis of generalization performance
- Documentation and preparation of final report
- Optional deployment as a web application

### Authors' Contributions

**Gabriel Fernando Missaka Mendes** led the data exploration and feature engineering phases, including repository setup, project structure organization, exploratory data analysis, dataset preprocessing, and the design of data augmentation strategies.

**Eduardo Takei Yaginuma** is primarily responsible for the modeling, validation, and testing stages, including the implementation of deep learning architectures, training procedures, and performance evaluation.

Both authors collaborated across all stages of the project, contributing to decision-making, experimental design, and iterative improvements to the pipeline.

---

## Development

### Sprint 1 — Data Exploration

The first sprint focused on problem contextualization, repository setup, dataset acquisition, and initial exploratory data analysis.

A literature review was conducted to understand the clinical relevance of melanoma detection, its diagnostic challenges, and the role of artificial intelligence in supporting early diagnosis. This step was essential to properly frame the problem and justify the choice of a binary classification approach centered on melanoma detection.

From a technical perspective, the project repository was created and structured with clear folder organization. The HAM10000 dataset was obtained from Kaggle and organized for analysis.

The HAM10000 dataset contains 10,015 dermatoscopic images categorized into seven diagnostic classes, reformulated into a binary problem (melanoma vs. non-melanoma). A key finding is the **severe class imbalance**: melanoma represents only 11.1% of the dataset (1,113 images) while non-melanoma accounts for 88.9% (8,902 images), resulting in an approximate ratio of 8:1. This imbalance has significant implications for modeling — accuracy alone is insufficient, and techniques such as class weighting, focal loss, and targeted augmentation will be necessary.

Within the non-melanoma group, melanocytic nevi (NV) dominate at approximately 67% of the total dataset. This class constitutes the primary set of hard negatives, as nevi can be visually very similar to melanoma, requiring the model to capture subtle visual patterns such as irregular borders, texture variations, and pigment distribution.

Qualitative inspection revealed high intra-class variability including differences in color, texture, lesion shape, and the presence of artifacts such as hair, reflections, and uneven illumination. All images share a standardized resolution of 600×450 pixels, simplifying preprocessing decisions.

Pixel intensity distribution analysis showed that melanoma and non-melanoma images have similar global color statistics, indicating that the classification task requires learning complex morphological and textural features rather than simple color or brightness differences.

Segmentation masks were inspected and confirmed to be present for all images. These masks provide precise lesion localization and will be incorporated as an optional preprocessing step to reduce background noise.

All activities in this sprint were conducted collaboratively. Gabriel Fernando Missaka Mendes focused on literature research, repository setup, and documentation. Eduardo Takei Yaginuma led the data exploration and analysis process.

### Sprint 2 — Feature Engineering and Preprocessing

The second sprint focused on defining the preprocessing pipeline, data augmentation strategy, and dataset splits.

The preprocessing pipeline was defined as: resize to 224×224, normalize with ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), with optional segmentation mask application to isolate lesion regions.

The augmentation strategy for training includes random horizontal and vertical flips, random rotation, color jitter, and random resized cropping — chosen to increase diversity while preserving clinically relevant visual features.

The dataset was split into train (70%), validation (15%), and test (15%) sets with stratification to maintain the class imbalance ratio across all splits. Sample weights were computed for the training set to address the 8:1 imbalance. A batch sanity check was performed to confirm correct label distribution and image normalization.

---

## Project Structure

```
skin-cancer-images-segmentation/
├── data/
│   ├── images/              # Dataset images (not tracked by git)
│   ├── masks/               # Segmentation masks (not tracked by git)
│   └── metadata/            # Train/val/test split CSVs
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Dataset analysis ✅
│   ├── 02_preprocessing.ipynb      # Augmentation & split strategy ✅
│   └── main.ipynb                  # Full training pipeline
├── outputs/
│   ├── figures/             # Plots from notebooks
│   └── models/              # Saved checkpoints (not tracked by git)
├── src/                     # Reusable Python modules
├── docs/                    # Project documents
├── setup_data.py            # Automated dataset download
├── requirements.txt         # Pinned dependencies
├── .gitignore
└── README.md
```

---

## References

AMERICAN CANCER SOCIETY. *Cancer Facts & Figures 2024*. Atlanta: American Cancer Society, 2024.

CARAVIELLO, Camila et al. Melanoma Skin Cancer: A Comprehensive Review of Current Knowledge. *Cancers*, Basel, v. 17, n. 2920, p. 1–35, 2025.

TSCHANDL, P.; ROSENDAHL, C.; KITTLER, H. The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 2018. doi:10.7910/DVN/DBW86T

VIEIRA, Larissa Silva Fontaine; BRANDÃO, Byron José Figueiredo. Diagnosis and prevention of melanoma: a systematic review. *BWS Journal*, v. 5, e220900160, p. 1–10, Sept. 2022.
