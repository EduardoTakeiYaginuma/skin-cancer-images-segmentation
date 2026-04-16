# Binary Melanoma Screening from Dermatoscopic Images
### Artificial Intelligence in Medicine and Healthcare

**Eduardo Takei Yaginuma | Gabriel Fernando Mendes Missaka**

## Project Report

## Introduction

### Problem Background

Cutaneous melanoma is the most aggressive and life-threatening form of skin cancer, accounting for a small proportion of cases (approximately 1% to 4%) but responsible for over 80% of skin cancer-related deaths (American Cancer Society, 2024; Vieira and Brandao, 2022). Its high mortality is primarily associated with its strong metastatic potential, particularly after the transition from radial to vertical growth, enabling invasion into deeper skin layers and access to vascular systems (Caraviello et al., 2025).

Early detection is critical for improving patient outcomes. When diagnosed at a localized stage, melanoma presents a 5-year survival rate above 99%, which drops to approximately 35% in cases of distant metastasis (American Cancer Society, 2024). However, accurate diagnosis remains challenging even for experienced dermatologists due to the visual similarity between malignant and benign lesions.

In this context, computational tools based on artificial intelligence can support clinical decision-making, particularly by reducing false negatives, which represent the most critical diagnostic error in melanoma screening.

## Project Proposal

This project proposes the development of a deep learning-based system for binary classification of dermatoscopic images, distinguishing melanoma from non-melanoma lesions.

Instead of addressing a multi-class classification problem, the task is reformulated into a binary setting, grouping all non-melanoma categories (e.g., basal cell carcinoma, benign keratosis, nevi) into a single class. This approach aligns with clinical priorities, focusing specifically on detecting melanoma due to its high lethality.

Special emphasis will be placed on challenging negative samples, lesions that visually resemble melanoma to improve model robustness and reduce false negatives.

The system will be trained and evaluated using the local ISIC 2020 training dataset, which already provides a binary melanoma target. Given the clinical objective, sensitivity (recall) will be prioritized as the primary evaluation metric, ensuring that melanoma cases are correctly identified.

To align model predictions with clinical use, the classification threshold will be selected based on maximizing sensitivity while maintaining a minimum acceptable level of specificity, ensuring a balance between early detection and false positive control.

Secondary metrics such as specificity and AUC-ROC will be used to provide a comprehensive evaluation of model performance.

## Dataset

The current dataset contains 33,126 dermatoscopic images and already encodes the task as a binary classification problem: melanoma (positive class) versus non-melanoma (negative class). In addition to the target, the metadata includes patient identifier, sex, approximate age, anatomical site, diagnosis string, and benign/malignant status.

A key characteristic of this dataset is its extreme imbalance: melanoma accounts for only 584 images (1.76%), while non-melanoma accounts for 32,542 images (98.24%), yielding an approximate ratio of 55.7:1. This reflects a realistic screening scenario, but it also requires explicit balancing and sampling strategies during model development.

Another critical property is the repeated-patient structure. The 33,126 images belong to only 2,056 unique patients, and 427 patients contain both positive and negative images. This makes image-level random splitting unsafe because it can leak patient-specific information across training, validation, and test sets.

The dataset also includes useful but partially incomplete metadata, especially for anatomical site. These variables remain valuable for exploratory analysis, bias assessment, and descriptive reporting, but the modeling pipeline should not depend on them being fully populated.

## Implementation and Deployment

The implementation prioritizes reproducibility, clarity, and modularity.

### Data Analysis and Preprocessing

An exploratory data analysis (EDA) will be conducted to assess class distribution, metadata patterns, patient repetition, image resolution variability, and potential sources of bias. Preprocessing steps include patient-aware splitting, train-only class balancing, resizing, normalization, artifact removal, and data augmentation techniques such as rotation and flipping.

### Development

The codebase will be version-controlled using GitHub to ensure reproducibility and experiment tracking.

### Deployment Considerations

The integration of the model into a web application (e.g., using Flask or Streamlit) is considered a future step, intended to demonstrate real-world applicability. Similarly, features such as real-time image capture via camera will be treated as possible extensions, rather than core deliverables of this project.

## Modeling

The modeling stage focuses on developing a robust deep learning approach for binary classification of dermatoscopic images (melanoma vs non-melanoma). Particular emphasis will be placed on transfer learning with architectures that are well established for dermatoscopic image analysis and can be trained reliably under severe class imbalance.

In addition to this approach, transfer learning will be applied using pre-trained convolutional neural networks such as ResNet50 and EfficientNet, which will serve as baseline and comparative models. These architectures are widely adopted in medical image classification due to their ability to capture hierarchical visual features.

For classification, the final layers of these networks will be adapted to output a single probability score using a sigmoid activation function. The training process will follow a two-stage strategy: initial training with frozen convolutional layers to preserve general features, followed by fine-tuning of deeper layers to learn domain-specific patterns present in dermatoscopic images.

To address class imbalance, techniques such as class weighting, weighted sampling, or focal loss will be employed, ensuring greater emphasis on melanoma cases during training. Data augmentation strategies and regularization methods, including dropout and early stopping, will be used to improve generalization and reduce overfitting.

## Evaluation Strategy

Model performance will be evaluated with a strong emphasis on sensitivity (recall), aiming to minimize false negatives due to their clinical impact. The classification threshold will not be fixed at 0.5; instead, it will be selected based on validation data, prioritizing high sensitivity while maintaining a minimum acceptable level of specificity.

To ensure a reliable and unbiased evaluation, the dataset will be split into training, validation, and test sets, with a portion of the images held out exclusively for final testing. This test set will not be used during model training or hyperparameter tuning, allowing for a fair assessment of the model's generalization performance.

Additionally, AUC-ROC will be used to assess performance across different thresholds, and model calibration may be considered to improve the reliability of predicted probabilities.

## Project Organization and Authors' Contributions

The project is organized into five development sprints to ensure a structured and iterative workflow, while also reflecting a clear division of responsibilities between the authors.

### Sprint 1 - Data Exploration

- Exploratory Data Analysis (EDA)
- Analysis of class distribution and imbalance
- Inspection of image quality and resolution
- Exploration of metadata (age, sex, anatomical site)
- Identification of potential biases and data issues

### Sprint 2 - Feature Engineering and Preprocessing

- Image resizing and normalization
- Data augmentation (rotation, flipping, color transformations)
- Artifact and noise reduction (e.g., hair removal)
- Patient-aware split and train-only class balancing
- Dataset splitting (train, validation, test)

### Sprint 3 - Modeling

- Selection of baseline architecture (e.g., ResNet50, EfficientNet)
- Implementation of transfer learning
- Training with frozen layers and fine-tuning
- Handling class imbalance (class weights or focal loss)
- Initial model training and baseline performance assessment

### Sprint 4 - Validation and Optimization

- Hyperparameter tuning
- Threshold selection prioritizing sensitivity with minimum specificity
- Performance evaluation using validation set (Recall, Specificity, AUC-ROC)
- Error analysis and model refinement
- Regularization strategies (dropout, early stopping)

### Sprint 5 - Testing and Finalization

- Final evaluation on a held-out test set
- Analysis of generalization performance
- Model calibration (if applicable)
- Documentation and preparation of the final report
- Optional deployment as a web application (future work)

The project tasks were divided to ensure both specialization and collaboration across all stages of development. Gabriel Fernando Mendes Missaka led the data exploration and feature engineering phases, including exploratory data analysis, dataset preprocessing, and the design of data augmentation strategies. Eduardo Takei Yaginuma was primarily responsible for the modeling, validation, and testing stages, including the implementation of deep learning architectures, training procedures, and performance evaluation.

Both authors collaborated across all stages of the project, contributing to decision-making, experimental design, and iterative improvements to the data pipeline and model performance.

## Development

### Sprint 1 - Data Exploration

The first sprint focused on problem contextualization, dataset acquisition, and initial exploratory data analysis. Initially, a literature review was conducted to better understand the clinical relevance of melanoma detection, its diagnostic challenges, and the role of artificial intelligence in supporting early diagnosis. This step was essential to properly frame the problem and justify the choice of a binary classification approach centered on melanoma detection.

From a technical perspective, the project repository was created to ensure proper version control and reproducibility. The local ISIC 2020 training dataset was then organized under `data/` and inspected through a dedicated exploration notebook to understand its structure, class distribution, patient-level repetition, metadata quality, and visual characteristics.

The exploratory analysis showed that the dataset is much larger than the one used in the earlier project version, but also substantially more imbalanced. Among 33,126 images, only 584 correspond to melanoma, while 32,542 correspond to non-melanoma, producing a ratio of approximately 55.7:1. This level of imbalance makes raw accuracy especially misleading and reinforces the need to prioritize sensitivity, recall, and AUC-oriented evaluation, alongside explicit imbalance mitigation during training.

One of the most important findings was that the image collection is highly patient-dependent. The full dataset contains only 2,056 unique patients, meaning that each patient contributes multiple images on average, and 427 patients contain both positive and negative lesions. This observation has direct methodological implications: any random image-level split would risk leakage between train, validation, and test sets. As a result, patient-aware splitting became a central design requirement rather than a secondary implementation detail.

The metadata profile also introduced new insights. The diagnostic labels are heavily dominated by `unknown`, which limits fine-grained subclass interpretation on the negative class. Even so, `nevus` remains the most relevant named hard-negative diagnosis, which is clinically important because nevi can still resemble melanoma visually. In addition, the metadata columns for sex, age, and anatomical site are usable but not perfectly complete, particularly for anatomical site, so they are better suited for exploration and bias assessment than as mandatory modeling inputs.

Qualitative inspection of the images revealed high intra-class variability, including differences in color, texture, lesion shape, framing, and the presence of artifacts such as hair, reflections, and uneven illumination. Another relevant finding is that image resolution is no longer uniform: the dataset mixes multiple acquisition formats, including high-resolution and low-resolution captures. Consequently, shape normalization is no longer optional and must be handled explicitly during preprocessing.

Analysis of global RGB statistics further suggested that melanoma and non-melanoma images remain broadly similar in their overall intensity distribution. Therefore, the classification task cannot rely on simple global brightness or color differences alone, but instead requires the model to learn richer morphological and textural patterns from heterogeneous dermatoscopic images.

All activities in this sprint were conducted collaboratively through online meetings and in-person discussions, ensuring continuous alignment between both authors. While both contributors participated in all stages of the sprint, Gabriel Fernando Mendes Missaka focused more on literature research and project organization, including structuring the repository and documentation. Eduardo Takei Yaginuma contributed primarily to environment setup and led the initial data exploration and analysis process.

Overall, the data exploration phase indicates that the dataset is realistic and challenging, with significant class imbalance, high visual variability, and subtle inter-class differences. These findings directly inform the modeling strategy, emphasizing the need for robust architectures, appropriate handling of class imbalance, careful preprocessing, and evaluation metrics aligned with clinical priorities.

### Sprint 2 - Feature Engineering and Preprocessing

The second sprint marked an important change in the project: the original dataset used in the earlier version of the pipeline was replaced by the ISIC 2020 training dataset. This transition required more than a simple path update. Because the new dataset is larger, more imbalanced, richer in metadata, and structurally different from the previous one, the preprocessing strategy had to be redesigned based on the findings obtained during the new data exploration stage.

The first consequence of this change was a full revalidation of the data source. The new metadata file was loaded in its native binary format and matched against the image folder under `data/train/`. This verification confirmed that the dataset contained 33,126 images, all mapped correctly to metadata rows, with no duplicated image identifiers. At the same time, the exploration notebook revealed a much more severe class imbalance than before: only 584 images correspond to melanoma, while 32,542 belong to the non-melanoma class, yielding an approximate ratio of 55.7:1. This finding directly affected the design of the training pipeline, since a naive use of the raw distribution would strongly bias the model toward the negative class.

The second major discovery was the patient structure of the dataset. Although the image count is large, the data correspond to only 2,056 unique patients, and 427 of them contain both positive and negative lesions. This meant that the split strategy used in the earlier project version was no longer adequate. Instead of splitting at image level, Sprint 2 introduced a patient-aware split as a core preprocessing rule. Using a 70% / 15% / 15% partition over patient groups, the pipeline ensures that no patient appears in more than one subset, reducing the risk of leakage and making validation and test estimates more reliable.

The exploratory analysis also showed that the diagnostic metadata are highly concentrated in the `unknown` label, while `nevus` remains the most relevant named hard-negative diagnosis. This observation influenced the balancing strategy adopted in preprocessing. After the raw split was defined, balancing was applied only to the training subset: all melanoma samples were kept, while non-melanoma examples were downsampled to a configurable ratio with diagnosis-aware stratification whenever possible. Validation and test subsets, in contrast, were preserved closer to the natural raw distribution so that model comparison and final evaluation would remain realistic.

Another important insight from the new exploration was that image geometry is heterogeneous. Unlike the earlier dataset version, the ISIC 2020 collection mixes multiple resolutions and acquisition formats. As a result, geometric normalization became mandatory. Because the new dataset does not provide segmentation masks in the same workflow, the preprocessing notebook was redesigned from a lesion-mask-centric approach to an image-centric one. For each sample, the RGB image is loaded, hair artifacts are attenuated through a black-hat morphological operation followed by inpainting, the frame is padded to a square canvas, and the result is resized to a fixed 224 x 224 resolution. This preserves a deterministic and reproducible pipeline without depending on lesion masks.

The metadata exploration also showed that sex, age, and anatomical site are informative but incomplete, particularly anatomical site. For that reason, these variables were preserved in the exported manifests for later descriptive analysis and potential bias assessment, but they were not treated as mandatory inputs for image preprocessing. In parallel, channel-wise normalization statistics were computed using only the balanced training split after deterministic preprocessing, preventing leakage from validation or test subsets.

Since the dataset remains extremely imbalanced even after train-only downsampling, Sprint 2 also incorporated weighting and augmentation decisions guided by the exploration results. Melanoma samples received higher sampling weight, and nevus samples received an additional boost because they remain a clinically relevant hard-negative group. Two export branches were then prepared: one with only deterministic preprocessing and another with the same preprocessing plus controlled offline augmentation in the training split. This produced a reproducible experimental structure in which training with and without augmentation can be compared directly under the same validation and test conditions.

Overall, Sprint 2 was not only a preprocessing sprint, but also an adaptation sprint driven by the change of dataset. The result was a new training-ready pipeline aligned with the actual properties of the ISIC 2020 data, incorporating the main lessons from exploration: extreme imbalance, repeated patients, heterogeneous image geometry, partially incomplete metadata, and the need for a patient-safe evaluation protocol.


## References

AMERICAN CANCER SOCIETY. Cancer Facts and Figures 2024. Atlanta: American Cancer Society, 2024.

CARAVIELLO, Camila et al. Melanoma Skin Cancer: A Comprehensive Review of Current Knowledge. Cancers, Basel, v. 17, n. 2920, p. 1-35, 2025.

VIEIRA, Larissa Silva Fontaine; BRANDAO, Byron Jose Figueiredo. Diagnosis and prevention of melanoma: a systematic review. BWS Journal, v. 5, e220900160, p. 1-10, Sept. 2022.

https://www.kaggle.com/datasets/volodymyrpivoshenko/skin-cancer-lesions-segmentation/code
