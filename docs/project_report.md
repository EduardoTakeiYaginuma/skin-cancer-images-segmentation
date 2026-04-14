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

The system will be trained and evaluated using the HAM10000 dataset, with labels adapted to a binary setting. Given the clinical objective, sensitivity (recall) will be prioritized as the primary evaluation metric, ensuring that melanoma cases are correctly identified.

To align model predictions with clinical use, the classification threshold will be selected based on maximizing sensitivity while maintaining a minimum acceptable level of specificity, ensuring a balance between early detection and false positive control.

Secondary metrics such as specificity and AUC-ROC will be used to provide a comprehensive evaluation of model performance.

## Dataset

The HAM10000 dataset contains 10,015 dermatoscopic images of pigmented skin lesions categorized into seven diagnostic classes. For this project, the dataset is reformulated into a binary classification task: melanoma (positive class) versus non-melanoma (negative class).

A key characteristic of the dataset is its class imbalance, with melanoma cases representing a minority. This reflects real-world clinical distributions and must be addressed during training.

The dataset also includes metadata such as age, sex, and anatomical site, which may be used for further analysis or bias assessment.

Additionally, a segmentation variant of the dataset provides lesion masks. In this project, segmentation will be treated as an optional preprocessing step, used to isolate the region of interest and reduce background noise. It is not treated as a separate modeling task.

## Implementation and Deployment

The implementation prioritizes reproducibility, clarity, and modularity.

### Data Analysis and Preprocessing

An exploratory data analysis (EDA) will be conducted to assess class distribution, metadata patterns, and potential biases. Preprocessing steps include resizing, normalization, artifact removal, and data augmentation techniques such as rotation and flipping.

If segmentation masks are used, they will be applied during preprocessing to focus the model on lesion regions, improving feature extraction without introducing a separate segmentation model.

### Development

The codebase will be version-controlled using GitHub to ensure reproducibility and experiment tracking.

### Deployment Considerations

The integration of the model into a web application (e.g., using Flask or Streamlit) is considered a future step, intended to demonstrate real-world applicability. Similarly, features such as real-time image capture via camera will be treated as possible extensions, rather than core deliverables of this project.

## Modeling

The modeling stage focuses on developing a robust deep learning approach for binary classification of dermatoscopic images (melanoma vs non-melanoma). Based on an initial review of existing implementations and Kaggle benchmarks, particular emphasis will be placed on architectures incorporating U-Net, which demonstrated strong performance in related tasks. In this project, U-Net will be primarily explored as a segmentation-based preprocessing strategy, allowing the model to focus on lesion regions before classification.

In addition to this approach, transfer learning will be applied using pre-trained convolutional neural networks such as ResNet50 and EfficientNet, which will serve as baseline and comparative models. These architectures are widely adopted in medical image classification due to their ability to capture hierarchical visual features.

For classification, the final layers of these networks will be adapted to output a single probability score using a sigmoid activation function. The training process will follow a two-stage strategy: initial training with frozen convolutional layers to preserve general features, followed by fine-tuning of deeper layers to learn domain-specific patterns present in dermatoscopic images.

To address class imbalance, techniques such as class weighting or focal loss will be employed, ensuring greater emphasis on melanoma cases during training. Data augmentation strategies and regularization methods, including dropout and early stopping, will be used to improve generalization and reduce overfitting.

If segmentation is applied, it will be incorporated strictly as a preprocessing step, ensuring that the overall pipeline remains a classification task rather than a separate segmentation problem.

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
- Optional application of segmentation masks as a preprocessing step
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

From a technical perspective, the project repository was created to ensure proper version control and reproducibility. The HAM10000 dataset was then obtained from Kaggle and organized for analysis. An initial data exploration phase was conducted to better understand the dataset's structure, class distribution, and visual characteristics.

The HAM10000 dataset contains 10,015 dermatoscopic images categorized into seven diagnostic classes, which were reformulated into a binary classification problem (melanoma vs non-melanoma) to align with clinical priorities. A key finding from the analysis is the severe class imbalance: melanoma represents only 11.1% of the dataset (1,113 images), while non-melanoma accounts for 88.9% (8,902 images), resulting in an approximate ratio of 8:1. This imbalance has significant implications for modeling, as it can bias the model toward the majority class. Therefore, accuracy alone is not a sufficient metric, and greater emphasis must be placed on sensitivity, recall, and AUC. Additionally, techniques such as class weighting, focal loss, and targeted data augmentation will be necessary to ensure adequate performance in detecting melanoma cases.

Within the non-melanoma group, melanocytic nevi (NV) dominate, representing approximately 67% of the total dataset. This class is particularly important because it constitutes the main set of hard negatives, as nevi can be visually very similar to melanomas. This observation highlights the need for models capable of capturing subtle visual patterns such as irregular borders, texture variations, and pigment distribution, rather than relying on coarse differences between classes.

Qualitative inspection of the images revealed high intra-class variability, including differences in color, texture, lesion shape, and the presence of artifacts such as hair, reflections, and uneven illumination. While this variability increases the complexity of the task, it also makes the dataset more representative of real-world conditions, which is beneficial for model generalization.

Another relevant finding is that all images share a standardized resolution of 600x450 pixels, simplifying preprocessing decisions and allowing resizing to be treated as a design choice rather than a requirement. Analysis of pixel intensity distributions (mean and standard deviation in RGB channels) showed that melanoma and non-melanoma images have similar global color statistics, indicating that the classification task cannot rely on simple color or brightness differences, but instead requires learning more complex morphological and textural features.

The dataset also provides segmentation masks for lesion regions, which represent a valuable additional resource. These masks can be used as an optional preprocessing step to isolate the lesion area and reduce background noise, potentially improving model performance without introducing a separate segmentation model.

All activities in this sprint were conducted collaboratively through online meetings and in-person discussions, ensuring continuous alignment between both authors. While both contributors participated in all stages of the sprint, Gabriel Fernando Mendes Missaka focused more on literature research and project organization, including structuring the repository and documentation. Eduardo Takei Yaginuma contributed primarily to environment setup and led the initial data exploration and analysis process.

Overall, the data exploration phase indicates that the dataset is realistic and challenging, with significant class imbalance, high visual variability, and subtle inter-class differences. These findings directly inform the modeling strategy, emphasizing the need for robust architectures, appropriate handling of class imbalance, careful preprocessing, and evaluation metrics aligned with clinical priorities.

### Sprint 2 - Feature Engineering and Preprocessing

The second sprint focused on transforming the raw HAM10000 segmentation dataset into a reproducible and training-ready pipeline. This stage was implemented in a dedicated preprocessing notebook, designed not only to prepare the data for modeling, but also to export finalized datasets to disk so that the training stage could be executed later without repeating the entire preprocessing workflow.

The first step consisted of validating dataset integrity. The metadata file was loaded and converted into the binary target adopted in the project, with melanoma (MEL) defined as the positive class and all remaining diagnostic categories grouped as non-melanoma. In addition to the binary label, the original seven-class annotation was preserved in the metadata so that specific subclasses, especially melanocytic nevi (NV), could still be tracked during analysis and sampling. At this stage, several consistency checks were performed to ensure that every metadata row had a corresponding image and segmentation mask, that no duplicated image identifiers were present, and that the spatial dimensions of the raw images and masks were uniform across the dataset.

Although the exploratory analysis had already shown that the original images shared a common resolution of 600 x 450 pixels, a deterministic preprocessing pipeline was still required to standardize the final model input. For each sample, the RGB image and binary lesion mask were loaded, thin dark hair artifacts were attenuated through a classical black-hat morphological operation followed by inpainting, and the lesion region was localized using the segmentation mask. A lesion-centered crop with a safety margin was then applied, followed by padding to a square canvas and resizing to a fixed 224 x 224 resolution. This strategy ensured that all samples presented the same spatial dimensions to the model while preserving the lesion as the visual center of the image, instead of relying on a naive global resize of the full dermatoscopic frame.

The segmentation masks also played an important role beyond simple cropping. They were used to quantify lesion coverage and bounding-box proportions, allowing the preprocessing notebook to characterize the effective region of interest across classes. These statistics helped confirm that the lesion occupies very different fractions of the image depending on the diagnosis, reinforcing the usefulness of a lesion-aware preprocessing stage to reduce irrelevant background information before training.

To guarantee reproducibility, the dataset was split into training, validation, and test subsets using a stratified 70% / 15% / 15% partition based on the binary label. This preserved the melanoma prevalence across all subsets and avoided contamination between splits. The same fixed partition is then used throughout the preprocessing and export stages, ensuring that all later experiments compare models on identical train/validation/test membership without rerunning the split logic manually.

Another important step in Sprint 2 was the computation of normalization statistics using only the training split after preprocessing. Channel-wise means and standard deviations were estimated from the processed training images and saved as reusable artifacts. This choice prevents data leakage from validation or test images into preprocessing parameters and keeps the normalization fully aligned with the actual visual distribution seen during training.

Since the dataset is strongly imbalanced, the preprocessing stage also introduced a weighted sampling strategy. Melanoma samples received higher sampling weight to compensate for their lower frequency, while NV samples received an additional boost because they represent the most relevant hard negatives in the binary setting. This decision reflects the clinical objective of the project: maximize melanoma detection while ensuring that the model also learns to distinguish melanoma from lesions that are visually similar.

Data augmentation was implemented as a controlled experimental branch rather than being mixed indiscriminately into the full pipeline. Two parallel training-ready dataset configurations were prepared. The first, a baseline version, applies only deterministic preprocessing and normalization. The second applies the same deterministic preprocessing but adds stochastic augmentation to the training set, including flips, 90-degree rotations, mild affine transformations, small color and contrast perturbations, and light blur. Validation and test sets were intentionally kept deterministic in both cases so that performance comparisons would remain fair and clinically meaningful. In other words, augmentation was treated as a factor to be tested, not as an irreversible preprocessing requirement.

To support direct reuse in the modeling stage, the preprocessing workflow was also extended to export the processed data to disk as folder-based datasets. Two roots were generated: one without augmentation and another with augmentation. In the baseline branch, the data are stored in train, validation, and test folders separated into binary class subfolders. In the augmented branch, the training directory is exported as train_aug and contains the original preprocessed training images together with additional augmented variants, while validation and test are exported as deterministic copies under val_aug and test_aug to preserve a consistent directory structure without introducing stochastic evaluation. This design allows the next modeling stages to start directly from ready-to-train image folders, for example through directory-based loaders such as ImageFolder, without having to rerun lesion cropping, artifact removal, resizing, or export logic.

Overall, Sprint 2 established a complete and reproducible preprocessing pipeline aligned with both the technical and clinical goals of the project. The final result was not only a cleaned and standardized dataset, but also a controlled experimental setup that enables direct comparison between training with and without augmentation, while preserving consistent validation and test conditions.


## References

AMERICAN CANCER SOCIETY. Cancer Facts and Figures 2024. Atlanta: American Cancer Society, 2024.

CARAVIELLO, Camila et al. Melanoma Skin Cancer: A Comprehensive Review of Current Knowledge. Cancers, Basel, v. 17, n. 2920, p. 1-35, 2025.

VIEIRA, Larissa Silva Fontaine; BRANDAO, Byron Jose Figueiredo. Diagnosis and prevention of melanoma: a systematic review. BWS Journal, v. 5, e220900160, p. 1-10, Sept. 2022.

https://www.kaggle.com/datasets/volodymyrpivoshenko/skin-cancer-lesions-segmentation/code
