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

The second sprint began with a concern that emerged directly from the first exploratory analyses: the number of melanoma images available for training was still limited for a clinically sensitive binary classification problem. Because melanoma is the positive class and also the most important class from the medical point of view, this limitation represented a concrete risk for model learning, evaluation stability, and generalization.

To address this issue, we searched for a larger alternative dataset and temporarily migrated the pipeline to a new source that was roughly twenty times larger in total volume. The expectation was that a much larger dataset would substantially increase the number of melanoma samples and therefore justify a change in the data pipeline. However, the additional exploration showed that, despite the strong increase in the total number of images, the number of melanoma examples available for our use case increased only marginally, on the order of about one hundred additional positive samples. In practice, this meant that the cost of changing dataset structure, metadata format, and preprocessing assumptions was not compensated by a meaningful gain in the positive class.

Based on this finding, we decided to return to the original dataset and focus on extracting more value from it through preprocessing and controlled augmentation. This decision was motivated by two advantages of the original source. First, it already had a cleaner structure for the project objective, with images, masks, and labels directly aligned. Second, it provided lesion segmentation masks, which made it possible to build a lesion-aware preprocessing pipeline instead of relying only on full-image resizing.

The final preprocessing workflow implemented in this sprint therefore starts by validating the local dataset structure, confirming that all metadata rows have matching images and masks and that the spatial dimensions are consistent across files. The seven original classes are preserved in the metadata for analysis, but the modeling target is converted to the binary setting adopted by the project: melanoma versus non-melanoma. Since the original class distribution is still imbalanced, all melanoma images are kept while only the negative class is downsampled, preserving the non-melanoma subclass mix as much as possible. This creates an effective dataset that is more suitable for training without discarding the positive class.

Once the effective dataset is defined, the preprocessing stage applies a deterministic lesion-centric pipeline. For each sample, the RGB image and its binary segmentation mask are loaded, thin hair artifacts are attenuated with a classical black-hat morphological operation followed by inpainting, the lesion region is localized using the mask, and a crop is extracted around the lesion with a safety margin. The cropped image is then padded to a square format and resized to a fixed resolution of 224 x 224 pixels. This procedure standardizes the inputs while preserving the lesion as the central visual structure, which is more appropriate than a naive global resize of the full dermoscopic frame.

After preprocessing, the effective dataset is split into training, validation, and test subsets using a reproducible 70% / 15% / 15% stratified partition based on the binary label. Train-only normalization statistics are then computed from the processed images in order to avoid information leakage from validation or test sets. To further mitigate the imbalance problem during model training, the pipeline also includes a weighted sampling strategy in which melanoma receives higher importance and melanocytic nevi are given additional emphasis as hard negatives, since they remain the most visually relevant contrast group.

Another important outcome of this sprint was the creation of two training-ready branches for the next experiments. The first branch uses only deterministic preprocessing and serves as the baseline condition. In this branch, the effective dataset contains 4,452 images, of which 1,113 are melanoma and 3,339 are non-melanoma. After the 70% / 15% / 15% stratified split, the training set contains 779 melanoma images and 2,337 non-melanoma images, while validation and test each contain 167 melanoma and 501 non-melanoma images. The second branch uses the same preprocessing but adds controlled offline augmentation in the training split, including flips, rotations, mild geometric transformations, and brightness or contrast perturbations. Since two additional augmented copies are generated for each training image while the original samples are kept, the augmented training set grows to 9,348 images, with 2,337 melanoma samples and 7,011 non-melanoma samples. In other words, augmentation substantially increases the volume of the training data, although it preserves the same class ratio of 3:1 already defined in the effective dataset.

In summary, Sprint 2 evolved from a search for more melanoma images into a more grounded data-engineering decision. Instead of adopting a larger dataset that did not significantly improve the positive class, we returned to the original source and built a more robust preprocessing pipeline around it. The final outcome is a controlled experimental setup in which the baseline branch preserves 779 melanoma and 2,337 non-melanoma images in training, while the augmented branch expands this to 2,337 melanoma and 7,011 non-melanoma training images through synthetic variation. Going forward, the project will use this finalized dataset preparation stage to test augmentation strategies, compare baseline and augmented training setups, and evaluate whether these techniques can improve melanoma detection without compromising reproducibility or clinical relevance.


### Sprint 3 - Modeling

The third sprint marked the transition from data preparation to the first concrete modeling experiments. During this phase, we reorganized the training pipeline so that preprocessing and data augmentation became two clearly separated stages. Preprocessing remained responsible for deterministic lesion-centered image preparation, while augmentation was isolated as an independent step focused only on increasing training variability. This separation improved the structure of the codebase and made the experimental pipeline easier to understand, compare, and maintain.

After consolidating this organization, we began training two different models for the binary melanoma versus non-melanoma task. The first model was developed by our team and represents our main custom modeling line. The second model was adapted from a ready-made Kaggle implementation, which we used as a practical baseline for comparison. The goal of training both was not yet to select a final architecture, but rather to validate the end-to-end pipeline and obtain initial evidence about whether the current data preparation and training setup were producing meaningful learning behavior.

We were able to start training both approaches and obtain preliminary results. These early outputs were important because they showed that the current workflow is functional: the datasets are correctly exported, the models can be trained, and the first metrics can already be monitored. However, these results should still be interpreted as exploratory rather than definitive, since the training strategy itself revealed limitations that affect both the methodological quality of the experiments and the practical speed of iteration.

After discussing the initial experiments with the professor, two main problems were identified. The first concerns the way data augmentation is currently applied. At this stage, augmentation is being generated offline, that is, before training, by exporting fixed augmented copies of the training images. Although this increases the apparent size of the dataset, it also causes repeated image versions to appear across training batches. As a result, the model does not benefit from the full variability that online augmentation could provide, where each pass through the data can generate new random transformations. For this reason, we concluded that augmentation should be moved into the training loop in the next sprint, so that it is applied dynamically during batch loading rather than statically during dataset export.

The second problem identified is computational. Training is currently too slow in the available environment, which limits the number of experiments we can run and makes it difficult to iterate on architectures, hyperparameters, and loss configurations. This is especially relevant now that the project is entering a phase in which more experimentation will be necessary. Because of this, we decided that the next sprint should also include the allocation of more powerful AWS machines in order to reduce training time and make broader experimentation feasible.

Therefore, Sprint 3 should be understood as an initial modeling sprint that successfully launched the experimental phase but also exposed two structural bottlenecks: augmentation strategy and computational capacity. Both issues are now clearly defined and will be treated as priorities in Sprint 4. In practical terms, the next sprint will focus on migrating augmentation to an online training setup and improving the hardware environment used for model training so that subsequent evaluations can be carried out more efficiently and under a more appropriate experimental design.


## References

AMERICAN CANCER SOCIETY. Cancer Facts and Figures 2024. Atlanta: American Cancer Society, 2024.

CARAVIELLO, Camila et al. Melanoma Skin Cancer: A Comprehensive Review of Current Knowledge. Cancers, Basel, v. 17, n. 2920, p. 1-35, 2025.

VIEIRA, Larissa Silva Fontaine; BRANDAO, Byron Jose Figueiredo. Diagnosis and prevention of melanoma: a systematic review. BWS Journal, v. 5, e220900160, p. 1-10, Sept. 2022.

https://www.kaggle.com/datasets/volodymyrpivoshenko/skin-cancer-lesions-segmentation/code
