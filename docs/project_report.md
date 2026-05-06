# Binary Melanoma Screening from Dermatoscopic Images

**Artificial Intelligence in Medicine and Healthcare**

**Authors**

- Eduardo Takei Yaginuma
- Gabriel Fernando Mendes Missaka

## Project Report

### Introduction

#### Problem Background

Cutaneous melanoma is the most aggressive and life-threatening form of skin cancer, accounting for a small proportion of cases (approximately 1%–4%) but responsible for over 80% of skin cancer-related deaths (American Cancer Society, 2024; Vieira & Brandão, 2022). Its high mortality is primarily associated with its strong metastatic potential, particularly after the transition from radial to vertical growth, enabling invasion into deeper skin layers and access to vascular systems (Caraviello et al., 2025).

Early detection is critical for improving patient outcomes. When diagnosed at a localized stage, melanoma presents a 5-year survival rate above 99%, which drops to approximately 35% in cases of distant metastasis (American Cancer Society, 2024). However, accurate diagnosis remains challenging even for experienced dermatologists due to the visual similarity between malignant and benign lesions.

In this context, computational tools based on artificial intelligence can support clinical decision-making, particularly by reducing false negatives, which represent the most critical diagnostic error in melanoma screening.

#### Project Proposal

This project proposes the development of a deep learning-based system for binary classification of dermatoscopic images, distinguishing melanoma from non-melanoma lesions.

Instead of addressing a multi-class classification problem, the task is reformulated into a binary setting, grouping all non-melanoma categories (e.g., basal cell carcinoma, benign keratosis, nevi) into a single class. This approach aligns with clinical priorities, focusing specifically on detecting melanoma due to its high lethality.

Special emphasis will be placed on challenging negative samples, lesions that visually resemble melanoma to improve model robustness and reduce false negatives.

The system will be trained and evaluated using the HAM10000 dataset, with labels adapted to a binary setting. Given the clinical objective, sensitivity (recall) will be prioritized as the primary evaluation metric, ensuring that melanoma cases are correctly identified.

To align model predictions with clinical use, the classification threshold will be selected based on maximizing sensitivity while maintaining a minimum acceptable level of specificity, ensuring a balance between early detection and false positive control.

Secondary metrics such as specificity and AUC-ROC will be used to provide a comprehensive evaluation of model performance.

### Dataset

The HAM10000 dataset contains 10,015 dermatoscopic images of pigmented skin lesions categorized into seven diagnostic classes. For this project, the dataset is reformulated into a binary classification task: melanoma (positive class) versus non-melanoma (negative class).

A key characteristic of the dataset is its class imbalance, with melanoma cases representing a minority. This reflects real-world clinical distributions and must be addressed during training.

The dataset also includes metadata such as age, sex, and anatomical site, which may be used for further analysis or bias assessment.

Additionally, a segmentation variant of the dataset provides lesion masks. In this project, segmentation will be treated as an optional preprocessing step, used to isolate the region of interest and reduce background noise. It is not treated as a separate modeling task.

### Implementation and Deployment

The implementation prioritizes reproducibility, clarity, and modularity.

#### Data Analysis and Preprocessing

An exploratory data analysis (EDA) will be conducted to assess class distribution, metadata patterns, and potential biases. Preprocessing steps include resizing, normalization, artifact removal, and data augmentation techniques such as rotation and flipping.

If segmentation masks are used, they will be applied during preprocessing to focus the model on lesion regions, improving feature extraction without introducing a separate segmentation model.

#### Development

The codebase will be version-controlled using GitHub to ensure reproducibility and experiment tracking.

#### Deployment Considerations

The integration of the model into a web application (e.g., using Flask or Streamlit) is considered a future step, intended to demonstrate real-world applicability. Similarly, features such as real-time image capture via camera will be treated as possible extensions, rather than core deliverables of this project.

### Modeling

The modeling stage focuses on developing a robust deep learning approach for binary classification of dermatoscopic images (melanoma vs non-melanoma). Based on an initial review of existing implementations and Kaggle benchmarks, particular emphasis will be placed on architectures incorporating U-Net, which demonstrated strong performance in related tasks. In this project, U-Net will be primarily explored as a segmentation-based preprocessing strategy, allowing the model to focus on lesion regions before classification.

In addition to this approach, transfer learning will be applied using pre-trained convolutional neural networks such as ResNet50 and EfficientNet, which will serve as baseline and comparative models. These architectures are widely adopted in medical image classification due to their ability to capture hierarchical visual features.

For classification, the final layers of these networks will be adapted to output a single probability score using a sigmoid activation function. The training process will follow a two-stage strategy: initial training with frozen convolutional layers to preserve general features, followed by fine-tuning of deeper layers to learn domain-specific patterns present in dermatoscopic images.

To address class imbalance, techniques such as class weighting or focal loss will be employed, ensuring greater emphasis on melanoma cases during training. Data augmentation strategies and regularization methods, including dropout and early stopping, will be used to improve generalization and reduce overfitting.

If segmentation is applied, it will be incorporated strictly as a preprocessing step, ensuring that the overall pipeline remains a classification task rather than a separate segmentation problem.

### Evaluation Strategy

Model performance will be evaluated with a strong emphasis on sensitivity (recall), aiming to minimize false negatives due to their clinical impact. The classification threshold will not be fixed at 0.5; instead, it will be selected based on validation data, prioritizing high sensitivity while maintaining a minimum acceptable level of specificity.

To ensure a reliable and unbiased evaluation, the dataset will be split into training, validation, and test sets, with a portion of the images held out exclusively for final testing. This test set will not be used during model training or hyperparameter tuning, allowing for a fair assessment of the model's generalization performance.

Additionally, AUC-ROC will be used to assess performance across different thresholds, and model calibration may be considered to improve the reliability of predicted probabilities.

### Project Organization and Authors' Contributions

The project is organized into five development sprints to ensure a structured and iterative workflow, while also reflecting a clear division of responsibilities between the authors.

#### Sprint 1 – Data Exploration

- Exploratory Data Analysis (EDA)
- Analysis of class distribution and imbalance
- Inspection of image quality and resolution
- Exploration of metadata (age, sex, anatomical site)
- Identification of potential biases and data issues

#### Sprint 2 – Feature Engineering and Preprocessing

- Image resizing and normalization
- Data augmentation (rotation, flipping, color transformations)
- Artifact and noise reduction (e.g., hair removal)
- Optional application of segmentation masks as a preprocessing step
- Dataset splitting (train, validation, test)

#### Sprint 3 – Modeling

- Selection of baseline architecture (e.g., ResNet50, EfficientNet)
- Implementation of transfer learning
- Training with frozen layers and fine-tuning
- Handling class imbalance (class weights or focal loss)
- Initial model training and baseline performance assessment

#### Sprint 4 – Validation and Optimization

- Hyperparameter tuning
- Threshold selection prioritizing sensitivity with minimum specificity
- Performance evaluation using validation set (Recall, Specificity, AUC-ROC)
- Error analysis and model refinement
- Regularization strategies (dropout, early stopping)

#### Sprint 5 – Testing and Finalization

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

Another relevant finding is that all images share a standardized resolution of 600×450 pixels, simplifying preprocessing decisions and allowing resizing to be treated as a design choice rather than a requirement. Analysis of pixel intensity distributions (mean and standard deviation in RGB channels) showed that melanoma and non-melanoma images have similar global color statistics, indicating that the classification task cannot rely on simple color or brightness differences, but instead requires learning more complex morphological and textural features.

The dataset also provides segmentation masks for lesion regions, which represent a valuable additional resource. These masks can be used as an optional preprocessing step to isolate the lesion area and reduce background noise, potentially improving model performance without introducing a separate segmentation model.

All activities in this sprint were conducted collaboratively through online meetings and in-person discussions, ensuring continuous alignment between both authors. While both contributors participated in all stages of the sprint, Gabriel Fernando Mendes Missaka focused more on literature research and project organization, including structuring the repository and documentation. Eduardo Takei Yaginuma contributed primarily to environment setup and led the initial data exploration and analysis process.

Overall, the data exploration phase indicates that the dataset is realistic and challenging, with significant class imbalance, high visual variability, and subtle inter-class differences. These findings directly inform the modeling strategy, emphasizing the need for robust architectures, appropriate handling of class imbalance, careful preprocessing, and evaluation metrics aligned with clinical priorities.

### Sprint 2 - Preprocessing

The second sprint began with a concern that emerged directly from the first exploratory analyses: the number of melanoma images available for training was still limited for a clinically sensitive binary classification problem. Because melanoma is the positive class and also the most important class from the medical point of view, this limitation represented a concrete risk for model learning, evaluation stability, and generalization.

To address this issue, we searched for a larger alternative dataset and temporarily migrated the pipeline to a new source that was roughly twenty times larger in total volume. The expectation was that a much larger dataset would substantially increase the number of melanoma samples and therefore justify a change in the data pipeline. However, the additional exploration showed that, despite the strong increase in the total number of images (`20x`), the number of melanoma examples available for our use case increased only marginally, on the order of about one hundred additional positive samples. In practice, this meant that the cost of changing dataset structure, metadata format, and preprocessing assumptions was not compensated by a meaningful gain in the positive class.

Based on this finding, we decided to return to the original dataset and focus on extracting more value from it through preprocessing and controlled augmentation. This decision was motivated by two advantages of the original source. First, it already had a cleaner structure for the project objective, with images, masks, and labels directly aligned. Second, it provided lesion segmentation masks, which made it possible to build a lesion-aware preprocessing pipeline instead of relying only on full-image resizing.

The final preprocessing workflow implemented in this sprint therefore starts by validating the local dataset structure, confirming that all metadata rows have matching images and masks and that the spatial dimensions are consistent across files. The seven original classes are preserved in the metadata for analysis, but the modeling target is converted to the binary setting adopted by the project: melanoma versus non-melanoma. Since the original class distribution is still imbalanced, all melanoma images are kept while only the negative class is downsampled, preserving the non-melanoma subclass mix as much as possible. This creates an effective dataset that is more suitable for training without discarding the positive class.

Once the effective dataset is defined, the preprocessing stage applies a deterministic lesion-centric pipeline. For each sample, the RGB image and its binary segmentation mask are loaded, thin hair artifacts are attenuated with a classical black-hat morphological operation followed by inpainting, the lesion region is localized using the mask, and a crop is extracted around the lesion with a safety margin. The cropped image is then padded to a square format and resized to a fixed resolution of `224 x 224` pixels. This procedure standardizes the inputs while preserving the lesion as the central visual structure, which is more appropriate than a naive global resize of the full dermoscopic frame.

After preprocessing, the effective dataset is split into training, validation, and test subsets using a reproducible `70% / 15% / 15%` stratified partition based on the binary label. Train-only normalization statistics are then computed from the processed images in order to avoid information leakage from validation or test sets. To further mitigate the imbalance problem during model training, the pipeline also includes a weighted sampling strategy in which melanoma receives higher importance and melanocytic nevi are given additional emphasis as hard negatives, since they remain the most visually relevant contrast group.

Another important outcome of this sprint was the creation of two training-ready branches for the next experiments. The first branch uses only deterministic preprocessing and serves as the baseline condition. In this branch, the effective dataset contains 4,452 images, of which 1,113 are melanoma and 3,339 are non-melanoma. After the `70% / 15% / 15%` stratified split, the training set contains 779 melanoma images and 2,337 non-melanoma images, while validation and test each contain 167 melanoma and 501 non-melanoma images. The second branch uses the same preprocessing but adds controlled offline augmentation in the training split, including flips, rotations, mild geometric transformations, and brightness or contrast perturbations. Since two additional augmented copies are generated for each training image while the original samples are kept, the augmented training set grows to 9,348 images, with 2,337 melanoma samples and 7,011 non-melanoma samples. In other words, augmentation substantially increases the volume of the training data, although it preserves the same class ratio of 3:1 already defined in the effective dataset.

### Sprint 3 – Modeling

The third sprint marked the transition from data preparation to the initial modeling phase, focusing on establishing a functional training pipeline and conducting the first experimental runs. At this stage, the primary objective was not to finalize model selection, but to validate the end-to-end workflow and ensure that the data processing and training components were correctly integrated.

To improve the clarity and modularity of the pipeline, preprocessing and data augmentation were reorganized into two distinct stages. Preprocessing remained a deterministic step, responsible for lesion-centered image preparation, including cropping, resizing, and normalization. In contrast, data augmentation was isolated as a separate component aimed at increasing variability in the training data. This structural separation improved code maintainability and enabled clearer comparisons between different experimental configurations.

Following this reorganization, two modeling approaches were implemented. The first corresponds to a custom model developed by the project team, representing the primary experimental direction. The second model was adapted from an existing Kaggle implementation and used as a practical baseline for comparison. The goal of training both models at this stage was to verify that the pipeline could successfully support multiple architectures and to obtain preliminary performance signals, rather than to conduct a definitive comparative evaluation.

Initial training runs were successfully executed for both models, confirming that the pipeline is operational: datasets are correctly prepared and loaded, models can be trained without errors, and performance metrics can be monitored throughout the training process. These early results provided important validation that the overall workflow is functioning as intended. However, they should be interpreted as exploratory, as limitations in the current setup were identified during experimentation.

Based on discussions and feedback, two main issues were recognized. The first relates to the augmentation strategy. At this stage, data augmentation is applied offline, meaning that augmented images are generated prior to training and stored as fixed samples in the dataset. While this approach increases the dataset size, it limits variability during training, as the same augmented versions are repeatedly presented to the model. This reduces the potential benefits of augmentation compared to an online approach, where transformations are applied dynamically at each training iteration. As a result, migrating augmentation to an online, on-the-fly strategy was identified as a priority for the next sprint.

The second issue concerns computational efficiency. Training time in the current environment proved to be a significant bottleneck, limiting the number of experiments that can be conducted and slowing down iteration cycles. This constraint is particularly critical at this stage of the project, where multiple architectures, hyperparameters, and training strategies must be explored. To address this limitation, it was decided that the next sprint will include the use of more powerful computational resources, specifically through the allocation of AWS instances, to enable faster and more scalable experimentation.

In summary, Sprint 3 successfully established the initial modeling pipeline and validated the feasibility of the experimental setup. At the same time, it revealed two key limitations, augmentation strategy and computational capacity, that must be addressed to support more robust experimentation. These insights define the main objectives for Sprint 4, which will focus on implementing online data augmentation and improving the training infrastructure to enable more efficient and reliable model development.

### Sprint 4 - Validation and Optimization

The fourth sprint focused on converting the initial modeling pipeline into a controlled benchmark for validation and optimization. Using the treated image manifest produced in preprocessing, we preserved a single stratified `train / validation / test` split across all experiments to ensure fair comparison. The balanced training set contained 779 melanoma and 2,337 non-melanoma images, while validation and test each contained 167 melanoma cases and more than 1,300 non-melanoma cases. This design allowed the models to be compared under identical data conditions while maintaining a clinically realistic evaluation set.

One of the main technical changes in this sprint was the migration from offline augmentation to online augmentation. Instead of training on fixed augmented copies, the pipeline now applies random transformations dynamically during training, including horizontal and vertical flips, `RandomRotate90`, small geometric perturbations, and mild brightness and contrast variation. This directly addressed the limitation identified in Sprint 3 by increasing effective variability at each epoch without artificially freezing the augmented samples.

Class imbalance was addressed at the data level by downsampling non-melanoma cases to a 3:1 ratio relative to melanoma. The training loop further uses a weighted sampler to give additional emphasis to melanocytic nevi (`NV`) as hard negatives. Because the dataset was already explicitly balanced through downsampling, we removed the `pos_weight` argument from `BCEWithLogitsLoss`, which had previously produced a double correction of the imbalance and was distorting the calibration of output probabilities.

Another important change was the threshold selection criterion. The sensitivity target for selecting `T_HIGH` was adjusted from `0.95` to `0.85`, producing a more conservative and better-calibrated threshold. Previous experiments with a `0.95` target caused some models, particularly `EfficientNet-B0` at `64×64`, to collapse to near-zero thresholds due to the aggressiveness of the recall requirement. The revised target maintained clinical relevance while producing more stable and interpretable threshold values across all eight configurations.

To evaluate the eight experiments (two architectures × four configurations), we trained both `EfficientNet-B0` and `ResNet50` with `BCEWithLogitsLoss`, `AdamW` optimizer, cosine annealing scheduler, and early stopping based on validation AUC. In addition to standard binary classification metrics, we implemented a dual-threshold three-zone system. A lower threshold, `T_LOW`, was defined as the `2%` quantile of melanoma probabilities on the validation set, producing three clinically interpretable zones: `Non-Melanoma`, `Possible Melanoma`, and `Melanoma`. This allowed the analysis to quantify how safely low-confidence predictions could be automatically dismissed, and how much clinical review burden the uncertain zone would impose in practice.

The benchmark results confirmed a consistent advantage for `ResNet50` over `EfficientNet-B0`, especially at `224×224` resolution with online augmentation. The best model according to the composite clinical score was `ResNet50 | aug_224x224`, which achieved a test AUC of `0.9128`.

| Model | Experiment | AUC | Mel. captured | Mel. hidden | Uncertain zone | Clinical score |
|-------|-----------|-----|--------------|------------|----------------|----------------|
| ResNet50 | aug_224x224 | 0.9128 | 85.63% | 2.99% | 22.62% | 0.45617 |
| ResNet50 | base_224x224 | 0.8901 | 87.43% | 4.79% | 16.50% | 0.45583 |
| EfficientNet-B0 | aug_224x224 | 0.9035 | 83.83% | 5.39% | 17.23% | 0.44066 |
| ResNet50 | base_64x64 | 0.8703 | 82.04% | 1.80% | 27.21% | 0.43790 |
| EfficientNet-B0 | base_64x64 | 0.7773 | 86.83% | 2.99% | 28.61% | 0.43765 |
| ResNet50 | aug_64x64 | 0.8677 | 82.63% | 3.59% | 19.36% | 0.43664 |
| EfficientNet-B0 | base_224x224 | 0.8897 | 82.63% | 4.19% | 22.02% | 0.43620 |
| EfficientNet-B0 | aug_64x64 | 0.7577 | 82.63% | 2.99% | 25.08% | 0.41968 |

The detailed three-zone breakdown for the two highlighted models on the test set is shown below. `ResNet50 | aug_224x224` ranks first by composite clinical score (best AUC and highest melanoma capture rate), while `ResNet50 | base_64x64` ranks fourth overall but presents the lowest automatic-dismissal risk, hiding only `1.80%` of melanomas in the Non-Melanoma zone.

**ResNet50 | aug_224x224** — AUC 0.9128, T_LOW = 0.1451, T_HIGH = 0.4432

| Zone | Total | % total | True mel. | % of mels | Mel. rate |
|------|-------|---------|-----------|-----------|-----------|
| Non-Melanoma | 781 | 51.96% | 5 | 2.99% | 0.64% |
| Possible Melanoma | 340 | 22.62% | 19 | 11.38% | 5.59% |
| Melanoma | 382 | 25.42% | 143 | 85.63% | 37.43% |
| **TOTAL** | **1503** | **100%** | **167** | **100%** | — |

**ResNet50 | base_64x64** — AUC 0.8703, T_LOW = 0.0798, T_HIGH = 0.3839

| Zone | Total | % total | True mel. | % of mels | Mel. rate |
|------|-------|---------|-----------|-----------|-----------|
| Non-Melanoma | 588 | 39.12% | 3 | 1.80% | 0.51% |
| Possible Melanoma | 409 | 27.21% | 27 | 16.17% | 6.60% |
| Melanoma | 506 | 33.67% | 137 | 82.04% | 27.08% |
| **TOTAL** | **1503** | **100%** | **167** | **100%** | — |

The two configurations represent different risk profiles. `ResNet50 | aug_224x224` offers stronger overall discrimination (AUC +4.25 pp) and captures more melanomas with high confidence (`85.63%` vs `82.04%`), but at the cost of a larger uncertain zone (`22.62%` vs `27.21%`). `ResNet50 | base_64x64` keeps the automatic-dismissal risk lower (`1.80%` vs `2.99%`, a difference of approximately two cases on the test set), which may be preferred in settings where any missed melanoma carries higher operational weight than the volume of cases referred for clinical review.

The broader comparison across all eight experiments also confirmed that `EfficientNet-B0` at `64×64` continues to show near-degenerate threshold values (`T_HIGH < 0.002`), indicating that this architecture at low resolution does not achieve sufficient class separation under the current training setup. The `224×224` configurations consistently produced higher AUC and better threshold calibration across both architectures.

In summary, Sprint 4 established the final validation protocol of the project and identified `ResNet50 | aug_224x224` as the strongest configuration. The next steps involve refining the threshold strategy to reduce the size of the uncertain zone without compromising the safety of the automatic dismissal zone, and finalizing the model for integration into the deployment pipeline.

## References

1. AMERICAN CANCER SOCIETY. *Cancer Facts & Figures 2024*. Atlanta: American Cancer Society, 2024.
2. CARAVIELLO, Camila et al. *Melanoma Skin Cancer: A Comprehensive Review of Current Knowledge*. *Cancers*, Basel, v. 17, n. 2920, p. 1–35, 2025.
3. VIEIRA, Larissa Silva Fontaine; BRANDÃO, Byron José Figueiredo. *Diagnosis and prevention of melanoma: a systematic review*. *BWS Journal*, [s. l.], v. 5, e220900160, p. 1–10, Sept. 2022.
4. Kaggle dataset: [Skin Cancer Lesions Segmentation](https://www.kaggle.com/datasets/volodymyrpivoshenko/skin-cancer-lesions-segmentation/code)
