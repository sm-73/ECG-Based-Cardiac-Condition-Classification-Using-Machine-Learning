# ECG-Based-Cardiac-Condition-Classification-Using-Machine-Learning

A machine learning project that classifies ECG records into four cardiac conditions — **Arrhythmia (ARR)**, **Atrial Fibrillation (AFF)**, **Congestive Heart Failure (CHF)**, and **Normal Sinus Rhythm (NSR)** — using the [ECG of Cardiac Ailments](https://www.kaggle.com/datasets/akki2703/ecg-of-cardiac-ailments-dataset/data) dataset from Kaggle. The dataset contains 1,200 records (300 per class) with 54 numeric ECG-derived features. The goal is to build an interpretable, high-performing classifier while exploring the data visually to understand what separates these conditions.

## Data Preprocessing

- Checked for duplicates and normality (Shapiro-Wilk test confirmed non-normal distributions across all features).
- Identified invalid zero values in several features and treated them as missing.
- Found 9 features with missing values, seven of them ~64% missing. Per-class analysis showed missingness rates differ systematically by class, indicating a **Missing Not at Random (MNAR)** pattern.
- Outliers were detected using the IQR method but retained, since removing extreme physiological values requires clinical judgment.
- Missing values were imputed using an **Iterative Imputer with a Decision Tree Regressor**, fitted separately on train/test splits during modeling to avoid data leakage (and once on a full copy of the data for visualization purposes).

## Data Analysis & Visualization

Exploratory analysis combined pairwise scatter/KDE plots, heart rate distribution and variability plots, correlation heatmaps, ECDFs, and t-SNE for dimensionality reduction. Key findings:

- **AFF and CHF overlap** substantially across features and in t-SNE space, making them the hardest classes to separate.
- **ARR** shows the widest spread and highest heart rate variability, consistent with its broad clinical spectrum.
- **NSR** forms the most distinct, compact cluster, reflecting its stable, healthy rhythm.
- **QRS angle and R-R mean** are strongly negatively correlated (r = -0.73), as expected physiologically.

## Feature Selection

Since baseline misclassifications concentrated between AFF and CHF, feature selection targeted signal that discriminates between them, using two complementary methods:

1. **Mann-Whitney U-test** (with Benjamini-Hochberg FDR correction) comparing AFF vs. CHF feature distributions.
2. **Random Forest impurity-based importance** trained on the full feature set.

Results from both methods were cross-referenced, and redundant/highly correlated features were dropped. The final improved feature set: `hbpermin`, `QRSarea`, `NNTot`, `IBISD`, `Pseg`, `PPmean`, `QTdis`.

## Model Training & Prediction

- **Model**: Random Forest, chosen for its robustness to non-linear relationships, multicollinearity, and non-Gaussian feature distributions.
- **Baseline model**: trained on 4 interpretable features (`hbpermin`, `RRmean`, `QRSang`, `QRSarea`), tuned via grid search (5-fold CV). Achieved **88% accuracy**, F1 = 0.88. Main weakness: confusion between AFF and CHF.
- **Improved model**: trained on the 7 selected features, achieved **97.8% accuracy**, F1 = 0.98, with AFF/CHF misclassifications reduced substantially.
- Both models were evaluated with confusion matrices, ROC/AUC curves, learning curves, and feature importance plots.
- A clinical case prediction example is included, demonstrating how the model handles missing input features at inference time.

## Limitations

The dataset is small, balanced (not reflective of real-world prevalence), and single-source, and high missingness in several features adds imputation uncertainty. This project is a decision-support proof of concept and is not a substitute for professional medical diagnosis.

## Authors

- Sedra Alramadan
- Sedra Albasha
