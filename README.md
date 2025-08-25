# Predictive Analysis & Early Detection of Chronic Kidney Disease (CKD)

A full machine learning pipeline to predict and stratify CKD risk using clinical, demographic, and lifestyle features. The project combines supervised learning (Random Forest, Logistic Regression), unsupervised learning (K-Means + PCA), and a percentile-based **Risk Scoring System** to support early detection and clinical decision-making.

---

## Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Selection](#feature-selection)
  - [Random Forest Classifier](#random-forest-classifier)
  - [Logistic Regression](#logistic-regression)
  - [PCA + K-Means Clustering](#pca--k-means-clustering)
  - [Risk Scoring System](#risk-scoring-system)
- [Key Results](#key-results)
- [Data](#data)
- [Environment & Installation](#environment--installation)
- [Limitations & Future Work](#limitations--future-work)
- [Contributors](#contributors)
- [Reference](#reference)

---

## Overview
This project builds a robust CKD analytics stack:
- **Supervised ML** for diagnosis prediction.
- **Unsupervised ML** for risk segmentation via clustering.
- **Risk Scoring** (percentile-based) for clinician-friendly triage.
- **Visualization** for interpretability (correlations, ROC, confusion matrix, PCA scatter, elbow/silhouette, etc.).

---

## Objectives
- Apply supervised and unsupervised ML models to predict CKD risk.  
- Evaluate key risk factors such as diabetes, hypertension, and family history.  
- Develop a **personalized risk scoring system** integrating clinical and lifestyle features.  
- Provide interpretable outputs for clinical decision-making.  

---

## Methodology

### Data Preprocessing
- Removed confidential/unhelpful variables (e.g., *DoctorInCharge*).  
- Conducted summary statistics, histograms, heatmaps, and correlation matrices.  
- Applied **Synthetic Minority Oversampling Technique (SMOTE)** to balance classes.  
- Created **CKD staging categories** using GFR values.  
- Engineered interactions (e.g., Age × Creatinine).  
- Normalized features with Z-score scaling.  

<img width="450" height="340" alt="image" src="https://github.com/user-attachments/assets/67e99e64-0ce7-4581-9ec4-7ec468b11ee0" />
<img width="519" height="600" alt="image" src="https://github.com/user-attachments/assets/4a8bff6e-01a6-4e6e-8fb5-a58005cae0c2" />

---

### Feature Selection
- Correlation ranking, SelectKBest, and RFE tested.  
- Final pipeline retained ~15 most important features (GFR, Serum Creatinine, BUN, glucose, BP, BMI, and symptom indicators).  

<img width="896" height="592" alt="image" src="https://github.com/user-attachments/assets/82a177fb-325a-4d1b-a02e-ba8089425e0b" />

---

### Random Forest Classifier
- Dataset split 80/20 (stratified).  
- Hyperparameter tuning with GridSearchCV (5-fold CV).  
- **Performance**:  
  - Accuracy: **94.59%**  
  - ROC-AUC: **0.9875**  
  - Balanced precision and recall; low false positives/negatives.  


<img width="694" height="543" alt="image" src="https://github.com/user-attachments/assets/f5db29ca-bb9b-4653-b601-e4e75cf6f4de" />

<img width="456" height="547" alt="image" src="https://github.com/user-attachments/assets/98a5ecb8-68f0-44ff-b179-e1ba47df978f" />

---

### Logistic Regression
- Used for **interpretability** (probabilities and coefficients).  
- Standardized features before training.  
- **Performance**:  
  - Accuracy: **74.1%**  
  - ROC-AUC: **0.81**  
  - High precision (97.4%), lower recall (73.7%).  


<img width="551" height="282" alt="image" src="https://github.com/user-attachments/assets/ea088fd7-9c1d-46da-b08d-1862e21e20a9" />

---

### PCA + K-Means Clustering
- PCA for dimensionality reduction before clustering.  
- **K=3** used 11 components (explained **91.3%** variance) → clear low, moderate, high-risk groups.  
- **K=5** used 5 components (explained **55.1%** variance) → more granularity, more overlap.  
- Diabetes and hypertension emerged as strongest cluster drivers; family history secondary.  


<img width="378" height="230" alt="image" src="https://github.com/user-attachments/assets/7fb45683-1647-4ab7-87ab-da7346de99c2" />

<img width="407" height="242" alt="image" src="https://github.com/user-attachments/assets/c8c76ae6-0ede-419e-821a-57036266641e" />

<img width="377" height="468" alt="image" src="https://github.com/user-attachments/assets/6fe8a285-2fc3-49a3-9d1a-785c7b120d31" />


---

### Risk Scoring System
- Weighted sum of selected features → logistic function → probability.  
- Percentile thresholds:  
  - **Low risk**: < 33rd percentile  
  - **Moderate risk**: 33rd–67th percentile  
  - **High risk**: > 67th percentile  
- Designed for real-time triage in clinical settings.  


<img width="463" height="228" alt="image" src="https://github.com/user-attachments/assets/825c8dfc-9f7e-47a3-9d6b-af081467c2f7" />

---

## Key Results
- **Random Forest**: 94.59% accuracy, ROC-AUC 0.9875.  
- **Logistic Regression**: 74.1% accuracy, ROC-AUC 0.81.  
- **Clustering**: Clear segmentation of low, moderate, and high-risk groups (K=3).  
- **Risk Score**: Clinically interpretable scoring beyond binary classification.

---

## Data
- **Source**: Kaggle — Chronic Kidney Disease Dataset  
  https://www.kaggle.com/datasets/rabieelkharoua/chronic-kidney-disease-dataset-analysis  
- **Records**: 3,048 (balanced after SMOTE)  
- **Feature groups**: demographics, lifestyle, family/medical history, vitals & labs, medications, symptoms.

---

## Environment & Installation
**Python**: 3.9+

Install requirements:
    
    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn scipy

*(Jupyter recommended to run the notebook end-to-end.)*

## Limitations & Future Work
- Validate on **external** and **longitudinal** cohorts.  
- Refine clustering thresholds and evaluate clinical utility curves.  
- Integrate clustering outputs as features in supervised models.  
- Package risk scoring as a **lightweight API** for EHR/clinical decision support.

---

## Contributors
- **Sena Bui**  
- **Johnathen Kent**  
- **Shria Narapaneni**  
- **Charlotte Anderson**

---

## Reference
El Kharoua, R. (2024). *Chronic Kidney Disease Dataset* [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/8658224
