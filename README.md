# 🎗️ Breast Cancer Prediction using Random Forest

> **A machine learning classification project that uses Random Forest, exploratory data analysis, outlier handling, hyperparameter tuning, cross-validation, and model evaluation to classify breast tumors as malignant or benign.**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Project Overview

Breast cancer classification is a binary machine learning problem where tumor measurements can be used to classify a tumor as **malignant** or **benign**.

This project develops a **Random Forest Classifier** using the **Scikit-learn Breast Cancer Wisconsin dataset**.

The project follows a complete machine learning workflow:

```text
Data
 ↓
Data Understanding
 ↓
Data Quality Checks
 ↓
Outlier Analysis
 ↓
Exploratory Data Analysis
 ↓
Feature Analysis
 ↓
Train/Test Split
 ↓
Hyperparameter Tuning
 ↓
Cross-Validation
 ↓
Random Forest Training
 ↓
Model Evaluation
 ↓
Feature Importance
 ↓
Model Serialization
```

The objective is not only to train a classifier, but also to understand the data, evaluate model behavior, identify influential features, and document limitations relevant to a healthcare-oriented classification problem.

---

# 🎯 Project Objectives

The project aims to:

* Explore the Breast Cancer Wisconsin dataset.
* Understand the distribution of malignant and benign cases.
* Perform data-quality checks.
* Detect potential feature outliers.
* Apply IQR-based outlier capping.
* Perform exploratory data analysis.
* Investigate feature correlations.
* Split the dataset using stratified sampling.
* Tune Random Forest hyperparameters using `GridSearchCV`.
* Evaluate the final model using multiple metrics.
* Analyze feature importance.
* Perform 5-fold cross-validation.
* Visualize the confusion matrix and ROC curve.
* Save the trained model for future inference.

---

# 📂 Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic dataset available through Scikit-learn**.

The dataset contains:

| Property         |   Value |
| ---------------- | ------: |
| Total samples    | **569** |
| Features         |  **30** |
| Target classes   |   **2** |
| Training samples | **457** |
| Test samples     | **112** |

The target variable represents:

```text
0 → Malignant
1 → Benign
```

The 30 input features describe characteristics of cell nuclei computed from digitized images of breast mass samples.

Examples include:

* Mean radius
* Mean texture
* Mean perimeter
* Mean area
* Mean smoothness
* Mean compactness
* Mean concavity
* Mean concave points
* Worst radius
* Worst perimeter
* Worst area
* Worst concave points

The notebook loads the dataset directly through:

```python
from sklearn.datasets import load_breast_cancer
```

and converts it into a Pandas DataFrame for analysis.

---

# 🔍 Understanding the Target

The classification task is:

```text
Tumor Measurements
        │
        ▼
Random Forest Model
        │
        ├───────────────┐
        ▼               ▼
   Malignant          Benign
      (0)               (1)
```

This is a **supervised binary classification** problem because the historical observations already contain known target labels.

---

# 🧹 Data Quality & Preprocessing

Before training the model, the dataset is checked for potential data-quality issues.

### Duplicate Detection

Duplicate rows are checked using:

```python
df.duplicated().sum()
```

### Missing Values

Missing values are checked using:

```python
df.isnull().sum()
```

The notebook also inspects data types and descriptive statistics before modeling.

---

# 📦 Outlier Detection

The project performs visual outlier analysis using box plots across the feature set.

Several features contain observations outside their typical interquartile range.

The project therefore applies **IQR-based capping**.

The IQR is calculated as:

```text
IQR = Q3 − Q1
```

with boundaries:

```text
Lower Bound = Q1 − 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

Values outside these boundaries are capped to the corresponding boundary.

The notebook applies this process across the feature columns before model development.

---

# 📊 Exploratory Data Analysis

The project performs EDA to understand:

### Class Distribution

The dataset contains approximately:

* **37% malignant cases**
* **63% benign cases**

This represents a moderate class imbalance.

### Feature Distributions

The distributions of important features are compared between malignant and benign observations.

The project specifically visualizes:

* `mean_radius`
* `mean_texture`
* `mean_perimeter`
* `mean_area`
* `mean_concavity`
* `worst_radius`

These distributions help investigate how tumor characteristics differ across the two target classes.

### Correlation Analysis

A correlation heatmap is used to identify relationships between numerical features.

The notebook also identifies highly correlated feature pairs using:

```text
|correlation| > 0.90
```

This is useful for understanding feature redundancy and potential future feature-selection opportunities.

---

# ✂️ Train-Test Split

The cleaned dataset is divided into training and testing subsets using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This produces:

```text
Training Set → 457 samples
Testing Set  → 112 samples
```

Stratification is used to preserve the relative class distribution across the training and testing datasets.

---

# 🌲 Why Random Forest?

Random Forest is an ensemble learning algorithm that combines multiple decision trees to generate a more robust classification model.

Instead of depending on one decision tree, Random Forest builds multiple trees using randomized subsets of observations and features.

Conceptually:

```text
                 Training Data
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Tree 1       Tree 2       Tree N
          │            │            │
          └────────────┼────────────┘
                       ▼
                Majority Voting
                       │
                       ▼
                  Final Class
```

This makes Random Forest useful for datasets containing multiple numerical features and potentially complex relationships between variables.

---

# ⚙️ Hyperparameter Tuning

Instead of relying only on default Random Forest parameters, the project uses **GridSearchCV** to search across multiple parameter combinations.

The search space includes:

| Parameter           | Values Tested |
| ------------------- | ------------- |
| `n_estimators`      | 50, 100, 200  |
| `max_depth`         | None, 5, 10   |
| `min_samples_split` | 2, 5          |
| `min_samples_leaf`  | 1, 2          |

The model is evaluated using **5-fold cross-validation** during the grid search.

This approach helps identify a better-performing configuration rather than selecting model parameters arbitrarily.

---

# 🔁 Cross-Validation

After selecting the best estimator, the project performs additional **5-fold cross-validation** on the training data.

The notebook reports:

* Individual fold scores
* Mean cross-validation accuracy
* Standard deviation

This provides additional information about model consistency across different training subsets.

---

# 📈 Model Evaluation

The final Random Forest model is evaluated using:

### Accuracy

Measures the overall proportion of correctly classified observations.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across classification thresholds.

### Classification Report

Provides:

* Precision
* Recall
* F1-score
* Support

### Confusion Matrix

Provides a detailed view of:

* True Negatives
* False Positives
* False Negatives
* True Positives

### ROC Curve

Visualizes the trade-off between:

* True Positive Rate
* False Positive Rate

The notebook implements all of these evaluation components.

---

# 📊 Model Performance

The notebook's recorded summary reports:

| Metric               |                  Result |
| -------------------- | ----------------------: |
| **Algorithm**        |           Random Forest |
| **Dataset**          | Breast Cancer Wisconsin |
| **Samples**          |                     569 |
| **Training Samples** |                     457 |
| **Testing Samples**  |                     112 |
| **Features**         |                      30 |
| **Test Accuracy**    |                 ~96–97% |
| **ROC-AUC**          |                   ~0.99 |

These figures are the project's reported results and should be interpreted as performance on this dataset and test split, rather than as evidence of clinical performance.

---

# 🔎 Feature Importance

Random Forest provides feature-importance estimates that can be used to understand which input variables contributed most to the model's predictions.

The project identifies the following among the most influential features:

1. `worst_radius`
2. `worst_perimeter`
3. `worst_area`
4. `mean_concave_points`
5. `worst_concave_points`

The notebook generates a complete feature-importance visualization and lists the top 10 features.

> **Important:** Feature importance indicates model association/contribution. It should not be interpreted as proof that a feature independently causes cancer or as a clinical diagnostic conclusion.

---

# 💾 Model Serialization

The trained Random Forest model is saved using Joblib:

```python
joblib.dump(model, 'breast_cancer_rf_model.pkl')
```

The saved model can later be loaded using:

```python
loaded_model = joblib.load('breast_cancer_rf_model.pkl')
```

and used for inference on appropriately prepared input data.

---

# 📁 Repository Structure

```text
Breast-Cancer-Prediction-By-Random-Forest/
│
├── Breast_Cancer_Prediction_RF_Improved.ipynb
├── breast_cancer_rf_model.pkl
└── README.md
```

The current GitHub repository contains the improved Random Forest notebook; the model file is generated by the notebook when the save-model cell is executed.

---

# 🛠️ Technology Stack

| Technology           | Purpose                    |
| -------------------- | -------------------------- |
| **Python**           | Programming language       |
| **Pandas**           | Data manipulation          |
| **NumPy**            | Numerical computation      |
| **Matplotlib**       | Visualization              |
| **Seaborn**          | Statistical visualization  |
| **Scikit-learn**     | Machine learning           |
| **Random Forest**    | Classification             |
| **GridSearchCV**     | Hyperparameter tuning      |
| **Cross-Validation** | Model stability evaluation |
| **Joblib**           | Model serialization        |
| **Jupyter Notebook** | Development environment    |

---

# 🚀 How to Run

## 1. Clone the repository

```bash
git clone https://github.com/aryan2026-mishra/Breast-Cancer-Prediction-By-Random-Forest.git
```

## 2. Navigate into the project

```bash
cd Breast-Cancer-Prediction-By-Random-Forest
```

## 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

## 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

## 5. Open

```text
Breast_Cancer_Prediction_RF_Improved.ipynb
```

Run the notebook cells sequentially.

The dataset is loaded directly from Scikit-learn, so a separate CSV download is not required for the current notebook implementation.

---

# ⚠️ Limitations & Technical Considerations

This project is intended as a **machine-learning learning and portfolio project**, not a clinical diagnostic system.

### 1. Potential Data Leakage

The notebook applies IQR outlier capping **before** the train-test split.

This means information from the complete dataset is used when calculating the outlier boundaries.

For a production-quality pipeline, the correct approach would be:

```text
Training Data
     │
     ▼
Calculate IQR Bounds
     │
     ▼
Transform Training Data
     │
     ▼
Apply Same Bounds
     │
     ▼
Transform Test Data
```

The notebook itself explicitly identifies this as a limitation.

### 2. Feature Selection

Several features show very high correlations.

Future versions could investigate removing redundant features or applying dimensionality-reduction techniques.

### 3. Model Comparison

Additional models could be evaluated against Random Forest, such as:

* Logistic Regression
* Support Vector Machine
* XGBoost
* Gradient Boosting
* K-Nearest Neighbors

### 4. Interpretability

Healthcare-oriented machine-learning applications benefit from interpretable predictions.

Future work could incorporate:

* SHAP
* Permutation Importance
* Partial Dependence
* Model calibration

### 5. Dataset Scope

The model is trained and evaluated on a single public dataset. External validation on independent data would be necessary before drawing conclusions about performance in other populations or clinical environments.

---

# 🔮 Future Improvements

The project can be extended into a more complete machine-learning application.

### 🔬 Modeling

* Compare multiple classification algorithms.
* Optimize hyperparameters using stratified cross-validation.
* Tune decision thresholds.
* Evaluate precision, recall, F1, ROC-AUC and PR-AUC.
* Perform probability calibration.

### 🧪 Data Pipeline

* Move preprocessing inside a Scikit-learn `Pipeline`.
* Fit outlier-treatment parameters only on training data.
* Add automated data validation.
* Add reproducible experiment tracking.

### 🧠 Explainability

* Add SHAP-based explanations.
* Analyze global and local feature importance.
* Generate individual prediction explanations.

### 🌐 Deployment

The trained model could be deployed through:

* Streamlit
* FastAPI
* Flask
* REST API

A future version could provide a simple interface where users enter model features and receive a classification output.

---

# 🧠 End-to-End Machine Learning Architecture

```text
┌───────────────────────┐
│ Breast Cancer Dataset │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Data Quality Checks   │
│ Nulls / Duplicates    │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Outlier Analysis      │
│ IQR Capping           │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Exploratory Analysis  │
│ Distribution / Corr.  │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Stratified Split      │
│ 80% Train / 20% Test  │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ GridSearchCV           │
│ Hyperparameter Tuning │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ Random Forest         │
│ Classification       │
└───────────┬───────────┘
            │
       ┌────┴────┐
       ▼         ▼
 Evaluation   Feature
 Metrics      Importance
       │
       ▼
┌───────────────────────┐
│ Saved ML Model        │
│ .pkl                  │
└───────────────────────┘
```

---

# 🎓 Skills Demonstrated

This project demonstrates practical experience with:

* Machine Learning
* Supervised Learning
* Binary Classification
* Random Forest
* Exploratory Data Analysis
* Data Cleaning
* Outlier Detection
* IQR-based Outlier Treatment
* Feature Correlation Analysis
* Feature Importance
* Hyperparameter Tuning
* GridSearchCV
* Cross-Validation
* Confusion Matrix
* ROC-AUC
* Classification Metrics
* Model Serialization
* Python
* Scikit-learn

---

# 👨‍💻 Author

## Aryan Mishra

**B.Tech — Computer Science & Engineering**

Aspiring Data Analyst | Machine Learning | Python | SQL | Power BI

🔗 **LinkedIn:**
https://www.linkedin.com/in/aryan-mishra-61561b298/

💻 **GitHub:**
https://github.com/aryan2026-mishra

---

## ⭐ Project Takeaway

This project demonstrates a complete supervised machine-learning workflow—from **data understanding and EDA to preprocessing, hyperparameter tuning, cross-validation, model evaluation, feature importance, and model serialization**.

The project also documents an important real-world lesson: **a high evaluation score does not automatically mean a model is production-ready**. Data leakage, external validation, interpretability, calibration, and clinical context must be considered before deploying a healthcare-oriented predictive model.
