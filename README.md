
# 🏥 In-Hospital Mortality and Readmission Prediction Using Multimodal NLP and ClinicalBERT LLM

An end-to-end machine learning pipeline is developed to predict in-hospital mortality and readmission using structured data and unstructured clinical notes from the MIMIC-III dataset. The system leverages ClinicalBERT embeddings, optimized XGBoost/CatBoost models, and SHAP interpretability to enable explainable healthcare predictions.

---

## 📌 Contents Overview

- [Problem Statement](#problem-statement)
- [Project Overview](#project-overview)
- [Dataset](#data-sources)
- [Methodology](#methodology)
- [Model Evaluation](#model-performance)
- [Model Interpretability (SHAP)](#shap-interpretability)
- [Streamlit Integration](#streamlit-integration)
- [Project Structure](#folder-structure)
- [Setup and Usage](#installation--usage)
- [Acknowledgements](#acknowledgements)

---

## ❗ Problem Statement

Hospital readmissions and in-hospital deaths are significant clinical outcomes that affect patient health and hospital performance metrics. Early prediction of these events can help improve the efficiency of care resource allocation.

---

## 📊 Project Overview

| Task | Description |
|------|-------------|
| **Mortality Prediction** | Binary classification using `HOSPITAL_EXPIRE_FLAG` |
| **Readmission Prediction** | Binary classification of `future_admission` within 30 days |
| **Data Types** | Structured (labs, vitals, demographics) + Clinical notes |
| **Embedding Model** | ClinicalBERT |
| **Models** | XGBoost, CatBoost, Ensemble Voting Classifier, Decision Tree, Linear Regression |
| **Evaluation** | AUC-RUC, Precision- Recall Curves, MCC, Sensitivity, Specificity, Confusion Matrix, SHAP |

---

## 🧾 Data Sources

- **Structured Data:** Vitals, diagnoses, procedures, insurance, demographics, Patients, Admission, ICU Stays (from MIMIC-III)
- **Unstructured Data:** Clinical notes from the NOTEEVENTS table
- **Embedding Model:** `clinicalBERT` via HuggingFace `sentence-transformers`

---

## 🧠 Methodology

1. **Data Cleaning & Feature Engineering**
   - Admission processing, comorbidities (sepsis, diabetes, ventilation)
   - Label generation for readmission detection
2. **Embedding Integration**
   - Generated ClinicalBERT embeddings per `HADM_ID`
   - Merged with structured features
3. **Model Training**
   - Mortality & readmission models trained using:
     - `XGBoost` (GPU-accelerated)
     - `CatBoost` (categorical-friendly)
     - `VotingClassifier` ensemble
     - `Decision Tree`
     - `Linear Regression`
   - Hyperparameter tuning via `RandomizedSearchCV`
4. **Evaluation**
   - ROC AUC, MCC, Sensitivity, Specificity, Accuracy
   - Visualizations: ROC, Precision-Recall
   - Explainability: `SHAP` value plots for feature importances

---

## 🥇 Best Model Performance(XG Boost)

| Metric | Mortality (Before) | Mortality (After Embeddings) | Readmission (Before) | Readmission (After Embeddings) |
|--------|--------------------|-------------------------------|----------------------|-------------------------------|
| **AUC** | 0.872 | **0.901** | 0.714 | **0.738** |
| **MCC** | 0.409 | **0.505** | 0.121 | **0.203** |
| **Sensitivity** | 0.260 | **0.382** | 0.054 | **0.151** |
| **Specificity** | 0.990 | 0.986 | 0.988 | 0.964 |
| **Accuracy** | 92% | 92% | 77% | 76% |

✅ **Clinical note embeddings significantly boosted predictive performance**, especially for **sensitivity**, which is vital in healthcare risk prediction.

---

## 🌟 SHAP Interpretability

- SHAP values highlighted ventilator use, sepsis, and admission source as dominant mortality predictors.
- Readmission prediction relied heavily on LOS, procedure count, and discharge-summary embeddings.
- Bar plots and interaction visualizations were used to improve clinical explainability.

---

## 🌐 Streamlit Integration

✔️ Final models are saved as `.pkl` and exported for integration with a **Streamlit** frontend for real-time prediction demo (included in roadmap).

---

## 🗂️ Folder Structure

```
📁 /project-root
├── Embeddings-Noteevents.ipynb      # ClinicalBERT embeddings + integration
├── ML-Pipeline-Structured.ipynb     # Full model training & evaluation
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation (you are here)
└── 📁 models/
    ├── mortality_model.pkl
    └── readmission_model.pkl
```

---

## ⚙️ Installation & Usage

```bash
# Clone the repo
git clone https://github.com/yourusername/mimic-prediction-project.git
cd mimic-prediction-project

# Install dependencies
pip install -r requirements.txt

# Run notebooks
jupyter notebook ML-Pipeline-Structured.ipynb
```

---

## Acknowledgements

- [PhysioNet](https://physionet.org/) for the MIMIC-III dataset
- [HuggingFace](https://huggingface.co/) for ClinicalBERT embeddings
- Professor Muhammad Ali Yousuf for project guidance
