# 🖋️ Automated Forensic Signature Verification System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-15B550?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)

A hierarchical, multi-model forensic tool designed to detect **"Skilled Forgeries"** in offline handwritten signatures. This system moves beyond traditional single-model approaches by implementing a novel **Three-Tiered "Forensic Triage" Architecture** that mimics real-world security clearance protocols.

---

## 🚀 Key Innovation: Three-Tiered "Forensic Triage"

Most signature verification systems rely on a single classifier. This project introduces a **Sequential Validation Logic** where a signature is authenticated as "Genuine" **only if it passes three strict security layers unanimously**:

1.  **🛡️ Tier 1: Generic Model (The Bouncer)**
    * **Function:** Screens out random, obvious forgeries (e.g., wrong name, shaky lines).
    * **Training:** Trained on the public **CEDAR** dataset.
2.  **🏢 Tier 2: Group Model (The Department Check)**
    * **Function:** Verifies if the signature style matches the authorized user group or department.
    * **Training:** Trained on a custom multi-user dataset.
3.  **🔐 Tier 3: Individual Model (The Biometric Lock)**
    * **Function:** A high-security check against a specific user's unique muscle memory.
    * **Training:** Trained *exclusively* on **one individual's** genuine vs. forged signatures.

---

## 🛠️ Technical Approach

* **Hybrid Features:** Fuses **HOG** (Macro-geometry/Shape) and **LBP** (Micro-texture/Pressure) to capture a complete biometric profile.
* **Dimensionality Reduction:** Uses **PCA (Principal Component Analysis)** to retain 95% of the variance while reducing computational load.
* **Classification:** Utilizes **XGBoost**, chosen for its superior performance on imbalanced forensic datasets.

---

## 📊 Results

* **Metric:** Area Under Curve (AUC)
* **Score:** **0.82** (Individual Model)
* **Impact:** The sequential validation logic significantly reduced the False Acceptance Rate (FAR) for skilled forgeries compared to baseline models.

![ROC Curve](results/individual_plot_7_roc_curve.png)
*Fig 1. ROC Curve showing the performance of the User-Specific Model*

---

## 📂 File Description

* **`app.py`**: The entry point for the Streamlit web interface. Handles image upload and visualization.
* **`main.py`**: Contains the core backend logic for image preprocessing (Otsu thresholding) and feature extraction (HOG + LBP).
* **`train_individual.py`**: The script used to train the specialized Tier 3 model.
* **`models/`**: Directory containing the pre-trained `.joblib` model files and PCA transformers.

---

## 📄 References

1.  **Kalera et al. (2004)**: Established CEDAR benchmarks.
2.  **Otsu (1979)**: Thresholding method used for preprocessing.
3.  **Ojala et al. (2002)**: LBP for texture analysis.
4.  **Dalal & Triggs (2005)**: HOG for shape analysis.
5.  **Alhadidi & Hiary (2024)**: Validated the efficiency of lightweight models over heavy Deep Learning architectures.

---

## 👨‍💻 Author

**Rohan A S Gowda**
Department of Electronics and Communication Engineering
