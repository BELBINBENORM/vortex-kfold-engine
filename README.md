# 🌪️ Vortex KFold Engine

**Vortex KFold Engine** is a high-performance cross-validation wrapper designed for robust model training and seamless experiment tracking. [cite_start]It automates the K-Fold process while integrating **Cloudpickle-based persistence** to avoid redundant training sessions[cite: 21, 30].
---
## 🚀 Features

* **Dual Task Support**: Optimized to handle both **Classification** (using StratifiedKFold) and **Regression** (using KFold) tasks automatically.
* [cite_start]**Automated Persistence**: Automatically detects and loads pre-trained model ensembles and Out-of-Fold (OOF) predictions if they exist in the specified path[cite: 25, 26].
* [cite_start]**Parallel Execution**: Utilizes `joblib` to execute $N$-fold cross-validation in parallel, significantly reducing training time[cite: 21, 28].
* [cite_start]**Cloudpickle Integration**: Uses `cloudpickle` instead of standard `pickle` to ensure custom classes and complex lambda functions are serialized correctly[cite: 21, 30].
* [cite_start]**Scikit-Learn Compatibility**: Inherits from `BaseEstimator` to function as a drop-in replacement in existing pipelines[cite: 21].
---
## 📓 Notebook Installation

For **Jupyter Notebook**, **JupyterLab**, or **Google Colab**, you can install the engine directly from GitHub:

```python
!pip install git+https://github.com/BELBINBENORM/vortex-kfold-engine.git
```
---
## 🚀 Quick Start in Notebook

Once installed, you can import and run the engine anywhere in your notebook:

```python
from vortex_kfold import VortexKFold
from sklearn.ensemble import RandomForestClassifier

# Initialize the engine
# task: 'classification' or 'regression'
vortex = VortexKFold(
    base_estimator=RandomForestClassifier(),
    task='classification',
    model_name="my_vortex_model",
    n_splits=5
)

# Train the model (or load if existing files are found)
vortex.fit(X_train, y_train)

# Make predictions
probs = vortex.predict_proba(X_test)
preds = vortex.predict(X_test)
```
---
## 📊 Summary Output

The engine provides detailed logging during the training process:

* [cite_start]**🔍 Search**: Checks for existing `.cloudpickle` and `.npy` files to skip training[cite: 5, 6].
* [cite_start]**📦 System**: Reports the number of folds and parallel jobs being utilized[cite: 8].
* [cite_start]**✅ Progress**: Confirms when individual folds and OOF generations are completed[cite: 9].
* [cite_start]**⭐ Performance**: Displays the final Cross-Validation (CV) score (ROC-AUC for classification or R2 for regression)[cite: 11].

---
*Developed by **BELBIN BENO R M***
