
# 1. Gestational Diabetes Prediction


This project focuses on predicting gestational diabetes (GD) using a combination of classical machine learning models, advanced preprocessing techniques, and contrastive learning. The pipeline is designed to enhance prediction accuracy, robustness, and generalization using health indicators and tabular data.

The methodology includes feature selection (wrapper methods and PCA), data augmentation (SMOTE and ADASYN), and a contrastive learning framework for discriminative representation learning.

# 2. Project Description

Gestational diabetes is a glucose intolerance diagnosed during pregnancy that poses serious health risks to both mother and fetus. Early and accurate prediction is crucial for effective intervention. This project develops an end-to-end ML pipeline using structured health indicator data to classify individuals at risk of gestational diabetes.

The pipeline supports:
- Feature engineering and reduction
- Data balancing with synthetic sampling
- Embedding learning through contrastive learning
- Model training using classic ML techniques (SVM, Random Forest, etc.)
- Model training using DL techniques (Tabnet, Renet, etc.)

# 3. Tools & Libraries Used

- Python
- scikit-learn – For machine learning and feature selection (RFE, PCA)
- imbalanced-learn – For data augmentation with SMOTE and ADASYN
- PyTorch – For contrastive learning using a Siamese network
- pandas / NumPy / matplotlib – For data handling and visualization

# 4. Project Workflow

Run scripts under the `GDP-ML` folder in the  Following Order:
------------------------------------
1. `./prepare_dataset.ipynb` – Load and clean the raw dataset.
2. `./data_augmentation.ipynb` – Apply SMOTE and ADASYN to handle class imbalance.
3. `./fs_manual_pca_with_da.ipynb` – Apply dimensionality reduction with PCA, and manual feature selection on dataset after DA.
4. `./fs_manual_pca_without_da.ipynb` – Apply dimensionality reduction with PCA, and manual feature selection on dataset without DA.
5. `./wrapper_method.ipynb` – Select features using Recursive Feature Elimination (RFE) and hybrid FS (PCA + RFE)
6. `./contrastive_learning.ipynb` – Train CL network with positive pair selection and Cosine loss to learn embeddings.
7. `./train_ML.ipynb` – Train ML classifiers (RF, SVM) before CL and evaluate their results.
8. `./train_ML_without_DA.ipynb` – Train ML classifiers (RF, SVM) without DA before CL and evaluate their results.
9. `./train_ML_CL.ipynb` – Train ML classifiers (RF, SVM) after CL and evaluate their results.
10. `./resnet.py` – Train a Resnet network and evaluate its results.
11. `./tabnet.ipynb` – Train Google's Tabnet network and evaluate its results.
12. `./boosting_models.ipynb` – Train a light GBM and XGB models and evaluate their results.




Key Features
------------
- Robust feature selection combining wrapper methods and PCA
- Data augmentation with SMOTE and ADASYN to address class imbalance
- Contrastive learning for improved feature representations
- Flexible and interpretable ML workflow for structured medical data

Output
------
- Cleaned and augmented dataset
- Reduced feature set
- Trained embeddings from contrastive learning
- Final classification model performance (accuracy, F1-score, etc.) 

The cleaned datasets are under could be generated via the scripts, but are also available in this [link](https://liveumoncton-my.sharepoint.com/:f:/g/personal/psb7953_umoncton_ca/EpeoDCTc_QVOuhipsOSVZswBUg_ZJ_VbeXEMwiyljX0_ag?e=amm3tq). \
The output metrics are under `results`



