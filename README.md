
# Wine Classification Using K-Nearest Neighbors (KNN)

## Project Overview

This project demonstrates a complete supervised machine learning workflow using Python and Scikit-learn.

The objective is to classify wine samples into three categories based on their chemical characteristics.

The project covers data exploration, preprocessing, model training, hyperparameter optimization, evaluation, and model persistence.

## Dataset

The Wine dataset is available through Scikit-learn and originates from the UCI Machine Learning Repository.

- **Samples:** 178
- **Features:** 13 numerical chemical characteristics
- **Target:** 3 wine classes
- **Task:** Multiclass classification

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Methodology

1. Load and explore the Wine dataset.
2. Split the dataset into training (80%) and testing (20%) sets.
3. Apply feature standardization using StandardScaler.
4. Build a KNN classifier.
5. Compare different K values using 5-fold stratified cross-validation.
6. Select the best K based on mean validation accuracy.
7. Train the final model using a Scikit-learn Pipeline.
8. Evaluate the model on the held-out test set.
9. Save and reload the trained model using Joblib.

## Results

| Metric | Result |
|---|---|
| Selected K | 7 |
| Training samples | 142 |
| Test samples | 36 |
| Correct test predictions | 36 |
| Test accuracy | 100% |

### Confusion Matrix

```text
[[12  0  0]
 [ 0 14  0]
 [ 0  0 10]]
```

All 36 test samples were correctly classified.

**Important:** A 100% test accuracy on 36 samples does not guarantee perfect performance on future data. Additional evaluation on independent datasets would strengthen the findings.

## How to Run

1. Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib
```

2. Download or clone this repository.
3. Run the Python script:

```bash
python knn_wine.py
```

## Key Skills Demonstrated

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature scaling
- Supervised machine learning
- K-Nearest Neighbors classification
- Cross-validation
- Hyperparameter selection
- Classification metrics
- Model persistence

## Author

Mohamed Kontao

MSc in Biostatistics

Interested in Data Science, Machine Learning, and applied statistical analysis.
