\# Lab 2: ML Model Training and Versioning with GitHub Actions



\## Overview

This lab demonstrates how to use GitHub Actions to automate the process of training a machine learning model, storing the model, and versioning it.



\## Dataset

\*\*Wine Dataset\*\* - A classic dataset for classification tasks containing 178 samples with 13 features. The goal is to classify wines into 3 categories based on their chemical properties.



\## Model

\*\*Support Vector Machine (SVM)\*\* - A powerful classifier that works well for medium-sized datasets with clear margins of separation.



\## Project Structure

```

lab2/

├── src/

│   ├── \_\_init\_\_.py

│   ├── train\_model.py      # Trains the SVM model

│   ├── evaluate\_model.py   # Evaluates model performance

│   └── calibrate\_model.py  # Calibrates model probabilities

├── test/

│   ├── \_\_init\_\_.py

│   └── test\_model.py       # Unit tests for the model

├── models/                  # Stored trained models

├── metrics/                 # Evaluation metrics

├── requirements.txt

└── README.md

```



\## How to Run Locally



\### 1. Install Dependencies

```bash

pip install -r requirements.txt

```



\### 2. Train the Model

```bash

python src/train\_model.py

```



\### 3. Evaluate the Model

```bash

python src/evaluate\_model.py

```



\### 4. Calibrate the Model

```bash

python src/calibrate\_model.py

```



\### 5. Run Tests

```bash

python -m pytest test/

```



\## GitHub Actions

The workflows automatically run when code is pushed to the main branch:

\- \*\*Model Training and Evaluation\*\* - Trains and evaluates the model

\- \*\*Model Calibration\*\* - Calibrates the trained model



\## Results

\- \*\*Accuracy:\*\* ~97%

\- \*\*F1 Score:\*\* ~0.97

\- \*\*Model:\*\* SVM with RBF kernel



\## Author

Anita

