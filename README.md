# 🚗 Used Car Price Estimator & Deal Classifier

An AI-powered web application that predicts the fair resale price of used cars in the Indian market and classifies deals as **Underpriced**, **Fair**, or **Overpriced**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-red)

## 📋 Features

- **Price Estimation**: Predicts fair market value using ensemble ML models
- **Deal Classification**: Classifies listings as Underpriced / Fair / Overpriced
- **Explainability**: SHAP-based feature importance for every prediction
- **Interactive Dashboard**: EDA charts, model comparison tables, confusion matrix
- **Modern UI**: Dark theme, responsive design, real-time predictions

## 📁 Project Structure

```
car-price-app/
├── data/
│   ├── raw/                    # Place Car details v3.csv here
│   └── processed/              # Cleaned data & metadata (auto-generated)
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
├── src/
│   ├── preprocess.py           # Data cleaning & pipeline
│   ├── eda.py                  # EDA chart generation
│   ├── train_regression.py     # Regression model training
│   ├── train_classification.py # Deal classifier training
│   └── evaluate.py             # Evaluation & SHAP analysis
├── models/                     # Saved model pipelines (auto-generated)
├── app/
│   ├── app.py                  # Flask backend
│   ├── templates/index.html    # Frontend HTML
│   └── static/
│       ├── css/style.css
│       ├── js/main.js
│       └── img/                # EDA & model charts (auto-generated)
├── tests/
│   └── test_app.py             # pytest test suite
├── requirements.txt
├── Procfile                    # For Render deployment
└── README.md
```

## 🚀 Setup & Installation

### 1. Clone & Install Dependencies

```bash
cd car-price-app
pip install -r requirements.txt
```

### 2. Download the Dataset

1. Go to [Kaggle — CarDekho Car details v3](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
2. Download `Car details v3.csv`
3. Place it in `data/raw/Car details v3.csv`

### 3. Train the Models

Run the scripts in order:

```bash
# Step 1: Preprocess data
python src/preprocess.py

# Step 2: Generate EDA charts & notebook
python src/eda.py

# Step 3: Train regression model
python src/train_regression.py

# Step 4: Train classification model
python src/train_classification.py

# Step 5: Evaluate & generate SHAP analysis
python src/evaluate.py
```

### 4. Run the Web App

```bash
python app/app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### 5. Run Tests

```bash
pytest tests/ -v
```

## 📊 Results

*(Metrics are auto-populated from training — these are example values)*

### Regression

| Model | MAE (₹) | RMSE (₹) | R² |
|-------|---------|----------|-----|
| Linear Regression | — | — | — |
| Ridge | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| **Best (tuned)** | — | — | — |

### Classification

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| SVM | — | — | — | — | — |
| XGBoost | — | — | — | — | — |
| **Best (tuned)** | — | — | — | — | — |

## 🖼️ Screenshots

*(Screenshots will be captured after running the app)*

## 🌐 Deployment on Render

1. Push the repo to GitHub.
2. Create a new **Web Service** on [Render](https://render.com).
3. Connect your GitHub repo.
4. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app/app.py`
   - **Environment**: Python 3.11
5. Ensure model files (`models/` directory) are included in your repo or use a build script to train them.

## 📄 License

MIT License — feel free to use and modify.
