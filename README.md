# Data-Driven Prediction of Energy Expenditure

An end-to-end Machine Learning pipeline and interactive desktop GUI application designed to forecast individual energy expenditure (calories burned) based on physiological indicators and physical activity metrics.

## 📊 System Architecture & Performance
* **Dataset Size:** 15,000 samples (Physiological & Exercise profiles)
* **Target Feature:** Calories Burnt

### Model Evaluation Summary:
* **XGBoost Regressor:** R² = [Insert Your XGBoost R2 Score, e.g., 0.9995] | MAE = [Insert MAE, e.g., 0.28] kcal
* **Random Forest:** R² = [Insert RF R2 Score] | MAE = [Insert MAE] kcal
* **Linear Regression:** R² = [Insert LR R2 Score] | MAE = [Insert MAE] kcal

## 🛠️ Tech Stack & Libraries Used
* **Language:** Python
* **Data Engineering:** Pandas, NumPy
* **Machine Learning & Pipeline Architecture:** Scikit-learn, XGBoost
* **GUI Desktop Dashboard:** CustomTkinter, Pillow (PIL)
* **Data Visualization:** Matplotlib, Seaborn

## 🚀 Key Architectural Features
1. **Automated Pipeline Production:** Integrates ColumnTransformer containing OrdinalEncoder for demographic attributes and StandardScaler for biometric continuity.
2. **Multi-Model Analytics Engine:** Simulates, evaluates, and dynamically cross-validates (5-Fold CV) three regression frameworks simultaneously.
3. **Interactive UI Evaluation:** Enables users to generate real-time evaluations alongside deep visual insights (Feature Importance and Correlation Matrix Heatmaps).
