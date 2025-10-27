# 🏠 California Housing Price Prediction – Capstone Project

### Author: *Krishnan Ramaswami*

---

## 🧭 Executive Summary

### **Project Overview and Goals**

Housing markets are shaped by multiple interrelated factors such as **median income**, **average number of rooms**, **population density**, **occupancy rate**, and **geographic coordinates**.
The goal of this project is to **predict the median house value (`MedHouseValue`)** in California using regression-based machine learning models.
Accurate predictions support **policy makers**, **developers**, and **home buyers** in making data-driven real-estate decisions.

---

### **Findings**

After comparing multiple regression algorithms, the **Tuned Random Forest Regressor** emerged as the top performer.

| Rank | Model                     | R² Score   | MAE        | MSE        | Key Insight                             |
| ---- | ------------------------- | ---------- | ---------- | ---------- | --------------------------------------- |
| 🥇 1 | **Random Forest (Tuned)** | **0.7923** | **0.3514** | **0.2722** | Excellent bias–variance balance         |
| 🥈 2 | **Gradient Boosting**     | 0.7719     | 0.3771     | 0.2989     | Strong ensemble learner                 |
| 🥉 3 | **Ridge Regression**      | 0.6432     | 0.5048     | 0.4675     | Robust linear model with regularization |

The **SGD Regressor** was unstable, with extreme error values (R² ≈ –72,906), confirming it is unsuitable for this dataset.

---

### **Results and Conclusion**

#### ✅ **Key Evaluation Metrics**

* **Best Model:** Tuned Random Forest (R² = 0.792, MAE = 0.351, MSE = 0.272)
* **Second Best:** Gradient Boosting (R² = 0.772)
* **Third Best:** Ridge Regression (R² = 0.643)

#### 📊 **Visual Insights**

* **Correlation Heatmap:** Reveals strong correlation between `MedInc` (Median Income) and `MedHouseValue`.
* **Geographic Distribution Plot:** High-value houses cluster near the California coast (higher latitudes).
* **Feature Importances:** `MedInc`, `Latitude`, and `Longitude` dominate predictive power.

#### 📈 **Learning Curves**

* **SGD Regressor:** High variance, poor generalization.
* **Tuned SVR:** Moderate bias and better variance control after optimization.

#### 🧩 **Conclusion**

The Random Forest model offers a **robust, interpretable, and high-accuracy** solution for housing price prediction in California. Ensemble methods outperform linear models by capturing nonlinear relationships and regional effects.

---

### **Future Research and Development**

* Experiment with **XGBoost**, **LightGBM**, and **CatBoost** for enhanced performance.
* Integrate **spatial features** (proximity to coastlines, amenities, highways).
* Deploy the model as an **interactive web API or dashboard** using **Streamlit** or **FastAPI**.
* Explore **deep learning regressors** (e.g., MLPRegressor, TensorFlow/Keras) for more complex patterns.

---

### **Next Steps and Recommendations**

1. Incorporate **real-time housing data** and inflation-adjusted price indexes.
2. Conduct **feature selection** via recursive elimination for model simplification.
3. Build a **deployment pipeline** for reproducibility.
4. Use **cross-region validation** to generalize performance beyond California.
5. Monitor and retrain models periodically with updated data.

---

## 🧠 Rationale

This dataset was chosen because it represents **real-world housing economics** with quantifiable features.
Predicting median house values helps explore how **socioeconomic and geographic factors** drive market variations across California regions.

---

## ❓ Research Question

> “How accurately can we predict the median house value of California districts based on demographics, housing characteristics, and geography?”

---

## 📂 Data Sources

### **Dataset Source**

* **Origin:** [scikit-learn – California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
* **Rows:** 20,640
* **Columns:** 9
* **Target Variable:** `MedHouseValue`
* **Null Values:** None

**Columns:**
`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`, `MedHouseValue`

---

### **Exploratory Data Analysis**

EDA included:

* **Distribution plots** for all numerical features
* **Scatterplots** showing spatial and income-value relationships
* **Correlation heatmap** identifying linear dependencies
* **Geographic visualization indicated that high-value houses tend to cluster near the California coastline

**Key Insights:**

* Median income (`MedInc`) strongly correlates with house value.
* Houses near latitude ~37–38 (San Francisco) have the highest prices.
* Features are continuous and normally distributed after scaling.

---

### **Cleaning and Preparation**

* Verified dataset integrity: no missing or duplicate records.
* Removed no columns due to completeness.
* Ensured feature types were numeric and consistent.

---

### **Preprocessing**

* **Scaling:** Standardized numerical features using `StandardScaler`.
* **Encoding:** None required (all numeric).
* **Outlier Treatment:** IQR-based capping for extreme values.
* **Train-Test Split:** 80/20 for robust evaluation.

---

### **Final Dataset**

| Metric                            | Value                         |
| --------------------------------- | ----------------------------- |
| Rows                              | 20,640                        |
| Columns                           | 9                             |
| Removed Columns                   | None                          |
| Target                            | `MedHouseValue`               |
| Feature Count (after engineering) | 11 (including derived ratios) |

---

## ⚙️ Methodology

### **Machine Learning Models Applied**

| Model                                | Technique             | Description & Benefits                                                      |
| ------------------------------------ | --------------------- | --------------------------------------------------------------------------- |
| **Linear Regression**                | Baseline linear model | Provides interpretability and baseline comparison.                          |
| **Ridge Regression**                 | Regularized linear    | Reduces multicollinearity via L2 penalty.                                   |
| **Decision Tree Regressor**          | Tree-based            | Captures nonlinearities, interpretable via splits.                          |
| **Random Forest Regressor (Tuned)**  | Ensemble averaging    | Reduces variance and overfitting using multiple trees.                      |
| **Gradient Boosting Regressor**      | Sequential boosting   | Improves accuracy through iterative error correction.                       |
| **K-Nearest Neighbors (KNN)**        | Distance-based        | Non-parametric approach using neighborhood similarity.                      |
| **SGD Regressor**                    | Online optimization   | Lightweight but unstable for this dataset.                                  |
| **Support Vector Regressor (Tuned)** | Kernel-based          | Captures nonlinear trends using RBF kernel; tuned via `RandomizedSearchCV`. |

### **Model Optimization**

* **Random Forest:** Two-stage tuning (RandomizedSearchCV + GridSearchCV).
* **SVR:** Tuned over C, gamma, epsilon ranges using randomized search.
* **Learning Curves:** Used for bias–variance trade-off evaluation.

---

## 📊 Model Evaluation and Results

| Model                 | R²         | MAE    | MSE       |
| --------------------- | ---------- | ------ | --------- |
| Random Forest (Tuned) | 0.7923     | 0.3514 | 0.2722    |
| Gradient Boosting     | 0.7719     | 0.3771 | 0.2989    |
| Ridge Regression      | 0.6432     | 0.5048 | 0.4675    |
| Linear Regression     | 0.6432     | 0.5051 | 0.4675    |
| Tuned SVR             | 0.6303     | 0.5006 | 0.4845    |
| Decision Tree         | 0.5969     | 0.4775 | 0.5281    |
| KNN Regressor         | 0.4855     | 0.6192 | 0.6741    |
| SGD Regressor         | –72,906.26 | 245.05 | 95,538.32 |

📎 **Graphs Generated:**

* Correlation Heatmap (`images/Correlation Heatmap.png`)
* Feature Histograms (`images/housing_feature_histograms.png`)
* Median Income vs House Value (`images/median_income_vs_housevalue.png`)
* Geographic Distribution (`images/geographic_distribution.png`)
* Random Forest Actual vs Predicted (`images/Random Forest (Optimized) - Actual vs Predicted Prices.png`)
* Feature Importances (`images/Top 15 Feature Importances.png`)
* Model Comparison Charts (`images/Model Comparison - Normalized - R2 score.png`, etc.)
* Learning Curves for SGD & SVR
* SVR Epsilon Tube Visualization

---

## 🗂️ Outline of Project

1. Import libraries and load dataset
2. Perform EDA with visual summaries
3. Preprocess and scale features
4. Engineer additional features
5. Train multiple regression models
6. Tune hyperparameters (Random Forest & SVR)
7. Evaluate and compare performance
8. Generate dynamic summary and recommendations
9. Save all results and charts

---

## 📬 Contact and Further Information

For queries or collaborations:
📧 *[drkrishnan@yahoo.com](mailto:drkrishnan@yahoo.com)*
🔗 LinkedIn: https://www.linkedin.com/in/krishnan-devakottai-ramaswami-8b39684/

---

## 🧾 Code Implementation

The full Python code implementing this pipeline is provided in the Github location mentioned below 
https://github.com/drkrishnan/UC-Berkeley-ML-and-AI-Course-Capstone-Project---California-Housing-Price-Prediction

---

## 🖼️ Sample Output Summary

| Model                 | R²    | MAE   | MSE   |
| --------------------- | ----- | ----- | ----- |
| Random Forest (Tuned) | 0.792 | 0.351 | 0.272 |
| Gradient Boosting     | 0.772 | 0.377 | 0.299 |
| Ridge Regression      | 0.643 | 0.505 | 0.468 |
