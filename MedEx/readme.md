# 🏥 MedEx: Medical Insurance Cost Prediction System
### *Machine Learning Regression Project | End-to-End Pipeline*

This project predicts **medical insurance charges** based on demographic and health-related factors.  
It follows a complete **Machine Learning workflow** including data cleaning, EDA, feature engineering, model building, tuning, evaluation, and deployment using **Streamlit**.

---

## 📌 Project Objective

The objective is to:

- Understand which factors influence medical charges  
- Build a model that accurately predicts costs  
- Compare multiple regression models  
- Deploy the final model in a user-friendly web app  

---

## 📂 Dataset Overview

The dataset contains **1,338 rows and 7 columns**:

| Column | Description |
|--------|-------------|
| age | Age of the person |
| sex | Male/Female |
| bmi | Body Mass Index |
| children | Number of dependent children |
| smoker | Smoker or non-smoker |
| region | Geographical region |
| charges | Actual medical insurance charges |

---

## 🔍 Project Workflow (Step-by-Step)

### **1️⃣ Problem Definition**
- Predict insurance `charges`
- Type: Regression  
- Target variable transformed to **log_charges** due to skewness

---

### **2️⃣ Basic Data Checks**
- Shape, datatypes  
- Null-value analysis  
- Duplicate checks  
- Summary statistics  
- Value counts  

---

### **3️⃣ Exploratory Data Analysis (EDA)**

#### **Univariate Analysis**
- Histograms  
- Boxplots  
- Countplots  

#### **Bivariate Analysis**
- charges vs smoker  
- charges vs region  
- charges vs sex  
- Scatterplots  

#### **Multivariate**
- Correlation heatmap  
- Pairplot
- 
---

### **4️⃣ Data Cleaning**
- Removed duplicates  
- Corrected strings (lowercasing/strip)  
- Outlier treatment using **IQR capping**  
- Created `log_charges` to fix skewness  
- Encoded categorical variables:  
  - Binary (sex, smoker)  
  - One-hot (region)  

---

### **5️⃣ Feature Engineering**
- Selected final features  
- Standardized features using `StandardScaler`  
- Final feature list:

```

['age','sex','bmi','children','smoker',
'region_northwest','region_southeast','region_southwest']

```

---

### **6️⃣ Train–Test Split**
- 80% Train  
- 20% Test  
- random_state = 42  

---

### **7️⃣ Model Building**
Trained multiple models:

- Linear Regression  
- Ridge  
- Lasso  
- ElasticNet  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost  

---

### **8️⃣ Model Evaluation**
Metrics:

- MAE  
- MSE  
- RMSE  
- R²  
- Adjusted R²  

### **🏆 Best Model:**  
**GradientBoosting Regressor**

Performance:

- RMSE ≈ **0.33**  
- R² ≈ **0.86**  
- MAE ≈ **0.18**  

---

### **9️⃣ Hyperparameter Tuning**
Used:

- `GridSearchCV`  
- `RandomizedSearchCV`

Tuned models: GradientBoosting, XGBoost, Ridge

---

### **🔟 Final Model Saving**
Saved with **pickle**:

```

final_model.pkl
final_scaler.pkl

````

These are loaded during deployment.

---

## 🚀 Deployment Using Streamlit

The Streamlit app allows users to input:

- Age  
- Sex  
- BMI  
- Children  
- Smoker  
- Region  

The app:

1. Encodes features  
2. Scales using saved scaler  
3. Predicts log_charges  
4. Converts back using exp()  
5. Displays estimated medical cost  

### Run locally:

```bash
streamlit run app.py
````

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Seaborn
* Matplotlib
* Scikit-Learn
* XGBoost
* Pickle
* Streamlit

---

## 🏁 Conclusion

This project demonstrates:

* A full machine learning lifecycle
* Real-world data preprocessing
* Handling skewness, outliers, encoding
* Training and comparing several regression models
* Saving and deploying the best model

**GradientBoosting Regressor** delivered the best accuracy and was deployed successfully.

---

## 🙌 Author

**Vandana Padhi**
Passionate about Machine Learning, Data Science, and Streamlit Deployments.


