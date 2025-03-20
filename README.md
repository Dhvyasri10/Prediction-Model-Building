# Prediction-Model-Building
# 🚀 **PowerCo Churn Prediction – Model Building**

## 📊 **Project Overview**
This project focuses on building a **predictive model** to identify customer churn at PowerCo, a gas and electricity utility company.  

### ✅ **Objectives**
- Train a **Random Forest Classifier** using the engineered dataset (`data_for_predictions.csv`) to predict customer churn.  
- Evaluate the model's performance using appropriate **classification metrics**:  
    - **Accuracy**
    - **Precision**
    - **Recall**
- Visualize **feature importance** to identify the most influential factors driving churn.  
- Present the findings with clear **comments and explanations**.  

---

## 📁 **Dataset Description**
The project uses the **final dataset** named `data_for_predictions.csv`, which contains:  
- **Cleaned and engineered features** from previous data processing steps.  
- The dataset includes customer details, usage patterns, pricing variations, and a **churn indicator** as the target variable.  

---

## ⚙️ **Tech Stack and Libraries**
- **Language:** Python  
- **Libraries:**  
    - `pandas`: for data manipulation  
    - `numpy`: for numerical operations  
    - `matplotlib`: for visualizations  
    - `seaborn`: for statistical plots  
    - `plotly`: for interactive visualizations  
    - `scipy`: for statistical transformations  
    - `sklearn`: for building the random forest model and evaluation metrics  
- **IDE:**  
    - Jupyter Notebook for interactive analysis and visualizations  

---

## 🔥 **Model Building Steps**

### ✅ **1. Data Loading and Preparation**
- Loaded the `data_for_predictions.csv` dataset using `pandas`.  
- Performed **train-test split** using `sklearn` to separate the data into:  
    - **Training set:** 80% of the data used to train the model.  
    - **Testing set:** 20% of the data used for model evaluation.  
- Ensured the target variable `churn` is properly encoded (0 for non-churners, 1 for churners).  

---

### 🔥 **2. Random Forest Model Training**
- Defined the **Random Forest Classifier** with key parameters:  
    - `n_estimators=100`: Number of trees in the forest.  
    - `max_depth`: To prevent overfitting by limiting tree depth.  
    - `random_state=42`: For reproducibility.  
- Trained the model on the **training set** using `.fit()` method.  

---

### 🔥 **3. Model Evaluation**
- Made **predictions** on the testing set using `.predict()`.  
- Evaluated the model performance using the following metrics:  

#### 📈 **Accuracy**
- Measures the **overall correctness** of the model.  
- Formula:  
    \[ \text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} \]  
- **Limitation:** Accuracy alone is not reliable in imbalanced datasets.  

#### 🔥 **Precision**
- Measures how many of the **positive predictions were actually correct**.  
- Formula:  
    \[ \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} \]  
- **Importance:** Precision is critical when the **cost of false positives is high** (e.g., predicting churn when it doesn’t happen).  

#### 🔥 **Recall**
- Measures how many of the **actual positives were correctly predicted**.  
- Formula:  
    \[ \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} \]  
- **Importance:** Recall is essential when the **cost of missing positives is high** (e.g., failing to detect churners).  

---

### 🔥 **4. Feature Importance Visualization**
- Extracted the **feature importance scores** from the trained Random Forest model.  
- Visualized the scores using a **bar chart** to highlight the most influential features driving churn.  
- Key insights:  
    - **Net margin** and **consumption over 12 months** were significant churn drivers.  
    - **Price sensitivity features** were scattered and did not emerge as strong predictors.  

---

## 📈 **Key Findings**

### ✅ **Model Performance Analysis**
- The **accuracy** was satisfactory, but **precision and recall** provided deeper insights.  
- The model accurately identified **non-churners** but struggled with churners, suggesting the features may need further refinement.  
- **False negatives** indicate that the model missed some churners, which could lead to customer retention issues in real-world applications.  

### 🔥 **Feature Importance**
- **Net Margin** and **Consumption Over 12 Months** were key features driving churn.  
- **Price sensitivity** did not have a significant impact, indicating that it may need further feature engineering or transformation.  


