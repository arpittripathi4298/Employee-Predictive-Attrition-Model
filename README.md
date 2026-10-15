# Employee-Predictive-Attrition-Model

This repository contains a machine learning pipeline designed to analyze and predict employee attrition. By integrating multiple HR datasets, the project builds a predictive model to identify the key drivers behind employee turnover. You can view the full code in the `Employee Attrition Predictive Model.ipynb` notebook.

## Project Overview

The core objective of this project is to consolidate employee data across various domains—demographics, performance, compensation, and absenteeism—to train a machine learning model capable of analyzing attrition risk. A Random Forest Classifier is utilized to extract feature importances, highlighting the specific factors that contribute most significantly to an employee's decision to leave the company.

## Datasets

The analysis merges four distinct CSV datasets based on a unique `EmployeeID`:
*   `employees_absenteeism_2000.csv`
*   `employees_performance_2000.csv`
*   `employees_compensation_2000.csv`
*   `employees_demographics_2000.csv`

## Workflow

1.  **Data Integration**: The four datasets are merged into a single comprehensive table (`Attrition_Table`) using the `EmployeeID`.
2.  **Data Preprocessing**:
    *   The target variable, `Attrition`, is mapped to binary values (`Yes`: 1, `No`: 0).
    *   The `EmployeeID` column is dropped as it is a non-predictive identifier.
    *   Categorical variables (`Department`, `Contract_Type`, `Mobility_Budget_Opt_In`) are one-hot encoded using `pd.get_dummies`.
3.  **Model Training**: 
    *   The data is split into training and testing sets (80/20 split) with stratification.
    *   A `RandomForestClassifier` is trained with 1000 estimators and balanced class weights to handle potential imbalances in the target variable.
4.  **Evaluation**: The model is evaluated using accuracy scores, confusion matrices, and a classification report.
5.  **Feature Importance Extraction**: The top drivers of attrition are extracted from the Random Forest model and visualized using Seaborn.
6.  **Export**: The resulting feature importances are saved to `Attrition Predictive Model.csv` for further reporting.

## Results & Key Findings

### Model Performance
*   **Accuracy**: The model achieves an overall accuracy of 84%.
*   **Classification Report Note**: While the overall accuracy is high, the precision and recall for the positive attrition class (1) are 0.00. This indicates a severe class imbalance that the current configuration does not fully capture, resulting in the model predicting the majority class. 

### Top 10 Drivers of Attrition
Based on the Random Forest feature importances, the top factors driving employee attrition are:
1.  **Base Salary (EUR)** (Importance: ~0.099)
2.  **Annual Bonus (EUR)** (Importance: ~0.098)
3.  **Commute Distance (km)** (Importance: ~0.097)
4.  **Average Overtime Hours Per Month** (Importance: ~0.092)
5.  **Salary Growth Rate (Pct)** (Importance: ~0.085)
6.  **Years Since Last Promotion** (Importance: ~0.084)
7.  **Training Hours Completed** (Importance: ~0.084)
8.  **Age** (Importance: ~0.078)
9.  **Vacation Days Taken** (Importance: ~0.069)
10. **Short Term Sick Leave Count** (Importance: ~0.040)

## Dependencies

To run this project, you will need the following Python libraries installed:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Usage

1. Clone the repository and ensure the four CSV datasets are located in your specified directory.
2. Open `Employee Attrition Predictive Model.ipynb`.
3. Update the file paths in the `pd.read_csv()` cells to point to your local data directory.
4. Run the Jupyter Notebook cells sequentially to merge the data, train the model, and view the visualization. 
5. The feature importance output will be automatically saved as a CSV file in your specified directory.
