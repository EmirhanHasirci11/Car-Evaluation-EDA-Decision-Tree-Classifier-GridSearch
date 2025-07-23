# Car-Evaluation-EDA-Decision-Tree-Classifier-GridSearch

## Overview

This project is a comprehensive analysis and classification task performed on the Car Evaluation dataset. The primary objective is to build a machine learning model that can accurately predict the acceptability of a car (`target` variable) based on its features.

The project employs a **Decision Tree Classifier**. An initial baseline model is first built and evaluated. Subsequently, **GridSearchCV** is used for hyperparameter tuning to find the optimal parameters for the Decision Tree, significantly improving its performance. The entire workflow, from data cleaning and exploratory data analysis (EDA) to model building and evaluation, is documented in the notebook.

## Dataset

The dataset used is the "Car Evaluation" dataset from the UCI Machine Learning Repository. It contains data on 1728 car instances, each described by six categorical attributes.

*   [KaggleLink](https://www.kaggle.com/code/emirhanhasrc/eda-feature-engineering-visualization)

### Features

The dataset consists of six predictor variables (features) and one multi-class target variable. All features are categorical.

| Feature         | Description                                     | Data Type | Possible Values                      |
|-----------------|-------------------------------------------------|-----------|--------------------------------------|
| `buying`        | The buying price of the car.                    | Categorical | `vhigh`, `high`, `med`, `low`        |
| `maint`         | The maintenance cost of the car.                | Categorical | `vhigh`, `high`, `med`, `low`        |
| `doors`         | The number of doors.                            | Categorical | `2`, `3`, `4`, `5more`               |
| `persons`       | The passenger capacity.                         | Categorical | `2`, `4`, `more`                     |
| `lug_boot`      | The size of the luggage boot.                   | Categorical | `small`, `med`, `big`                |
| `safety`        | The estimated safety level of the car.          | Categorical | `low`, `med`, `high`                 |
| **`target`**    | **(Target)** The evaluation class of the car.   | Categorical | `unacc`, `acc`, `good`, `vgood`      |

## Project Pipeline

### 1. Exploratory Data Analysis (EDA)

The initial phase involved a thorough exploration of the dataset to understand its characteristics.

*   **Data Inspection**: It was observed that the raw CSV file did not contain column headers. Headers were added manually for clarity.
*   **Missing Values**: The dataset was checked for null values and confirmed to be complete, with no missing data.
*   **Class Distribution**: An analysis of the `target` variable revealed that the dataset is **highly imbalanced**. The majority class, `unacc` (unacceptable), constitutes over 70% of the data, while classes like `good` and `vgood` are significant minorities. This imbalance is a critical factor to consider during model evaluation.
*   **Visualizations**:
    *   **Bar Plots**: Bar plots were generated for each feature to visualize the distribution of its categories.
    *   **Pie Chart**: A pie chart was used to visually represent the imbalanced class distribution of the `target` variable.

### 2. Data Preprocessing

To prepare the data for machine learning, the following preprocessing steps were performed:

1.  **Data Type Conversion**: The `doors` and `persons` columns, which contained string values like '5more' and 'more', were converted to numerical integers (`5` and `6`, respectively).
2.  **Train-Test Split**: The data was split into a training set (70%) and a testing set (30%) using `train_test_split`.
3.  **Ordinal Encoding**: Since all features are categorical and have an inherent order (e.g., `low` < `med` < `high`), `OrdinalEncoder` was used to convert them into a numerical format that preserves this order. A `ColumnTransformer` was employed to apply this encoding only to the specified categorical columns.

### 3. Model Training and Evaluation

#### a. Initial Decision Tree Classifier

A baseline **Decision Tree Classifier** with a `max_depth` of 3 was initially trained. This simple model achieved an accuracy of approximately **78.6%**. However, the classification report showed very poor performance for the minority classes (`good` and `vgood`), with precision and recall scores of 0. This is a classic symptom of a model trained on imbalanced data.

A visualization of the tree was plotted to understand its decision-making logic.

#### b. Hyperparameter Tuning with GridSearchCV

To improve the model's performance, **`GridSearchCV`** was used to perform an exhaustive search for the best hyperparameters. The following parameters and their ranges were tested:
*   `criterion`: `['gini', 'entropy', 'log_loss']`
*   `splitter`: `['best', 'random']`
*   `max_depth`: `[1, 2, 3, 4, 5, 15, None]`
*   `max_features`: `['sqrt', 'log2', None]`

`GridSearchCV` used 5-fold cross-validation to find the optimal combination.

## Results

The hyperparameter tuning process yielded a significantly more powerful and accurate model.

*   **Best Parameters Found**:
    *   `criterion`: 'entropy'
    *   `max_depth`: 15
    *   `max_features`: None
    *   `splitter`: 'best'

*   **Final Model Performance**: The optimized Decision Tree model achieved a test accuracy of **97.3%**.

*   **Classification Report (Tuned Model)**:
    ```
                  precision    recall  f1-score   support

             acc       0.91      0.98      0.94       121
            good       1.00      0.86      0.92        21
           unacc       1.00      0.97      0.99       356
           vgood       0.95      1.00      0.98        21

        accuracy                           0.97       519
       macro avg       0.97      0.95      0.96       519
    weighted avg       0.97      0.97      0.97       519
    ```
    The final report shows a dramatic improvement in precision and recall for all classes, including the minority classes, demonstrating the effectiveness of the tuning process.

*   **Confusion Matrix**: The confusion matrix for the tuned model visually confirms the high number of correct predictions across all four classes.
