# Diabetes Prediction Analysis

## Author: Chao Gao

Git-hub repository at: 
https://github.com/Chaoccz/DiabetesAnalysis

- Jupyter notebook: **Diabetes Prediction.ipynb**
- Data Set: diabetes_prediction_dataset.csv

  ![diabetes](diabetes.jpg)

# Table of contents
[Introduction](#introduction)

1. [Exploratory Data Analysis](#eda)
2. [Correlation](#correlation)
3. [Predcitive Analysis](#pa)

[Summary](#summary)

[Suggestion](#suggestion)

## 1. Introduction <a name = 'introduction'></a>
The aim of this analysis is to investigate a range of health-related factors and their interconnections **to classify diabetes accurately**. These factors include aspects such as **age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level**. This comprehensive examination will not only provide insights into the patterns and trends in diabetes risk but will also create a solid base for further research. Specifically, research can be built on how these variables interact and influence diabetes occurrence and progression, crucial knowledge for improving patient care and outcomes in this increasingly critical area of healthcare.

### Domain Knowledge
1. Age: Age is an important factor in predicting diabetes risk. As individuals get older, their risk of developing diabetes increases. This is partly due to factors such as reduced physical activity, changes in hormone levels, and a higher likelihood of developing other health conditions that can contribute to diabetes.

2. Gender: Gender can play a role in diabetes risk, although the effect may vary. For example, women with a history of gestational diabetes (diabetes during pregnancy) have a higher risk of developing type 2 diabetes later in life. Additionally, some studies have suggested that men may have a slightly higher risk of diabetes compared to women.

3. Body Mass Index (BMI): BMI is a measure of body fat based on a person's height and weight. It is commonly used as an indicator of overall weight status and can be helpful in predicting diabetes risk. Higher BMI is associated with a greater likelihood of developing type 2 diabetes. Excess body fat, particularly around the waist, can lead to insulin resistance and impair the body's ability to regulate blood sugar levels.

4. Hypertension: Hypertension, or high blood pressure, is a condition that often coexists with diabetes. The two conditions share common risk factors and can contribute to each other's development. Having hypertension increases the risk of developing type 2 diabetes and vice versa. Both conditions can have detrimental effects on cardiovascular health.

5. Heart Disease: Heart disease, including conditions such as coronary artery disease and heart failure, is associated with an increased risk of diabetes. The relationship between heart disease and diabetes is bidirectional, meaning that having one condition increases the risk of developing the other. This is because they share many common risk factors, such as obesity, high blood pressure, and high cholesterol.

6. Smoking History: Smoking is a modifiable risk factor for diabetes. Cigarette smoking has been found to increase the risk of developing type 2 diabetes. Smoking can contribute to insulin resistance and impair glucose metabolism. Quitting smoking can significantly reduce the risk of developing diabetes and its complications.

7. HbA1c Level: HbA1c (glycated hemoglobin) is a measure of the average blood glucose level over the past 2-3 months. It provides information about long-term blood sugar control. Higher HbA1c levels indicate poorer glycemic control and are associated with an increased risk of developing diabetes and its complications.

8. Blood Glucose Level: Blood glucose level refers to the amount of glucose (sugar) present in the blood at a given time. Elevated blood glucose levels, particularly in the fasting state or after consuming carbohydrates, can indicate impaired glucose regulation and increase the risk of developing diabetes. Regular monitoring of blood glucose levels is important in the diagnosis and management of diabetes.

**✔️These features, when combined and analyzed with appropriate statistical and machine learning techniques, can help in predicting an individual's risk of developing diabetes**.

### Preface
In this analysis, we have chosen the RandomForest classifier as our model. The RandomForest algorithm is an ensemble learning method that operates by constructing a multitude of decision trees during training and outputting the class that is the mode of the classes for classification or mean prediction of the individual trees for regression.

Several reasons guided our choice of Random Forest for this task:

1. Handling of Large Data: Random Forest is capable of efficiently handling large datasets with high dimensionality. Our dataset, containing a substantial number of rows and several features, falls into this category.

2. Robustness to Overfitting: Random Forest reduces the risk of overfitting, which is a common problem with decision trees. The algorithm accomplishes this by creating a set of decision trees (a "forest") and making the final prediction based on the majority vote of the individual trees.

3. Handling Mixed Data Types: In our dataset, we have both numerical and categorical features. Random Forest handles such mixtures smoothly, which makes it an ideal choice.

4. Feature Importance: Random Forest provides a straightforward way to estimate feature importance. Given our aim to investigate the impact of different factors on diabetes, this characteristic is particularly useful.

5. Non-linearity: Medical data often contains complex and non-linear relationships. Random Forest, being a non-linear model, can capture these relationships effectively.

<div class="alert alert-block alert-info">⚠️It's worth noting that while Random Forest is a strong candidate given its mentioned advantages, the choice of model should always be considered with a grain of salt. Other models might perform better on the task, and it's generally a good practice to try several models and compare their performance. However, for the purpose of this analysis and given our dataset, Random Forest is a practical and reasonable starting point.</div>

### Data Overview
  ![1](images/1.png)

## 1. Exploratory Data Analysis <a name = 'eda'></a>

### 1. Data Quality

#### Handling Duplicates
We will be handling duplicates by dropping the duplicated data:

  ![2](images/2.png)

#### Uniqueness

  ![3](images/3.png)

#### Missing Values

  ![4](images/4.png)

#### Describe the Data

  ![5](images/5.png)

### 2. Univariate Analysis

#### Histogram for age

  ![6](images/6.png)

#### Bar plot for gender

  ![7](images/7.png)

#### Distribution plot for BMI

  ![8](images/8.png)


#### Count plots for binary variables

  ![9](images/9.png)
  ![10](images/10.png)

#### Count plot for smoking history

  ![11](images/11.png)

### 3. Bivariative Analysis

#### Boxplot BMI vs Diabetes classification

  ![12](images/12.png)

####  Boxplot Age vs Diabetes classification

  ![13](images/13.png)

#### Count plot of gender vs diabetes

  ![14](images/14.png)

#### Boxplot HbA1c level vs Diabetes classification

  ![15](images/15.png)

#### Boxplot blood glucose level vs Diabetes classification

  ![16](images/16.png)

#### Pair plot for numeric features

  ![17](images/17.png)

### 4. Multivariate Analysis

#### Scatterplot Age vs BMI colored by Diabetes classification

  ![18](images/18.png)

#### Violin plot of BMI against diabetes classification split by gender

  ![19](images/19.png)

#### Interaction between gender, BMI and diabetes

  ![20](images/20.png)

#### Interaction between gender, Age and diabetes

  ![21](images/21.png)

## 2. Correlation <a name = 'correlation'></a>

### 1. Data Preparation
We define a function to map the existing categories to new ones, then apply the function to the 'smoking_history' column, and then check the new value counts:

  ![22](images/22.png)

### 2. Encoding
We perform one-hot encoding on the gender variable, and then perform one-hot encoding on the smoking history variable

### 3. Correlation Matrix

  ![23](images/23.png)
  ![24](images/24.png)

## 3. Predictive Analysis <a name = 'pa'></a>

### 1. Class Imbalance
From the EDA ,the dataset is imbalanced (with 9% positive cases for diabetes and 91% negative cases), it's essential to balance the data to ensure that the model doesn't get biased towards the majority class. For this purpose, the Synthetic Minority Over-sampling Technique (SMOTE) is used, which generates synthetic samples for the minority class.

  ![25](images/25.png)

### 2. Preprocessing: Scaler
Preprocessing is a crucial step before training the model. In this case, numerical features are standardized (mean removed and scaled to unit variance), and categorical features are one-hot encoded. <font color = 'blue'>Standardization</font> is not required for all models but is generally a good practice. <font color = 'blue'>One-hot encoding</font> is necessary for categorical variables to be correctly understood by the machine learning model.

The <font color = 'blue'>StandardScaler</font> in sklearn is based on the assumption that the data, Y, follows a distribution that might not necessarily be Gaussian (normal), but we still transform it in a way that its distribution will have a mean value 0 and standard deviation of 1.
This transformation is also known as Z-score normalization.

### 3. Model Building and Hyperparamter Tuning
A pipeline is constructed which first applies the preprocessing steps and then trains a model on the data. We use a <font color = 'blue'>RandomForestClassifier</font>, which is a popular and powerful algorithm for classification tasks. The model's hyperparameters are tuned using <font color = 'blue'>GridSearchCV</font> , which performs an exhaustive search over the specified parameter values for the estimator. The best performing model is selected based on cross-validation.

  ![26](images/26.png)

#### Intepret the results
The result shows the best parameters for our <font color = 'blue'>Random Forest model</font> that were found during the hyperparameter tuning process:

1. **max_depth of 10**: This indicates that the maximum depth of the trees in the forest is 10 levels. Constraining the depth of the tree helps in reducing overfitting. It appears from this result that a medium-complexity tree works best for our data. Too much complexity (a deeper tree) may capture noise, and too little (a shallower tree) may not capture the underlying structure of the data.

2. **min_samples_leaf of 1**: This means that each leaf (the end node of a decision tree, where predictions are made) must contain at least one sample. This parameter, like max_depth, is a way to control overfitting. By requiring at least one sample to make a prediction, the model prevents fitting to outliers or noise in the training data.

3. **min_samples_split of 10**: This tells us that a node must contain at least ten samples in order to be split (to create two child nodes). Similar to the min_samples_leaf parameter, this can help control overfitting.

4. **n_estimators of 50**: This is the number of decision trees in the forest. The Random Forest algorithm works by averaging the predictions of many decision trees to make a final prediction, which helps reduce overfitting and variance. In this case, it seems that having 50 trees in the forest gives us the best performance.

 <div class="alert alert-block alert-info">💬These parameters are a result of the Hyperparameter tuning process , and they give us insight into the structure of the data and the complexity of the model that best captures that structure. The moderately constrained tree depth and the requirements for the number of samples at each node suggest a model that is complex enough to capture the important patterns in the data, but not so complex that it overfits to noise or outliers.This balance is crucial in creating a model that will generalize well to new data.</div>

  ![27](images/27.png)

### 4. Confusion Matrix

  ![28](images/28.png)

#### Intepret the results
Our trained <font color = 'blue'>Random Forest Model</font> achieved an accuracy of around 95%. This indicates that the model correctly classified around 95% of all cases in the test set.

Looking deeper into the classification metrics, let's dissect the performance for each class (0 and 1) separately:

**A | Class 0 (Non-diabetes):** 
The model has a high precision (0.98) for class 0, meaning that among all instances where the model predicted non-diabetes, 98% were indeed non-diabetes.
The recall for class 0 is also high (0.96). This means that our model correctly identified 96% of all actual non-diabetes cases in the dataset.

**B | Class 1 (Diabetes):**
The precision for class 1 is lower around (0.65), which indicates that when the model predicted diabetes, it was correct around 65% of the time.
However, the recall is reasonably high around (0.80). This means that our model was able to capture around 80% of all actual diabetes cases.
The F1 score, a harmonic mean of precision and recall, is around 0.97 for class 0 and around 0.72 for class 1. The weighted average F1 score is around 0.94, in line with the overall accuracy.

This discrepancy in performance between classes is likely due to the imbalance in the original dataset. Class 0 (Non-diabetes) is the majority class and has more examples for the model to learn from.

However, the higher recall for class 1 (Diabetes) is promising. This is an essential aspect for a healthcare model, as missing actual positive cases (false negatives) can have serious implications.

<div class="alert alert-block alert-info">📝In summary, while our model performs well overall, it particularly excels with the majority class (non-diabetes). To enhance performance on the minority class (diabetes), we can further address class imbalance or adjust model parameters. Despite these areas for improvement, the model's ability to accurately identify a high percentage of actual diabetes cases is encouraging at this early stage of model development. Subsequent iterations and refinements are expected to enhance precision in diabetes predictions without compromising recall.</div>

### 5. Feature Importance
Finally, the importance of each feature is computed. This is the total decrease in node impurity (weighted by the probability of reaching that node, which is approximated by the proportion of samples reaching that node) averaged over all trees of the ensemble. **The feature importance gives insight into which features are most useful for making predictions.** The features are ranked by their importance and visualized using a bar plot.

  ![29](images/29.png)

#### Intepret the results
The feature importance results provide insight into which features are most influential in predicting diabetes using our <font color = 'blue'>Random Forest Model</font>. The importance of a feature is calculated based on how much the tree nodes that use that feature reduce impurity across all trees in the forest.

<font color = 'blue'>Here are the key findings from the feature importance results:</font>

1. **HbA1c_level** is the most important feature with an importance of 0.44. HbA1c is a measure of the average levels of blood glucose over the past 2 to 3 months, so it's not surprising that it's a significant predictor of diabetes.

2. **The blood_glucose_level** the second most important feature with an importance of 0.32. This aligns with medical knowledge, as blood glucose levels are directly used to diagnose diabetes.

3. **Age** the third most important feature with an importance of 0.14. It's well known that the risk of type 2 diabetes increases as you get older.

4. **BMI** comes fourth in terms of importance at 0.06. Body Mass Index is a key risk factor for diabetes, and its role is well documented in medical literature.

5. Other features like **hypertension** and **heart_disease** show some importance (0.02 and 0.01, respectively), indicating that these health conditions might have some relevance in predicting diabetes, though not as significant as the top four factors.

6. **Smoking history** ('smoking_history_non-smoker', 'smoking_history_past_smoker', 'smoking_history_current') and gender ('gender_Female', 'gender_Male') are shown to have minimal or zero importance in our model. This could be due to a number of reasons including that these factors may not be as influential in the development of diabetes or it could be a result of how the data was collected or structured.

<div class="alert alert-block alert-info">⚠️These results, however, should be interpreted with caution. The importance of a feature in a Random Forest model doesn't necessarily mean a casual relationship, and it is specific to this model and this dataset. Other models might find different results. Additionally, low importance doesn't mean that the feature is unimportant for predicting diabetes in general, it may just mean that the feature is not useful in the presence of the other features. A thorough feature analysis should be considered for a better understanding of the contribution of each feature in the prediction.</div>

Overall, our findings do align well with medical knowledge and literature about risk factors for diabetes. The most important features are blood-related measurements, followed by age and BMI, with less importance seen for comorbid conditions like hypertension and heart disease.

## Summary <a name = 'summary'></a>
The analysis employed a <font color = 'blue'>Random Forest classifier</font> to predict diabetes based on various health indicators and lifestyle factors. The model was trained and evaluated on a dataset of 100,000 records, and <font color = 'blue'>Hyperparameter tuning</font> was performed to optimize the model's performance.

The model achieved an **accuracy of approximately 95.1%**, with precision of 0.98 for class 0 (non-diabetic) and 0.69 for class 1 (diabetic). It was also able to recall 96% of non-diabetic cases and 81% of diabetic cases correctly. The relatively high accuracy and balanced performance on both classes indicate that the model is well-tuned and robust.

Feature importance analysis highlighted **HbA1c_level** and **blood_glucose_level** as the most critical factors in predicting <font color = 'blue'>Diabetes</font>. **Age** and **BMI** also showed significant importance. However, some features, such as **smoking history** and **gender**, had minimal or no impact on the model's predictions.

## Suggestion <a name = 'suggestion'></a>
1. <font color = 'blue'>**Data Collection**:</font> If further data collection is possible, we could aim to gather more information about lifestyle factors and other potential diabetes risk factors not covered in this dataset. For instance, detailed diet information, physical activity level, family history of diabetes, and more precise information on heart disease or hypertension might improve the model's predictive capabilities.

2. <font color = 'blue'>**Model Exploration**:</font> While the **Random Forest model** has performed well, it might be worth exploring other machine learning models. For instance, gradient boosting models like **XGBoost** or **LightGBM** could potentially offer improved performance.

3. <font color = 'blue'>**Feature Engineering**:</font> More sophisticated feature engineering could potentially improve model performance. Interaction features, polynomial features, or other transformations might be worth exploring.

4. <font color = 'blue'>**Model Interpretation**:</font> To better understand the influence of each feature, we could use interpretability tools such as **SHAP** (SHapley Additive exPlanations) or **permutation feature importance**, which can offer a more nuanced view of feature importance than traditional feature importance based on impurity reduction.

5. <font color = 'blue'>**Addressing Class Imbalance**:</font> Despite using **SMOTE** to balance the classes, there is still room for improvement in the performance metrics for the minority class. **Other oversampling methods, undersampling methods, or cost-sensitive learning methods could be explored to improve the recall and precision for the minority class.**
