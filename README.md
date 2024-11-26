# Predicting Diabetes Using Machine Learning and Deep Learning Models

> This README acts as our **one page** presentation answering the questions as well as presenting the results of our project.
> Note for TA and instructor: Some of these models were trained on the BigRed200 supercomputer, which has a lot of computational power. If you try replicating the results on your local machine, it will take a lot of time to train the models. We recommend using a machine with a good GPU or a cloud-based service like Google Colab.

**Presenters**: `Raja Allmdar Tariq Ali` and `Avinash Pandey`

## Introduction
- Our objective was to develop a predictive model for diabetes using machine learning and deep learning techniques
- The motivation behind this project was to address the rising global challenge of diabetes and the potential of predictive analytics

![study](img/image0.png)

**Fig 0:** Number of new T2DM, and incidence rate of T2DM across Brazil, China, India, Russian Federation, and South Africa between 1990 and 2019
- Data was [sourced from kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

## Exploratory Data Analysis
- Features
  - **Numerical**: Age, BMI, HbA1c level, Blood Glucose Level
   - **Categorical**: Gender, Hypertension, Heart Disease, Smoking History

- Target Variable 
  - **Diabetes**: 0 (No Diabetes), 1 (Diabetes)
- There were no missing values in the dataset


| Attribute             | Count       | Mean       | Std Dev     | Min    | 25%    | 50%    | 75%    | Max     |
|-----------------------|-------------|------------|-------------|--------|--------|--------|--------|---------|
| Age                   | 100,000     | 41.89      | 22.52       | 0.08   | 24.00  | 43.00  | 60.00  | 80.00   |
| BMI                   | 100,000     | 27.32      | 6.64        | 10.01  | 23.63  | 27.32  | 29.58  | 95.69   |
| HbA1c Level           | 100,000     | 5.53       | 1.07        | 3.50   | 4.80   | 5.80   | 6.20   | 9.00    |
| Blood Glucose Level   | 100,000     | 138.06     | 40.71       | 80.00  | 100.00 | 140.00 | 159.00 | 300.00  |

**Fig 1:** Data Distribution of Numerical Features

![histogram of age](img/image.png)

**Fig 2:** Histogram of Age

![histogram of bmi](img/image2.png)

**Fig 3:** Histogram of BMI

![histogram of hba1c](img/image3.png)

**Fig 4:** Histogram of HbA1c Level

![histogram of blood glucose](img/image4.png)

**Fig 5:** Histogram of Blood Glucose Level

![data distribution](img/image5.png)

**Fig 6:** Data Distribution of Diabetic and Non-diabetic Patients

![gender](img/image8.png)

**Fig 7:** Data Distribution of Categorical Column `Gender`

![smoking](img/image9.png)

**Fig 8:** Data Distribution of Categorical Column `smoking_history`

![hypertension](img/image10.png)

**Fig 9:** Histogram of `hypertension`

![heart disease](img/image11.png)

**Fig 10:** Histogram of `heart_disease`



## Challenges in Data Cleaning and Preprocessing
1. **Handling Outliers**: 
   - The Body Mass Index (BMI) feature indicated extreme values, indicating the presence of outliers that could potentially skew the model's performance
   - Outliers can distort certain statistical measures like the mean and standard deviation.
   - They may mislead the learning algorithms, causing inaccurate predictions
   - **Solution**: We used the Interquartile Range (IQR) method to detect and remove outliers from the BMI feature
      - This is the lower bound being used for outlier detection: `14.705`
      - This is the upper bound being used for outlier detection: `38.504999999999995`
      - Number of Outliers Detected in BMI: `7086`


![boxplot of bmi](img/image6.png)

**Fig 11:** Boxplot of BMI

![class distribution](img/image7.png)

**Fig 12:** Class Distribution Before and After Outlier Removal

2. **Handling Imbalanced Data**:
   - The dataset was imbalanced with a higher number of non-diabetic patients compared to diabetic patients
   - Imbalanced data can lead to biased models that predict the majority class more accurately than the minority class
   - **Solution**: We applied multiple techniques such as `Synthetic Minority Oversampling Technique` (SMOTE) and  `SMOTE-Tomek` to balance classes *[More details regarding these techniques will be discussed later]*

3. **Categorical Variables Encoding**:
   - The dataset contained categorical variables that needed to be encoded for the machine learning models
   - **Solution**: We used the `One-Hot Encoding` technique to convert categorical variables into numerical form

![one hot encoding](img/image13.png)

**Fig 13:** Code Snippet for One-Hot Encoding

4. **Ambiguities in smoking history categories (e.g., `"ever"`)**
    - `Not Current` = Refers to individuals who used to smoke but are currently not smoking
    - `Former` = Individuals who used to smoke but are currently not smoking AND have been abstinent for a longer period of time than those in the "not current" category.
    - `Current` = Indicates that the individual is currently a smoker at the time of data collection. It means that the person is actively smoking or has reported smoking recently.
    - `Ever` = Represents individuals who have ever smoked in their lifetime, regardless of their current smoking status (e.g. Current and Former). It includes individuals who are currently smoking ("Current"), as well as those who have previously smoked but may have quit at the time of data collection ("Former").
    - The `"ever"` category was ambiguous as it could include both current and former smokers, making it challenging to interpret the data accurately. Therefore, we decided to drop this category from the analysis.



## Exploratory Data Analysis [Continued]
1. **Correlation Matrix**:
   - We analyzed the correlation between numerical features and the target variable
   - The correlation matrix helped identify the relationship between features and the target variable
   - **Key Insights**:
     - Age and BMI had a positive correlation with diabetes
     - HbA1c level and Blood Glucose level had a strong positive correlation with diabetes


![correlation matrix](img/image12.jpeg)

**Fig 14:** Correlation Matrix


## Model Selection
1. **Decision Tree Classifier**:
   - Decision Trees are a popular choice for classification tasks due to their interpretability and ease of implementation
   - When tree depths are handled correctly, decision trees can provide a good balance between bias and variance and avoid overfitting
   - Good baseline model for predicting diabetes

2. **Random Forest Ensemble**:
   - Random Forest is an ensemble learning method that combines multiple decision trees to improve the model's performance
   - Random Forest reduces overfitting by averaging the predictions of multiple decision trees
   - Mainly used for handling non-linear relatonships as well as feature importance

3. **Support Vector Machine (SVM)**:
   - SVM is a powerful classification algorithm that can handle both linear and non-linear data
   - SVM is effective in high-dimensional spaces and is versatile in handling different kernel functions

4. **Feedforward Neural Network (FNN) with Multiple Layer Perceptron (MLP)**:
   - FNN is a deep learning model that consists of multiple layers of neurons
   - FNNs are effective in capturing non-linear relationships in the data

**Considerations for Model Selection**:
- We chose these models based on their ability to handle non-linear relationships and their performance in classification tasks
- We also made sure to select models that could handle imbalanced data effectively
- We had to balance interpretability and performance when selecting the models


## Model Implementation and Evaluation
1. Random Forest
   - Initial model might be overfitting due to class imbalance.
   - Applied SMOTE for balancing.
   - Hyperparameter tuning using GridSearchCV.
2. Decision Tree
   - Experimented with `max_depth`, `min_samples_split`, and `min_samples_leaf`.
   - Final model selected based on optimal trade-off between bias and variance
3. Support Vector Machine (SVM)
   - Applied standard scaling due to sensitivity to feature scales.
   - Experimented with different kernels (linear and rbf more specifically).
   - Handled imbalance using SMOTE and SMOTE-Tomek
4. FNN with MLP
   - Used Basic Architecture at first
   - Addressed class imbalance using SMOTE
   - Enhanced model with batch normalization, regularization, and learning rate adjustments.

![model implementation](img/image15.png)

**Fig 15:** Results


## Key Factors Influencing Model Performance
1. Models performed better after balancing the dataset using `SMOTE` and `SMOTE-Tomek`
2. Adding interaction terms and transformations improved model performance. Below is a code block on what exactly we did:
```python
# Adding interaction terms
# We add these interaction terms to capture the combined effects of these variables
# For example, the risk of diabetes might be higher in individuals who have both high BMI and are older.
Kaggle_Diabetes_Data['age_bmi_interaction'] = Kaggle_Diabetes_Data['age'] * Kaggle_Diabetes_Data['bmi']
Kaggle_Diabetes_Data['hypertension_heart_interaction'] = Kaggle_Diabetes_Data['hypertension'] * Kaggle_Diabetes_Data['heart_disease']

# Applying Transformations
Kaggle_Diabetes_Data['log_bmi'] = np.log(Kaggle_Diabetes_Data['bmi'] + 1) # Adding 1 to avoid log(0)
Kaggle_Diabetes_Data['sqrt_age'] = np.sqrt(Kaggle_Diabetes_Data['age'])
```

3. Optimizing parameters like `n_estimators`, `max_depth`, and learning rates during hyperparamter tuning (using `GridSearchCV`) enhanced model accuracy and ROC AUC.
4. Neural networks captured complex patterns but required careful tuning to prevent overfitting.
5. Standardizing features improved the performance of SVM and neural networks


![maxdepth](img/image14.png)

**Fig 16:** Impact of `max_depth` on Model Performance (Decision Tree Classifier)

![training history](img/image16.png)
![training history](img/image17.png)

**Fig 17:** Training History for Multiple Layers Perceptron (MLP) Model, showing signs of neither underfitting or overfitting


## Interpretation of Results
- In the Random Forest, feature importance indicated `blood_glucose_level` and `HbA1c_level` as good predictors.
- The decision tree classifier showed something similar, with `blood_glucose_level` and `HbA1c_level` being the most important features

![decision tree](img/image18.png)

**Fig 18:** Decision Tree Classifier

- For the FNN with MLP, despite lower accuracy after balancing, the recall and ROC AUC improved significantly, indicating better performance in predicting diabetic patients
- SVM struggled with the high-dimensionality and imbalance without proper tuning and balancing techniques.


## Insights from Data and Analysis
- Blood Glucose Level and HbA1c Level are strong indicators.
- Age and BMI interactions play a significant role.
- Proper handling of class imbalance is critical for building effective models.
- Neural networks outperform traditional models in capturing complex patterns when properly tuned and balanced.

## Conclusion
> Best Performing Model: Feedforward Neural Network with Multiple Layer Perceptron (MLP)