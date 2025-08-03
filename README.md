# Heart-Disease-Prediction-
It is a ML model which will show the probability of Heart disease of the patient on the basis of different features.

💓 Heart Disease Prediction Web App
This project presents a Machine Learning-based web application designed to predict the likelihood of heart disease in individuals based on medical data. The primary goal is to offer a simple, interactive, and accessible platform that can assist in preliminary health screening using a trained Random Forest model.

📘 Project Overview
Heart disease remains one of the leading causes of death globally. Early detection can be life-saving. This project uses the Heart dataset and applies a Random Forest Classifier to predict whether a person is likely to have heart disease or not, based on 13 clinical features.

The project includes the full pipeline: data loading, preprocessing, exploratory data analysis (EDA), feature normalization, model training, evaluation, and deployment using Streamlit.

📂 Workflow Summary
1. Dataset
Source: Heart Dataset

Features: 13 predictive features + 1 target variable (target)

3. Data Preprocessing
Checked for missing/null values

Encoded categorical variables

Normalized numeric features where necessary

Separated features (X) and labels (y)

3. Exploratory Data Analysis
Generated visual insights using:

Scatter plots (age vs chol, thalach vs oldpeak)

Pairplot of all features

Correlation heatmap

Observed that age, chest pain type, and max heart rate had significant influence

4. Modeling
Model Used: RandomForestClassifier

Why Random Forest? It handles feature interactions well, reduces overfitting, and is highly accurate on structured data.

Model trained using train_test_split (80/20)

Achieved strong accuracy and robust performance in testing

5. Model Evaluation
Metrics used:

Accuracy

Confusion Matrix

Classification Report

ROC AUC Curve

Overall accuracy: ~85–90% on unseen test data

6. Model Saving
Trained model saved as heart_disease_model.pkl using joblib for deployment

7. Web App (Frontend with Streamlit)
Created an interactive web UI using Streamlit

Accepts user inputs via dropdowns, sliders, and number inputs

Automatically converts and formats the data for prediction

Displays result with intuitive visual feedback:

✅ Low Risk

⚠️ High Risk

8. UI Enhancements
Added professional styling using layout formatting and columns

Included header icons and prediction result illustrations:

heart_icon.png

predict_icon.png

9. Deployment
The app is deployable on Streamlit Cloud or locally

All code and assets are packaged for GitHub
