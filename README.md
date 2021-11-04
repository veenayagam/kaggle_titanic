# Predict survival on the Titanic
# Project Overview 
* Use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.
* Used Titanic datasets from Kaggle.(kaggle link provided below)
* Optimized Logistic Regression, Random Forest Classifier, and Gradient Boosting Classifier using MLflow Tracking to reach the best model. 

## Code and Resources Used 
**Python Version:** 3.8  
**Packages:** pandas, numpy, mlflow, cloudpickle, psutil, scikit-learn, typing-extensions  
**For Web Framework Requirements:**  ```pip install -r requirements.txt```  
**Kaggle Dataset:** https://www.kaggle.com/c/titanic/data

## Data Collection
In the project, the dataset were collected from Kaggle.

## Feature Engineering
After the data has been collected, we need to clean it up before using in the models. I cleaned and changed variables as follows:

*	Drop features that are not required
*	Remove NA values
*	Performed log transformation
* Used Ordinal Encoding to transform categorical variables into dummy variables
*	Outlier Removal Using Business Logic

Outliner in Fare column after log transformation
![Fare_log_box_plot](https://user-images.githubusercontent.com/72549846/140296002-22f7ed0a-45a8-4b9a-ac66-3a5fd257f238.png)

## Model Building 
I split the data into train size of % and test size of 25% respectively.  

I implemented three different models and evaluated them with different parameters.
*	**Logistic Regression** – Classification model
*	**Random Forest Classifier**
*	**Gradient Boosting Classifier**

## Model Performance
The Gradient Boosting Classifier far outperformed the other models.
![model](https://user-images.githubusercontent.com/72549846/140297360-7def321d-30de-4006-8617-d542b02f5a5a.png)

