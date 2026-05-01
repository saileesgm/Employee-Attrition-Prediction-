import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#preprocess - clean data, select useful features, seperate x and y, split train tesyt sets

def preprocess_data(df):
    data = df.copy() #prevents modification of original df
    data["Attrition"] = data["Attrition"].map({"Yes": 1, "No": 0})

    # numeric only selected
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_remove = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Attrition']
    feature_cols = [c for c in numeric_cols if c not in cols_to_remove]

    X = data[feature_cols].fillna(data[feature_cols].mean()) #if missing fill with coln mean
    y = data["Attrition"] #target 

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test, feature_cols