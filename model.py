#Training + Single Prediction
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=120, random_state=42) #uses 120 trees, ensures repeatable results
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return model, acc


def prepare_input_row(columns, age, dr, income, dist, years, promo, sat, wlb, overtime):
    #ensures the prediction input matches the model's expected features
    df = pd.DataFrame(columns=columns)
    df.loc[0] = 0 #fills all colums with 0 temporarily

    mapping = {
        "Age": age,
        "DailyRate": dr,
        "MonthlyIncome": income,
        "DistanceFromHome": dist,
        "YearsAtCompany": years,
        "YearsSinceLastPromotion": promo,
        "JobSatisfaction": sat,
        "WorkLifeBalance": wlb,
    }
    #fill df with actual values
    for col, value in mapping.items():
        if col in df.columns:
            df[col] = value
    #convert overtime to numeric
    if "OverTime" in columns:
        df["OverTime"] = 1 if overtime == "Yes" else 0

    return df


def predict_single(model, input_df):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    return pred, prob