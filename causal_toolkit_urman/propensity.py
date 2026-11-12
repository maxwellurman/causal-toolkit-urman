import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.linear_model import LogisticRegression,  LinearRegression

### Task 1
def ipw(df, ps_formula: str, T: str, Y:str):
    X = dmatrix(ps_formula, df, return_type="dataframe")
    model = LogisticRegression(penalty=None, max_iter=1000)
    model.fit(X, df[T])
    ps = model.predict_proba(X)[:, 1]
    df["ps"] = ps
    
    ATE = np.mean(
        df[Y] * ((df[T] - df["ps"]) / ((df["ps"]) * (1 - df["ps"]))))
    return ATE

### Task 2
def doubly_robust(df: pd.DataFrame, formula: str, T: str, Y: str):
    X = dmatrix(formula, df, return_type="dataframe")

    model = LogisticRegression(penalty=None, max_iter=1000)
    model.fit(X, df[T])
    ps = model.predict_proba(X)[:, 1]

    model_0 = LinearRegression().fit(X[df[T]==0], df.loc[df[T]==0, Y])
    mu_0 = model_0.predict(X)

    model_1 = LinearRegression().fit(X[df[T]==1], df.loc[df[T]==1, Y])
    mu_1 = model_1.predict(X)

    t = df[T].to_numpy()
    y = df[Y].to_numpy()

    ATE = np.mean(
        (t * (y - mu_1) / ps + mu_1)
        - ((1 - t) * (y - mu_0) / (1 - ps) + mu_0)
    )

    return float(ATE)