import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_predict

# Task 1: S-Learner
def s_learner_discrete(train, test, X, T, y):
    """Return a copy of the test DataFrame with an additional column name, "cate" that contains CATE for the test DataFrame."""
    
    features = X + [T]
    
    train_x = train[features]
    train_y = train[y]

    model = LGBMRegressor()
    model.fit(train_x, train_y)
    
    test_x_t_1 = test[features].copy()
    test_x_t_1[T] = 1
    test_x_t_0 = test[features].copy()
    test_x_t_0[T] = 0
    
    result = test.copy()
    result['cate'] = model.predict(test_x_t_1) - model.predict(test_x_t_0)
    return result

# Task 2: T-Learner
def t_learner_discrete(train, test, X, T, y):
    """Return a copy of the test DataFrame with an additional column name, "cate" that contains CATE for the test DataFrame."""
    
    train_0 = train[train[T] == 0]
    train_1 = train[train[T] == 1]

    model_0 = LGBMRegressor()
    model_0.fit(train_0[X], train_0[y])

    model_1 = LGBMRegressor()
    model_1.fit(train_1[X], train_1[y])

    result = test.copy()
    result['cate'] = model_1.predict(test[X]) - model_0.predict(test[X])
    return result

# Task 3: X-Learner
def x_learner_discrete(train, test, X, T, y):
    """Return a copy of the test DataFrame with an additional column name, "cate" that contains CATE for the test DataFrame."""

    # Propensity Score Model
    model_ps = LogisticRegression(penalty=None)
    model_ps.fit(train[X], train[T])
    test_ps = model_ps.predict_proba(test[X])[:, 1]

    # Outcome Models
    train_0 = train[train[T] == 0]
    train_1 = train[train[T] == 1]

    model_0 = LGBMRegressor()
    model_0.fit(train_0[X], train_0[y])

    model_1 = LGBMRegressor()
    model_1.fit(train_1[X], train_1[y])

    #pseudo treatment effects
    d_0 = model_1.predict(train_0[X]) - train_0[y]
    d_1 = train_1[y] - model_0.predict(train_1[X])

    # Second Stage Model
    model_tau_0 = LGBMRegressor()
    model_tau_0.fit(train_0[X], d_0)

    model_tau_1 = LGBMRegressor()
    model_tau_1.fit(train_1[X], d_1)

    # Final Estimate
    tau_0 = model_tau_0.predict(test[X])
    tau_1 = model_tau_1.predict(test[X])

    result = test.copy()
    result['cate'] = test_ps * tau_0 + (1- test_ps) * tau_1
    return result  

# Task 4: double_ml_cate
def double_ml_cate(train, test, X, T, y):
    """Return a copy of the test DataFrame with an additional column name, "cate" that contains CATE for the test DataFrame."""
    
    model_treatment = LGBMRegressor()
    train_t_res = train[T] - cross_val_predict(model_treatment, train[X], train[T], cv=5)
    
    model_outcome = LGBMRegressor()
    train_y_res = train[y] - cross_val_predict(model_outcome, train[X], train[y], cv=5)

    train_y_star = train_y_res / train_t_res
    w = train_t_res ** 2

    model_cate = LGBMRegressor()
    model_cate.fit(train[X], train_y_star, sample_weight=w)

    results = test.copy()
    results['cate'] = model_cate.predict(test[X])
    return results