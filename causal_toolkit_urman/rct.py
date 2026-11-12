import numpy as np
import pandas as pd

def calculate_ate_ci(data, alpha = 0.05):
    from scipy import stats
    z = np.abs(stats.norm.ppf(alpha/2))
    
    treatment_mean = data.loc[data['T'] == 1, 'Y'].mean()
    treatment_var = data.loc[data['T'] == 1, 'Y'].var()
    treatment_n = len(data.loc[data['T'] == 1, 'Y'])
    
    control_mean = data.loc[data['T'] == 0, 'Y'].mean()
    control_var = data.loc[data['T'] == 0, 'Y'].var()
    control_n = len(data.loc[data['T'] == 0, 'Y'])

    ATE_estimate = treatment_mean - control_mean
    SE_diff = np.sqrt((treatment_var/treatment_n) + (control_var/control_n))
    CI_upper = ATE_estimate + z*SE_diff
    CI_lower = ATE_estimate - z*SE_diff

    return (ATE_estimate, CI_lower, CI_upper) 

def calculate_ate_pvalue(data):
    from scipy import stats
    
    treatment_mean = data.loc[data['T'] == 1, 'Y'].mean()
    treatment_var = data.loc[data['T'] == 1, 'Y'].var()
    treatment_n = len(data.loc[data['T'] == 1, 'Y'])
    
    control_mean = data.loc[data['T'] == 0, 'Y'].mean()
    control_var = data.loc[data['T'] == 0, 'Y'].var()
    control_n = len(data.loc[data['T'] == 0, 'Y'])

    ATE_estimate = treatment_mean - control_mean
    SE_diff = np.sqrt((treatment_var/treatment_n) + (control_var/control_n))
    
    t_statistic = (ATE_estimate - 0) / SE_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(t_statistic)))

    return (ATE_estimate, t_statistic, p_value)