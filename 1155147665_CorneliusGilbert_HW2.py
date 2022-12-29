"""
Homework 2
Name: Cornelius Gilbert
SID: 1155147665
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

#Question 1
main_folder = "C:\\Users\\ASUS\\OneDrive - The Chinese University of Hong Kong\\CUHK\\Term 5 (Fall 2022)\\FINA4380\\Homework\\HW2\\hw2.xlsx"

df_equity = pd.read_excel(main_folder, sheet_name ="equity", index_col=0)
equity = df_equity.to_numpy()
equity = np.diff(equity, axis=0)/equity[:-1,:]

df_factor = pd.read_excel(main_folder, sheet_name ="factor", index_col=0)
factor = df_factor.to_numpy()
factor = np.diff(factor, axis=0)/factor[:-1,:]

#Question 2
reqExp = 0.8
reqCorr = 0.4
reqFCorr = 0.7

#Question 3
cov_factor = np.cov(factor, rowvar = False)

eig_value, eig_vector = np.linalg.eig(cov_factor)
sort = eig_value.argsort()[::-1]
eig_value = eig_value[sort]
eig_vector = eig_vector[:,sort]

sum_lambda = np.sum(eig_value)
for i in range (len(eig_value)):
    power = np.sum(eig_value[:i+1])/sum_lambda
    number_pcs = i+1
    if power >= reqExp:
        break

pc = np.matmul(factor, eig_vector[:,:number_pcs])

#Question 4
all_factor = list(range(factor.shape[1]))
important_factor = []

for i in range (number_pcs):
    for j in all_factor:
        corr = pearsonr(factor[:,j], pc[:,i])[0]
        if not important_factor:
            if abs(corr) >= reqCorr:
                important_factor.append(j)
        else:
            fCorr = []
            for k in important_factor:
                fCorr.append(abs(pearsonr(factor[:,j], factor[:,k])[0]))
            if abs(corr) >= reqCorr and max(fCorr)<=reqFCorr:
                important_factor.append(j)

important_factor_label = list(df_factor.columns[important_factor])

#Question 5
scaler = StandardScaler()
equity_scaled = scaler.fit_transform(equity)

scaler2 = StandardScaler()
factor_scaled = scaler2.fit_transform(factor[:,important_factor])

#Question 6
beta=[]
t_val=[]
rsq=[]

X = sm.add_constant(factor_scaled)
for i in range(equity_scaled.shape[1]):
    model = sm.OLS(equity_scaled[:,i], X)
    res = model.fit()
    rsq.append(res.rsquared)
    t_val.append(res.tvalues)
    beta.append(res.params)

important_factor_label.insert(0, "const")
row_name = list(df_equity)

df_rsq = pd.DataFrame(rsq, index = row_name)
df_tval = pd.DataFrame(t_val, columns = important_factor_label, index = row_name)
df_beta = pd.DataFrame(beta, columns = important_factor_label, index = row_name)

df_rsq.to_csv('C:\\Users\\ASUS\\OneDrive - The Chinese University of Hong Kong\CUHK\Term 5 (Fall 2022)\\FINA4380\\Homework\\HW2\\Rsq.csv')
df_tval.to_csv('C:\\Users\\ASUS\\OneDrive - The Chinese University of Hong Kong\CUHK\Term 5 (Fall 2022)\\FINA4380\\Homework\\HW2\\tvalue.csv')
df_beta.to_csv('C:\\Users\\ASUS\\OneDrive - The Chinese University of Hong Kong\CUHK\Term 5 (Fall 2022)\\FINA4380\\Homework\\HW2\\beta.csv')