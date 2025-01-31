import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
sns.set_theme()


raw_data = pd.read_csv(r'C:\Users\Noc\Desktop\line\001 1.04.Real-life-example.csv')
data = raw_data.drop(['Model'],axis=1)
data_no_mv = data.dropna(axis=0)
# sns.displot(data_no_mv['Price'])
q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage']<q]
data_3 = data_2[data_2['EngineV']<6.5]
q= data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]
data_cleaned = data_4.reset_index(drop= True)
print(data_cleaned)