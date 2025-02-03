import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
data_cleaned['Log_Price'] = np.log(data_cleaned["Price"])
f, (ax1,ax2,ax3) = plt.subplots(1,3, sharey= True, figsize = (15,3))

ax1.scatter(data_cleaned['Year'],data_cleaned['Log_Price'])
ax1.set_title("Log_Price and Year")

ax2.scatter(data_cleaned['EngineV'],data_cleaned['Log_Price'])
ax2.set_title("Log_Price and EngineV")

ax3.scatter(data_cleaned['Mileage'],data_cleaned['Log_Price'])
ax3.set_title("Log_Price and Mileage")
# plt.show()
data_cleaned = data_cleaned.drop('Price', axis= 1)

variables = data_cleaned[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables,i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns

data_no_multicollinearity = data_cleaned.drop('Year',axis= 1)
print(data_no_multicollinearity)