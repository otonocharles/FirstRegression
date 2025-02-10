import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
sns.set_theme()

# lINEAR REGRESSION WITH SKLEARN
# raw_data = pd.read_csv(r'C:\Users\Noc\Desktop\line\001 1.04.Real-life-example.csv')
# data = raw_data.drop(['Model'],axis=1)
# data_no_mv = data.dropna(axis=0)
# sns.displot(data_no_mv['Price'])
# q = data_no_mv['Price'].quantile(0.99)
# data_1 = data_no_mv[data_no_mv['Price']<q]
# q = data_1['Mileage'].quantile(0.99)
# data_2 = data_1[data_1['Mileage']<q]
# data_3 = data_2[data_2['EngineV']<6.5]
# q= data_3['Year'].quantile(0.01)
# data_4 = data_3[data_3['Year']>q]
# data_cleaned = data_4.reset_index(drop= True)
# data_cleaned['Log_Price'] = np.log(data_cleaned["Price"])
# f, (ax1,ax2,ax3) = plt.subplots(1,3, sharey= True, figsize = (15,3))

# ax1.scatter(data_cleaned['Year'],data_cleaned['Log_Price'])
# ax1.set_title("Log_Price and Year")

# ax2.scatter(data_cleaned['EngineV'],data_cleaned['Log_Price'])
# ax2.set_title("Log_Price and EngineV")

# ax3.scatter(data_cleaned['Mileage'],data_cleaned['Log_Price'])
# ax3.set_title("Log_Price and Mileage")
# plt.show()
# data_cleaned = data_cleaned.drop('Price', axis= 1)

# variables = data_cleaned[['Mileage','Year','EngineV']]
# vif = pd.DataFrame()
# vif['VIF'] = [variance_inflation_factor(variables,i) for i in range(variables.shape[1])]
# vif['Features'] = variables.columns

# data_no_multicollinearity = data_cleaned.drop('Year',axis= 1)
# print(data_no_multicollinearity)
# data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first =True,dtype= int)
# cols = list(data_with_dummies.columns.values)
# if 'Log_Price' in cols:
#     cols.insert(0,cols.pop(cols.index('Log_Price')))
# data_preprocessed = data_with_dummies[cols]

# targets = data_preprocessed['Log_Price']
# inputs = data_preprocessed.drop(['Log_Price'],axis=1)

# scaler  = StandardScaler()
# scaler.fit(inputs)
# scaled_inputs = scaler.transform(inputs)

# x_train, x_test, y_train,y_test = train_test_split(scaled_inputs,targets, test_size=0.2,random_state=365)

# reg = LinearRegression()
# reg.fit(x_train,y_train)
# y_hat = reg.predict(x_train)

# plt.scatter(y_hat,y_train)
# plt.xlabel("Targets (Y_train)",size=18)
# plt.ylabel("Targets (Y_prediction)",size=18)
# plt.xlim(6,13)
# plt.ylim(6,13)
# # plt.show()


# sns.displot(y_train - y_hat)
# plt.title("residual_pdf",size=18)
# plt.show()

# print(reg.score(x_train,y_train))
# print(reg.intercept_)
# print(reg.coef_)

# reg_summary = pd.DataFrame(inputs.columns.values,columns=['Features'])
# reg_summary['Weights'] = reg.coef_
# print(reg_summary)

# y_hat_test =reg.predict(x_test)

# plt.scatter(y_hat_test,y_test,alpha=.2)
# plt.xlabel("Targets (Y_test)",size=18)
# plt.ylabel("Predictions (Y_hat_test)",size=18)
# plt.xlim(6,13)
# plt.ylim(6,13)
# plt.show()


# LOGISTIC REGRESSION
pd.set_option('display.max_rows',1000)
raw_data = pd.read_csv(r'C:\Users\Noc\Desktop\line\002 2.01.Admittance.csv')
data = raw_data.copy()

# data_dummies = pd.get_dummies(data,drop_first=True,dtype= int)
# print(data_dummies)

data['Admitted']= data['Admitted'].map({'Yes':1,'No':0})
print(data)