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
# %%
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

pd.set_option('display.max_rows',1000)
raw_data = pd.read_csv(r'C:\Users\Noc\Desktop\line\002 2.01.Admittance.csv')
data = raw_data.copy()

# data_dummies = pd.get_dummies(data,drop_first=True,dtype= int)
# print(data_dummies)

data['Admitted']= data['Admitted'].map({'Yes':1,'No':0})
print(data)
# %%
y = data['Admitted']
x1 = data['SAT']

plt.scatter(x1,y,c ='red')
plt.xlabel('SAT',fontsize =20)
plt.ylabel('Admitted',fontsize =20)
plt.show()
# %%
x = sm.add_constant(x1)
reg_lin = sm.OLS(y,x)
results_ln = reg_lin.fit()

plt.scatter(x1,y, c='C0')
y_hat = x1 * results_ln.params[1]+results_ln.params[0]
plt.scatter(x1,y_hat, lw=2.5, c= 'C8')
plt.xlabel('SAT',fontsize =20)
plt.ylabel('Admitted',fontsize =20)
plt.show()
# %%
reg_log = sm.Logit(y,x)
result_log = reg_log.fit()
print(result_log.summary())

def f(x,b0,b1):
    return np.array(np.exp(b0+b1 * x)/(1 + np.exp(b0 + b1 * x)))

f_sorted = np.sort(f(x1,result_log.params[0],result_log.params[1]))
x_sorted  = np.sort(np.array(x1))

# plt.scatter(x1,y,c ='C0')
# plt.xlabel('SAT',fontsize =20)
# plt.ylabel('Admitted',fontsize =20)
# plt.plot(x_sorted,f_sorted, color = 'C8')

# %%
pd.set_option('display.float_format','{0:0.2f}'.format)
raw_data = pd.read_csv(r'C:\Users\Noc\Desktop\line\010 2.02.Binary-predictors.csv')
data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1,'No':0})
data['Gender'] = data['Gender'].map({'Male':0,'Female':1})
y = data['Admitted']
x1 = data[['SAT','Gender']]
x = sm.add_constant(x1)
reg_log= sm.Logit(y,x)
result_log = reg_log.fit()
result_log.summary()
# result_log.predict()
result_log.pred_table()
x
# %%
test = pd.read_csv(r'C:\Users\Noc\Desktop\line\015 2.03.Test-dataset.csv')
test['Admitted'] = test['Admitted'].map({'Yes':1,'No':0})
test['Gender'] = test['Gender'].map({'Male':0,'Female':1})
test_actual = test['Admitted']
test_data = test.drop(['Admitted'],axis =1)
test_data = sm.add_constant(test_data)

def confusion_matrix(data,actual_values,model):
    pred_values = model.predict(data)
    bins = np.array([0,0.5,1])
    cm= np.histogram2d(actual_values,pred_values, bins =bins)[0]
    accuracy = (cm[0,0] + cm[1,1]/cm.sum())
    return cm,accuracy

cm= confusion_matrix(test_data, test_actual, result_log)
cm
# %%
# CLUSTER ANALYSIS

from sklearn.cluster import KMeans

data =pd.read_csv(r'C:\Users\Noc\Desktop\line\002 3.01.Country-clusters.csv')
plt.scatter(data['Longitude'],data['Latitude'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()
# %%
x = data.iloc[:,1:3]
x
# %%
Kmeans = KMeans(3)
Kmeans.fit(x)
# %%
identified_cluster = Kmeans.fit_predict(x)
identified_cluster
data_with_cluster = data.copy()
data_with_cluster['Cluster'] = identified_cluster
print(data_with_cluster)
plt.scatter(data_with_cluster['Longitude'],data_with_cluster['Latitude'],c=data_with_cluster['Cluster'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()
# %%
data
# %%
data_mapped = data.copy()
data_mapped['Language']= data_mapped['Language'].map({'English':0,'French':1,'German':2})
data_mapped
# %%
x= data_mapped.iloc[:,1:3]
Kmeans = KMeans(2)
print(x)
Kmeans.fit(x)
# %%
identified_cluster = Kmeans.fit_predict(x)
identified_cluster
data_with_cluster = data.copy()
data_with_cluster['Cluster'] = identified_cluster
print(data_with_cluster)
plt.scatter(data_with_cluster['Longitude'],data_with_cluster['Latitude'],c=data_with_cluster['Cluster'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()
# %%
Kmeans.inertia_
# %%
wcss =[]
for i in range(1,7):
    Kmeans = KMeans(i)
    Kmeans.fit(x)
    wcss_iter = Kmeans.inertia_
    wcss.append(wcss_iter)
# %%
wcss
# %%
number_clusters = range(1,7)
plt.plot(number_clusters,wcss)
plt.title("The Elbow Method")
plt.ylabel("Within-cluster sum of Squares")
plt.xlabel("Number Of Clusters")
plt.show()
# %%
# CLUSTER MARKET SEGMENTATION

data =pd.read_csv(r'C:\Users\Noc\Desktop\line\011 3.12.Example.csv')
data
# %%
plt.scatter(data['Satisfaction'],data['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.plot()
# %%
x= data.copy()
Kmeans = KMeans(2)
Kmeans.fit(x)
# %%
cluster = x.copy()
cluster['cluster_pred']= Kmeans.fit_predict(x)
plt.scatter(cluster['Satisfaction'],cluster['Loyalty'],c=cluster['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
# %%
from sklearn import preprocessing
x_scaled = preprocessing.scale(x)
x_scaled
# %%
wcss =[]
for i in range(1,9):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
wcss
# %%
plt.plot(range(1,9), wcss)
plt.xlabel("number of Clusters")
plt.ylabel("wcss")

# %%
