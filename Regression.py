import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from difflib import SequenceMatcher as sc
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Functions ------------------------------------------------------------------------------------------------------------
def onehot(data, key):
    list = data[key].unique()
    for i, k in enumerate(list):
        data.insert(data.columns.get_loc(key), key + ' ' + str(i) + ' ' + str(k), np.zeros(len(data[key])))
    for i, k in enumerate(list):
        for idx, val in enumerate(data[key]):
            if val == k:
                data[key + ' ' + str(i) + ' ' + str(k)][idx] = 1
    data.drop(key, axis=1, inplace=True)


# Init -----------------------------------------------------------------------------------------------------------------
data = pd.read_csv("CarPrice.csv", index_col=0, header=0)
data_np = np.array(data)
print(data.info())

print(data.describe())
print("Highest price is: ", data.get("price").max())
print("Lowest price is: ", data.get("price").min())
print("The standard diviation of the Prices is: ", data.get("price").std())


# Car Names' Processing-------------------------------------------------------------------------------------------------
vehicle_names = [
    "alfa-romero", "audi", "bmw", "chevrolet", "dodge", "honda", "isuzu", "jaguar", "mazda", "buick", "mercury",
    "mitsubishi", "nissan", "peugeot", "plymouth", "porsche", "renault", "saab", "subaru", "toyota", "volkswagen",
    "vw", "volvo"
]

# Seperating car brands from the complete name string
for idx, name in enumerate(data["CarName"]):
    for bname in vehicle_names:
        if name.lower().find(bname) != -1:
            data["CarName"][idx+1] = bname
            break

# Similarity double-checking and correction
for idx, name in enumerate(data["CarName"]):
    if name.lower() not in vehicle_names:
        for bname in vehicle_names:
            if sc(a=name.lower()[0:name.find(" ")], b=bname).ratio() > 0.7:
                data["CarName"][idx + 1] = bname
                #print(name, idx)
                break

# Final Check
for idx, name in enumerate(data["CarName"]):
    if name.lower() not in vehicle_names:
        PassFlag = True
        print(name)

for idx, name in enumerate(data["CarName"]):
    for idx2, bname in enumerate(vehicle_names):
        if name.lower().find(bname) != -1:
            data["CarName"][idx+1] = idx2


data['CarName'] = data['CarName'].astype(int)
onehot(data, 'fueltype')
print(data[data.columns[1:4]][60:80])
data["aspiration"].replace({'std':0, 'turbo':1}, inplace=True)
data["doornumber"].replace({'two':1, 'four':0}, inplace=True)
onehot(data, 'carbody')
print(data[data.columns[5:10]][60:80])
data["drivewheel"].replace({'rwd':2, 'fwd':0, '4wd':1}, inplace=True)
data["enginelocation"].replace({'front':0, 'rear':1}, inplace=True)
onehot(data, 'enginetype')
data["cylindernumber"].replace({'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'eight':8, 'twelve':12}, inplace=True)
onehot(data, 'fuelsystem')

print(data.info())

# Standardization & Normalization --------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

scaler = StandardScaler()
d = scaler.fit_transform(data)
data = pd.DataFrame(data=d, columns=data.columns)
#
# print("\n Standardized Data:")
# print(data.head(10))

# d = normalize(data, norm="l1", axis=0)
# data = pd.DataFrame(data=d, columns=data.columns)

# print("\n Normalized Data:")
# print(data.head(10))

# Correlation matrix ---------------------------------------------------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=False, fmt='.2f', cbar=True, cmap="Blues")
#plt.matshow(data.corr())
#plt.savefig('correlation.png')
plt.show()

# splitting the data ---------------------------------------------------------------------------------------------------
x = data[data.columns[:-1]]
y = data['price']

# jointplots -----------------------------------------------------------------------------------------------------------
plt.figure(figsize=(10,10))
sns.jointplot(x=data['horsepower'], y=data['price'], kind='scatter', data=data)
plt.grid()
#plt.savefig('hp-price_jointplot.png')
#plt.show()

plt.figure(figsize=(10,10))
sns.jointplot(x=data['enginesize'], y=data['price'], kind='scatter', data=data)
plt.grid()
#plt.savefig('enginesize-price_jointplot.png')
#plt.show()

plt.figure(figsize=(10,10))
sns.jointplot(x=data['citympg'], y=data['price'], kind='scatter', data=data)
plt.grid()
#plt.savefig('citympg-price_jointplot.png')
#plt.show()

# Selecting top ten features and splitting -------------------------------------------------------------------------------------------
selector = SelectKBest(f_regression, k=10)
x_new = selector.fit_transform(x, y)
print(x.columns[selector.get_support()])
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.3, random_state=42)

# Simple Linear Regression Fitting -------------------------------------------------------------------------------------
linear = LinearRegression()
linear.fit(x_train, y_train)
print("\nSimple Linear Regression Scores")
print('train data R2 score: ' + str(linear.score(x_train, y_train)))
print('test data R2 score: ' + str(linear.score(x_test, y_test)))

# Lasso Regression -----------------------------------------------------------------------------------------------------
lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(x_train, y_train)

print("\nLasso Linear Regression Scores:")
print('train data R2 score: ' + str(lasso.score(x_train, y_train)))
print('test data R2 score: ' + str(lasso.score(x_test, y_test)))

# Ridge Regression -----------------------------------------------------------------------------------------------------
rlr = Ridge(alpha=0.1)
rlr.fit(x_train, y_train)
rlr.score(x_test, y_test)

print("\nRidge Linear Regression Scores:")
print('train data R2 score: ' + str(rlr.score(x_train, y_train)))
print('test data R2 score: ' + str(rlr.score(x_test, y_test)))

# SVR ------------------------------------------------------------------------------------------------------------------
svr = make_pipeline(StandardScaler(), SVR(kernel = "rbf", C=1.0, epsilon=0.2))
svr.fit(x_train, y_train)

print("\nSupport Vector Regression Scores:")
print('train data R2 score: ' + str(svr.score(x_train, y_train)))
print('test data R2 score: ' + str(svr.score(x_test, y_test)))

# RMSE score ----------------------------------------------------------------------------------------
linear_pred = linear.predict(x_test)
print('\nlinear RMSE score: ')
print(mean_squared_error(y_test, linear_pred))

lasso_pred = lasso.predict(x_test)
print('\nlasso RMSE score: ')
print(mean_squared_error(y_test, lasso_pred))

ridge_pred = rlr.predict(x_test)
print('\nridge RMSE score: ')
print(mean_squared_error(y_test, ridge_pred))

svr_pred = svr.predict(x_test)
print('\nsvr RMSE score: ')
print(mean_squared_error(y_test, svr_pred))
