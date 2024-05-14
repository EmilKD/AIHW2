import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Data Init -----------------------------------------------------
data = pd.read_csv("./diabetes.csv")
print(data.info())
print(data.describe())

# Counting Nans and computing Nan percentages -------------------
nansums = []
data_length = len(data)
nanpercentages = []
for d in data.columns:
    nansums.append(data[d].isna().sum())
    nanpercentages.append(nansums[-1]/data_length * 100)

print("\nNumber of Nan Values for each column: ")
for i in range(len(data.columns)):
    print(str(data.columns[i]) + ": " + str(nansums[i]))

print("\nPercentage of Nan Values for each column: ")
for i in range(len(data.columns)):
    print(str(data.columns[i]) + ": " + str(nanpercentages[i]))

# imputation ----------------------------------------------------
for c in data.columns:
    data[c] = data[c].fillna(data[c].mean())

# Removing Samples with Nan values ------------------------------
# Removing rows with missing values
#-data.dropna(axis=0)

#Removing columns with missing values
#data.dropna(axis=1) dropping columns removes all columns
print(data.info())


# Testing standardization (Tested almost no difference)
# scaler = StandardScaler()
# d = scaler.fit_transform(data[data.columns[:-1]])
# data[data.columns[:-1]] = pd.DataFrame(data=d, columns=data.columns[:-1])

 # Testing normalization (Tested good results)
# d = normalize(data[data.columns[:-1]], norm="l1")
# data[data.columns[:-1]] = pd.DataFrame(data=d, columns=data.columns[:-1])

# Splitting the data --------------------------------------------
x = data[data.columns[:-1]]
y = data['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

# Correlation matrix --------------------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=False, fmt='.2f', cbar=True, cmap="Blues")
plt.savefig('categorical_corr.png')
plt.show()

# Pregnancies and BMI are correlated with outcome more than the others

# Occurrences count plots ----------------------------------------
plt.figure()
sns.countplot(data, x="BMI")
plt.savefig('occurrences_bmi.png')
plt.show()

plt.figure()
sns.countplot(data, x="Pregnancies")
plt.savefig('occurrences_pregs.png')
plt.show()

# joint plots ---------------------------------------------------
plt.figure()
sns.jointplot(x=data['BMI'], y=data['Outcome'], kind='scatter', data=data)
plt.savefig('bmi_out_jointplot_scatter.png')
plt.show()

plt.figure()
sns.jointplot(x=data['BMI'], y=data['Outcome'], kind='hex', data=data)
plt.savefig('bmi_out_jointplot_hex.png')
plt.show()

plt.figure()
sns.jointplot(x=data['Pregnancies'], y=data['Outcome'], kind='scatter', data=data)
plt.savefig('pregs_out_jointplot_scatter.png')
plt.show()

plt.figure()
sns.jointplot(x=data['Pregnancies'], y=data['Outcome'], kind='hex', data=data)
plt.savefig('pregs_out_jointplot_hex.png')
plt.show()

print('\n------------------- Pre Parameter Optimization -------------------')
# Model Creation and training -----------------------------------
logistic = LogisticRegression(random_state=0, max_iter=10000).fit(x_train, y_train)
lr_train_score = logistic.score(x_train, y_train)
lr_score = logistic.score(x_test, y_test)
print('\nLR train score: ')
print(lr_train_score)
print('\nLR test score: ')
print(lr_score)

KNN = KNeighborsClassifier(n_neighbors=10)
KNN.fit(x_train, y_train)
knn_train_score = KNN.score(x_train, y_train)
knn_score = KNN.score(x_test, y_test)
print('\nKNN train score: ')
print(knn_train_score)
print('\nKNN test score: ')
print(knn_score)

DT = DecisionTreeClassifier(random_state=0)
DT.fit(x_train, y_train)
dt_train_score = DT.score(x_train, y_train)
dt_score = DT.score(x_test, y_test)
print('\nDT train score: ')
print(dt_train_score)
print('\nDT test score: ')
print(dt_score)

RF = RandomForestClassifier()
RF.fit(x_train, y_train)
rf_train_score = RF.score(x_train, y_train)
rf_score = RF.score(x_test, y_test)
print('\nRF train score:')
print(rf_train_score)
print('\nRF test score:')
print(rf_score)

SVM = SVC() #Regularization parameter. The strength of the regularization is inversely proportional to C
SVM.fit(x_train, y_train)
svm_train_score = SVM.score(x_train, y_train)
svm_score = SVM.score(x_test, y_test)
print('\nSVM train score:')
print(svm_train_score)
print('\nSVM test score:')
print(svm_score)

models = [logistic, KNN, DT, RF, SVM]
model_names = ['Logistic Regression', 'K Nearest Neighbors', 'Decision Tree', 'Random Forest', 'Support Vector Machine']
for i in range(5):
    pred = models[i].predict(x_test)
    confix = confusion_matrix(y_true=y_test, y_pred=pred)
    plt.figure(i)
    #plt.title(model_names[i])
    sns.heatmap(confix, cmap='Blues',annot=True)
    plt.savefig('cm_'+str(model_names[i])+'.png')
    plt.show()

# Parameter optimization ---------------------------------------------
# parameters = {
# 'fit_intercept':[True, False],
# 'C':[1,10, 20, 50, 100],
# 'solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}
# loreg = LogisticRegression(max_iter=10000, solver='saga', tol=0.1)
# gridsearch = GridSearchCV(loreg, parameters, cv=5).fit(x_train, y_train)
# print(gridsearch.best_params_)
# #Result: {'C': 10, 'fit_intercept': True, 'penalty': 'l2', 'solver': 'newton-cholesky'}
#
# # KNN
# gridsearch = GridSearchCV(KNeighborsClassifier(), {'n_neighbors':[2,5,10,15,20,30]}, cv=5).fit(x_train, y_train)
# print(gridsearch.best_params_)
# # Result: {'n_neighbors': 20}
#
# # Decision Tree
# parameters = {
#     'criterion':['gini', 'entropy', 'log_loss'],
#     'splitter':['best', 'random'],
#     'max_depth':[2,4,6,8,10,12],
#     'min_samples_split':[2,4,6,8,10,12]}
#
# gridsearch = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5).fit(x_train, y_train)
# print(gridsearch.best_params_)
# # Result: {'criterion': 'gini', 'max_depth': 2, 'min_samples_split': 2, 'splitter': 'best'}
#
# # Decision Tree
# parameters = {
#     'n_estimators':[100,200,300,400],
#     'criterion':['gini', 'entropy', 'log_loss'],
#     'max_depth':[2,4,6,8],
#     'min_samples_split':[2,4,6,8,10,12]}
#
# # Random Forest
# gridsearch = GridSearchCV(RandomForestClassifier(), parameters, cv=5).fit(x_train, y_train)
# print(gridsearch.best_params_)
# # Result: {'criterion': 'gini', 'max_depth': 8, 'min_samples_split': 6, 'n_estimators': 300}
#
# # SVM
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# SVM = SVC()
# gridsearch = GridSearchCV(SVM, parameters, cv=5).fit(x_train, y_train)
# print(gridsearch.best_params_)
# # Result: {'C': 1, 'kernel': 'linear'}

print('\n------------------- Post Parameter Optimization -------------------')
# Model Re-Evaluation after parameter optimization -----------------------------------
logistic = LogisticRegression(solver='newton-cholesky',random_state=0,C= 10,fit_intercept = True,penalty='l2').fit(x_train, y_train)
new_lr_train_score = logistic.score(x_train, y_train)
new_lr_score = logistic.score(x_test, y_test)
print('\nLR new train score: ')
print(new_lr_train_score)
print('\nLR new test score: ')
print(new_lr_score)
print(str(new_lr_train_score/lr_train_score * 100) + '% train score change')
print(str(new_lr_score/lr_score * 100) + '% test score change')

KNN = KNeighborsClassifier(n_neighbors=20)
KNN.fit(x_train, y_train)
new_knn_train_score = KNN.score(x_train, y_train)
new_knn_score = KNN.score(x_test, y_test)
print('\nKNN new train score: ')
print(new_knn_train_score)
print('\nKNN new test score: ')
print(new_knn_score)
print(str(new_knn_train_score/knn_train_score * 100) + '% train score change')
print(str(new_knn_score/knn_score * 100) + '% test score change')

DT = DecisionTreeClassifier(random_state=0, criterion='gini',max_depth=2,min_samples_split=2,splitter='best')
DT.fit(x_train, y_train)
new_dt_train_score = DT.score(x_train, y_train)
new_dt_score = DT.score(x_test, y_test)
print('\nDT new train score: ')
print(new_dt_train_score)
print('\nDT new test score: ')
print(new_dt_score)
print(str(new_dt_train_score/dt_train_score * 100) + '% train score change')
print(str(new_dt_score/dt_score * 100) + '% test score change')

RF = RandomForestClassifier(criterion='gini',max_depth=8,min_samples_split=6,n_estimators=300)
RF.fit(x_train, y_train)
new_rf_train_score = RF.score(x_train, y_train)
new_rf_score = RF.score(x_test, y_test)
print('\nRF new train score:')
print(new_rf_train_score)
print('\nRF new test score:')
print(new_rf_score)
print(str(new_rf_train_score/rf_train_score * 100) + '% train score change')
print(str(new_rf_score/rf_score * 100) + '% test score change')

SVM = SVC(kernel='linear', C=1) #Regularization parameter. The strength of the regularization is inversely proportional to C
SVM.fit(x_train, y_train)
new_svm_train_score = SVM.score(x_train, y_train)
new_svm_score = SVM.score(x_test, y_test)
print('\nSVM new train score:')
print(new_svm_train_score)
print('\nSVM new test score:')
print(new_svm_score)
print(str(new_svm_train_score/svm_train_score * 100) + '% train score change')
print(str(new_svm_score/svm_score * 100) + '% test score change')

models = [logistic, KNN, DT, RF, SVM]
model_names = ['Logistic Regression', 'K Nearest Neighbors', 'Decision Tree', 'Random Forest', 'Support Vector Machine']
for i in range(5):
    pred = models[i].predict(x_test)
    confix = confusion_matrix(y_true=y_test, y_pred=pred)
    plt.figure(i)
    #plt.title('optimized' + str(model_names[i]))
    sns.heatmap(confix, cmap='Blues',annot=True)
    plt.savefig('cm_optimized'+str(model_names[i])+'.png')
    plt.show()

# KNN and RF have been overfitted
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.15, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.21428, random_state=42)

KNN.fit(x_train, y_train)
valid_knn_score = KNN.score(x_val, y_val)
print('\nKNN validation score: ')
print(valid_knn_score)
print(str(valid_knn_score/new_knn_score * 100) + '% score change with validation data')

RF.fit(x_train, y_train)
valid_rf_score = RF.score(x_val, y_val)
print('\nRF validation score: ')
print(valid_rf_score)
print(str(valid_rf_score/new_rf_score * 100) + '% score change with validation data')