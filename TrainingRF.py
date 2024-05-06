
from sklearn import  metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from matplotlib.pylab import rcParams
import pickle
import xgboost as xgb
import shap
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from xgboost import cv
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import seasonXloader
import seasonYloader

df_alg,df_nonalg,df_nonveg=seasonYloader.load()


xtrain= pd.concat([df_alg.iloc[:,45:-1],df_nonalg.iloc[:,45:-1], df_nonveg.iloc[:,45:-1]],ignore_index=True)
#xtrain= pd.concat([df_alg.iloc[:,2:7],df_nonalg.iloc[:,2:7], df_nonveg.iloc[:,2:7]],ignore_index=True)
#scaler = MinMaxScaler()
#xtrain= pd.DataFrame(scaler.fit_transform(xtrain), columns=xtrain.columns)
ytrain= pd.concat([df_alg.iloc[:,-1:],df_nonalg.iloc[:,-1:], df_nonveg.iloc[:,-1:]],ignore_index=True)

data_dmatrix = xgb.DMatrix(data=xtrain, label=ytrain)

print(xtrain.info())
print(xtrain.describe())
print("............................")
print("............................")
print("............................")
print("............................")
print("............................")

print(ytrain.info())
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=0.2,random_state=5)
train_accuracy = []
test_accuracy = []

# Define a range of number of trees to try
num_trees_range = range(30, 31)
evalset = [(xtrain, ytrain), (xtest, ytest)]
X_train=xtrain
y_train=np.ravel(ytrain)
X_test=xtest
y_test=np.ravel(ytest)
for num_trees in num_trees_range:
    print(num_trees)
    clf = RandomForestClassifier(n_estimators=(num_trees), random_state=42)
    clf.fit(X_train, y_train)

    # Predictions on training and test set
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # Calculate accuracy on training and test set
    train_accuracy.append(accuracy_score(y_train, y_train_pred))
    test_accuracy.append(accuracy_score(y_test, y_test_pred))

# Plot the learning curve
plt.plot(num_trees_range, train_accuracy, label='Train Accuracy')
plt.plot(num_trees_range, test_accuracy, label='Test Accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Learning Curve of Random Forest')
plt.legend()
plt.legend()
plt.savefig('ACC_plot_RF.png')
plt.show()
#xgbr = xgb.XGBClassifier(objective='reg:squarederror',max_depth= depth, colsample_bylevel=col ,learning_rate=lr,n_estimators= n)
RF.fit(xtrain, ytrain)
with open("model/RF_model_all_Y.pkl", "wb") as model_file:
  pickle.dump(RF, model_file)
#from sklearn.metrics import mean_squared_error

#mse = mean_squared_error(Y, ypred)
# print("depth: %s learning_rate: %s n_estimator: %s colsample_bylevel: %.2f RMSE: %.2f" % (depth,lr,n,col,mse**(1/2.0)))
ypred = RF.predict(xtest)
print(xtest)
print(ypred)
accuracy = accuracy_score(ytest, ypred)
cm = confusion_matrix(ytest, ypred)
print(cm)
precision = precision_score(ytest, ypred, average='weighted')
recall = recall_score(ytest, ypred, average='weighted')
f1 = f1_score(ytest, ypred, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
cv_scores = cross_val_score(RF, xtest, ytest, cv=5, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())
# show the legend


print("Accuracy:", accuracy)
print("depth: %s learning_rate: %s n_estimator: %s colsample_bylevel: %.2f accuracy: %f" % (depth, lr, n, col, accuracy))
explainer = shap.Explainer(RF)
shap_values = explainer.shap_values(xtrain)

# Summarize the effects of all the features
shap.summary_plot(shap_values, xtrain)
params = {"objective": "multi:softprob", 'colsample_bytree': 0.7, 'learning_rate': 0.1,
          'max_depth': 3, 'alpha': 10,'num_class': 3}
xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=5,
            num_boost_round=100, early_stopping_rounds=10, metrics="mlogloss", as_pandas=True, seed=123)

print(xgb_cv.head())
