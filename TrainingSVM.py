import pickle
import xgboost as xgb
import shap
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, classification_report
import seasonYYlooder
#import seasonYYlooder

df_alg,df_nonalg,df_nonveg=seasonYYlooder.load()


xtrain_0= pd.concat([df_alg.iloc[:,45:-1],df_nonalg.iloc[:,45:-1], df_nonveg.iloc[:,45:-1]],ignore_index=True)
ytrain_0= pd.concat([df_alg.iloc[:,-1:],df_nonalg.iloc[:,-1:], df_nonveg.iloc[:,-1:]],ignore_index=True)

print(xtrain_0.info())
print(ytrain_0.info())
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(xtrain_0, ytrain_0, test_size=0.2,random_state=5)

model = SVC(kernel='linear', C=1, decision_function_shape='ovr', random_state=42)
evalset = [(xtrain, ytrain), (xtest, ytest)]
#xgbr = xgb.XGBClassifier(objective='reg:squarederror',max_depth= depth, colsample_bylevel=col ,learning_rate=lr,n_estimators= n)
model.fit(xtrain, ytrain)
print("okey")
with open("model/SVM_model_all_Y.pkl", "wb") as model_file:
  pickle.dump(model, model_file)
#from sklearn.metrics import mean_squared_error

#mse = mean_squared_error(Y, ypred)
# print("depth: %s learning_rate: %s n_estimator: %s colsample_bylevel: %.2f RMSE: %.2f" % (depth,lr,n,col,mse**(1/2.0)))
ypred = model.predict(xtest)
print(xtest)
print(ypred)
accuracy = accuracy_score(ytest, ypred)
print(f'Accuracy: {accuracy:.2f}\n')
cm = confusion_matrix(ytest, ypred)
print(cm)
precision = precision_score(ytest, ypred, average='weighted')
recall = recall_score(ytest, ypred, average='weighted')
f1 = f1_score(ytest, ypred, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
# Display classification report
print('Classification Report:\n', classification_report(y_test, y_pred))

# Plot decision boundaries
h = .02  # Step size in the mesh
x_min, x_max = 0 - 1, 1+ 1
y_min, y_max = 0 - 1, 2+ 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', marker='o', s=80)
plt.title('SVM Decision Boundaries for 3-Class Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
