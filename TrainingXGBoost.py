
from sklearn import  metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from matplotlib.pylab import rcParams
import pickle
import xgboost as xgb
import shap
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from xgboost import cv
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt

import seasonXloader
import seasonYloader
df_alg,df_nonalg,df_nonveg=seasonYloader.load()
#df_alg0,df_nonalg0,df_nonveg0=seasonXloader.load()
#df_alg1,df_nonalg1,df_nonveg1=seasonYloader.load()
#df_alg=pd.concat([df_alg0,df_alg1],ignore_index=True)
#df_nonalg=pd.concat([df_nonalg0,df_nonalg1],ignore_index=True)
#df_nonveg=pd.concat([df_nonveg0,df_nonveg1],ignore_index=True)
xtrain= pd.concat([df_alg.iloc[:,45:-1],df_nonalg.iloc[:,45:-1], df_nonveg.iloc[:,45:-1]],ignore_index=True)
ytrain= pd.concat([df_alg.iloc[:,-1:],df_nonalg.iloc[:,-1:], df_nonveg.iloc[:,-1:]],ignore_index=True)

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
for n in [600]:
  for lr in [0.5]:
    for depth in [3]:
      for col in [0.7]:
        xgbr = xgb.XGBClassifier(
          objective='multi:softprob',
          num_class=3,
          max_depth=depth,
          learning_rate=lr,
          colsample_bytree=col,
          n_estimators=n
        )
        evalset = [(xtrain, ytrain), (xtest, ytest)]
        #xgbr = xgb.XGBClassifier(objective='reg:squarederror',max_depth= depth, colsample_bylevel=col ,learning_rate=lr,n_estimators= n)
        xgbr.fit(xtrain, ytrain, eval_metric='mlogloss', eval_set=evalset)
        with open("model/xgboost_model_seasonY.pkl", "wb") as model_file:
          pickle.dump(xgbr, model_file)
        #from sklearn.metrics import mean_squared_error

        #mse = mean_squared_error(Y, ypred)
       # print("depth: %s learning_rate: %s n_estimator: %s colsample_bylevel: %.2f RMSE: %.2f" % (depth,lr,n,col,mse**(1/2.0)))
        ypred = xgbr.predict(xtest)
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
        results = xgbr.evals_result()
        plt.plot(results['validation_0']['mlogloss'], label='train')
        plt.plot(results['validation_1']['mlogloss'], label='test')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()

        print("Accuracy:", accuracy)
        print("depth: %s learning_rate: %s n_estimator: %s colsample_bylevel: %.2f accuracy: %f" % (depth, lr, n, col, accuracy))
        explainer = shap.Explainer(xgbr)
        shap_values = explainer.shap_values(xtrain)

        # Summarize the effects of all the features
        shap.summary_plot(shap_values, xtrain)
        params = {"objective": "multi:softprob", 'colsample_bytree': 0.7, 'learning_rate': 0.1,
                  'max_depth': 3, 'alpha': 10,'num_class': 3}
        xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=5,
                    num_boost_round=100, early_stopping_rounds=10, metrics="mlogloss", as_pandas=True, seed=123)

        print(xgb_cv.head())
