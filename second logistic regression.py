import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv('Bank_data.csv')

data = raw_data.drop('Unnamed: 0', axis=1)
data['y'] = data['y'].map({'yes':1, 'no':0})
data.describe()

#Variables
y = data['y']
x1 = data['duration']

#Regression
x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()
results_log.summary()

results_log.predict()

np.array(data['y'])

#Confusion matrix
cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0:'Actual 0', 1:'Actual 1'})

#Multivariate logistic model
estimators = ['interest_rate', 'march', 'credit', 'previous', 'duration']

X1 = data[estimators]

X = sm.add_constant(X1)
reg_logit = sm.Logit(y, X)
results_logit = reg_logit.fit()
print(results_logit.summary())

#Confusion Matrix
def confusion_matrix(data, actual_values, model):
    pred_values = model.predict(data)
    bins = np.array([0,0.5,1])
    cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
    accuracy = (cm[0,0]+cm[1,1])/cm.sum()
    return cm, accuracy

print(confusion_matrix(X,y,results_logit))

#Test
test = pd.read_csv('Bank_data_testing.csv')
test = test.drop('Unnamed: 0', axis=1)
test['y'] = test['y'].map({'yes':1, 'no':0})

test_x = sm.add_constant(test[estimators])
test_y = test['y']

print(confusion_matrix(test_x, test_y, results_logit))