import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv('Databases/Example_bank_data.csv')

data = raw_data.copy()
data['y'] = data['y'].map({'yes':1 , 'no':0})
data = data.drop('Unnamed: 0', axis=1)

y = data['y']
x1 = data['duration']

x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()

np.set_printoptions(formatter={'float': lambda x: '{0:0.2f}'.format(x)})
results_log.predict()

np.array(data['y'])

##Confusion Matrix
cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0:'Actual 0', 1: 'Actual 1'})
print(cm_df)

##Calculate accuracy of the model
cm = np.array(cm_df)
accuracy_train = (cm[0,0]+cm[1,1])/cm.sum()
print(accuracy_train)
