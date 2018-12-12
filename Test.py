import numpy as np
import pandas as pd
import scipy as sp
import sklearn
import pickle
import matplotlib as mat
import matplotlib.pyplot as plt
import plotly.plotly as py
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn import svm

# Read the data
df = pd.read_csv('/home/mgs/PycharmProjects/NLP_Final/SpamHam_SVM/Spam.csv')
df.head()

# Data normalization -- after nomalization
minmax = MinMaxScaler()
norm = pd.DataFrame(minmax.fit_transform(df.drop(['spam'],axis=1)), columns = df.drop(['spam'], axis=1).dtypes.index)
norm.head()
df_norm = norm.join(df['spam'])
df_norm.head()
standardized_data = df.drop('spam', axis=1)
standardized_data.head()
# print('The standardized data is ', standardized_data.shape)

# Data preparation -- randomize the order of the string
shuffle_index = np.random.permutation(len(df_norm))     # Not in order
test_size = int( len(shuffle_index) * 0.3)
test_data = shuffle_index[:test_size]
train_data = shuffle_index[test_size:]
train = df_norm.iloc[train_data]
test = df_norm.iloc[test_data]
print('The training data is ', train.shape)
# print('The testing data is ', test)

# Train the model
svc = SVC(kernel='linear',gamma=1)
svc.fit(train.drop('spam',axis=1), train['spam'])
predictions = svc.predict(test.drop('spam',axis=1))

# Show the result
accuracy_score(test['spam'],predictions)
print("The final result is ", accuracy_score(test['spam'],predictions))