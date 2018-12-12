# The code for SVM
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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read the data -- try with three dataset
data_1 = '/home/mgs/PycharmProjects/NLP_Final/UDACITY-MLND-Spam-Detection/SMSSpamCollection'
df = pd.read_table(data_1, names=['label', 'sms_message'])
# print('The head of the dataset is ', df.head())

# Convert the ham and spam to label -- ham:0, spam:1 -- this is im
df.loc[: , 'label'] = df.label.map({'ham' : 0, 'spam' : 1})
# print(df.shape)
# print(df.head())

# Training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(df['sms_message'], df['label'], random_state = 1)
# print('The training data is ', X_train)
# print('Number of rows in the total set: {}'.format(df.shape[0]))
# print('Number of rows in the training set: {}'.format(X_train.shape[0]))
# print('Number of rows in the test set: {}'.format(X_test.shape[0]))

# BOW to the dataset -- bag of words
# Define a new count vectorizer
count_vector = CountVectorizer()

# Fit the training data and return the matrix -- use .toarray() to see the matrix
training_data = count_vector.fit_transform(X_train)
# print('The training data of the first operation is ', training_data.toarray())

# Transform testing data
testing_data = count_vector.transform(X_test)
# print('The testing data of the second operation is ', testing_data)

# Bayes Theorem implementation -- apply the bayes transform
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, Y_train)

# Apply the prediction
predictions_NB = naive_bayes.predict(testing_data)
# print("The prediction result is ", predictions) # The output vector labels

# The SVM operator -- feature extractions
svc = SVC(kernel='linear', gamma=1)
svc.fit(training_data, Y_train)
predictions_SVM = svc.predict(testing_data)

# print("The output prediction of SVM is ", predictions_SVM)
np.savetxt('svm.txt', predictions_SVM)
idx = np.where(predictions_SVM == 1)

# print('The index is ', idx)
# print('The testing y label is ', Y_test)
# print('Accuracy score: {}'.format(accuracy_score(Y_test, predictions_SVM)))
# print('Precision score: {}'.format(precision_score(Y_test, predictions_SVM)))
# print('Recall score: {}'.format(recall_score(Y_test, predictions_SVM)))
# print('F1 score: {}'.format(f1_score(Y_test, predictions_SVM)))

# Random forest -- problem -- how to adjust the parameter -- this is im
clf = RandomForestClassifier()
clf.fit(training_data, Y_train)
predictions_RF = clf.predict(testing_data)
idx = np.where(predictions_RF == 1)

print('The index is ', idx)
# print('The testing y label is ', Y_test)
print('Accuracy score: {}'.format(accuracy_score(Y_test, predictions_RF)))
print('Precision score: {}'.format(precision_score(Y_test, predictions_RF)))
print('Recall score: {}'.format(recall_score(Y_test, predictions_RF)))
print('F1 score: {}'.format(f1_score(Y_test, predictions_RF)))