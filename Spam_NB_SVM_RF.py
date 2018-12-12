## import statements ##
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as ms
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# % matplotlib inline

# Load the data
train_data = []
data_1 = '/home/mgs/PycharmProjects/NLP_Final/Youtube-Comments-Spam-Detection/Youtube01-Psy.csv'
data_2 = '/home/mgs/PycharmProjects/NLP_Final/Youtube-Comments-Spam-Detection/Youtube02-KatyPerry.csv'
data_3 = '/home/mgs/PycharmProjects/NLP_Final/Youtube-Comments-Spam-Detection/Youtube03-LMFAO.csv'
data_4 = '/home/mgs/PycharmProjects/NLP_Final/Youtube-Comments-Spam-Detection/Youtube04-Eminem.csv'
data_5 = '/home/mgs/PycharmProjects/NLP_Final/Youtube-Comments-Spam-Detection/Youtube05-Shakira.csv'
data_files = [data_1, data_2, data_3, data_4, data_5]

# Train the original data files
for file in data_files:
    data = pd.read_csv(file)
    train_data.append(data)
train_data = pd.concat(train_data)
train_data.info()
# print('The shape of the data is ', train_data.shape)
# print("The data is shown in ", train_data)drop_fectures(['COMMENT_ID','AUTHOR','DATE'],train_data)

# Drop the feature
def drop_fectures(features,data):

    data.drop(features, axis=1, inplace = True)

def process_content(content):

    return " ".join(re.findall("[A-Za-z]+", content.lower()))

drop_fectures(['COMMENT_ID','AUTHOR','DATE'], train_data)
# print("The content in the training data is ", train_data.shape)

# Lower case processing
train_data['processed_content'] = train_data['CONTENT'].apply(process_content)
drop_fectures(['CONTENT'], train_data)
# print(train_data.head())

x_train, x_test, y_train, y_test = train_test_split(train_data['processed_content'],train_data['CLASS'],test_size=0.2,random_state=57)

# Create the CNN feature map
count_vect = CountVectorizer(stop_words='english')
x_train_counts = count_vect.fit_transform(x_train)
# print("The current count vector is ", x_train_counts.toarray().shape)

# TF-IDF operation
# Training data
tranformer = TfidfTransformer()
x_train_tfidf = tranformer.fit_transform(x_train_counts)
print("The tf idf aftere operation is ", x_train_tfidf.shape)
print("The tf idf data is ", x_train_tfidf)
# Testing data
x_test_counts = count_vect.transform(x_test)
x_test_tfidf = tranformer.transform(x_test_counts)

# Model selection -- This is im as well -- logistic regression
model = LogisticRegression()        # This is im
model.fit(x_train_tfidf, y_train)   #
predictions = model.predict(x_test_tfidf)
confusion_matrix(y_test, predictions)
# print(classification_report(y_test, predictions))



