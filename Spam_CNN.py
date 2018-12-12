import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re
from bs4 import BeautifulSoup
import sys
import os
os.environ['KERAS_BACKEND']='theano' # Why theano why not
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.utils import to_categorical
from keras.preprocessing import sequence, text
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Conv1D, GlobalMaxPooling1D, Dropout, Dense, Input, Embedding, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd

# Define the word embedding parameters
MAX_WORDS_IN_SEQ = 1000
EMBED_DIM = 100

# Load the data from the original emails
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
print(train_data.head())

emails = train_data['processed_content'].values
labels = train_data['CLASS'].values
max_len = max(map(lambda x: len(x), emails))

print("The emails are ", emails)
print("The labels of emails are ", len(labels))
print("The maximum length is ", max_len)

# Preprocess the data -- tokenize the word lists
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(emails)
sequences = tokenizer.texts_to_sequences(emails)
word2index = tokenizer.word_index
num_words = len(word2index)

# print("The sequences are ", len(sequences))     # The number of email messages -- this is im
# print("The word2index is ", word2index)
print("The unique tokens are ", num_words)

data = sequence.pad_sequences(sequences, maxlen = MAX_WORDS_IN_SEQ, padding='post', truncating='post')
labels = to_categorical(labels)     # Binary operation to the input
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

print("The latest data is ", x_train.shape)
print("The data of the labels is ", x_train.shape)

# The input and output operation -- this is im as well
input_seq = Input(shape=[MAX_WORDS_IN_SEQ, ], dtype='int32')
embed_seq = Embedding(num_words + 1, EMBED_DIM, embeddings_initializer = 'glorot_uniform', input_length = MAX_WORDS_IN_SEQ)(input_seq)

# print("The input sequence is ", input_seq)
# print("The embedded sequence is ", embed_seq)

conv_1 = Conv1D(128, 5, activation='relu')(embed_seq)
conv_1 = MaxPooling1D(pool_size=5)(conv_1)
conv_2 = Conv1D(128, 5, activation='relu')(conv_1)
conv_2 = MaxPooling1D(pool_size=5)(conv_2)
conv_3 = Conv1D(128, 5, activation='relu')(conv_2)
conv_3 = MaxPooling1D(pool_size=35)(conv_3)
flat = Flatten()(conv_3)

# flat = Dropout(0.25)(flat)
fc1 = Dense(128, activation='relu')(flat)

# dense_1 = Dropout(0.25)(flat)
fc2 = Dense(2, activation ='softmax')(fc1)

model = Model(input_seq, fc2)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.summary()

MODEL_PATH = "/home/mgs/PycharmProjects/NLP_Final/spam-detection-using-deep-learning/spam_detect_char"

model.fit(
    x_train,
    y_train,
    batch_size = 128,
    epochs = 5,
    callbacks = [ModelCheckpoint(MODEL_PATH, save_best_only = True)],
    validation_data = [x_test, y_test]
    )