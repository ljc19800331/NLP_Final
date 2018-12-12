# Reference
# Embedding: https://cloud.tencent.com/developer/news/217727

# The RNN implementation
# Word embedding
# Feature matrix
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re
from bs4 import BeautifulSoup
import sys
import os
os.environ['KERAS_BACKEND']='theano'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence, text
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

# Define the word embedding parameters
MAX_WORDS_IN_SEQ = 1000
EMBED_DIM = 100
VALIDATION_SPLIT = 0.2

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

data = sequence.pad_sequences(sequences, maxlen=MAX_WORDS_IN_SEQ)
labels = to_categorical(labels)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print("The shape of training data is ", x_train.shape)
print("The training data is ", x_train)
print("The current word2index is ", word2index)

input_seq = Input(shape=[MAX_WORDS_IN_SEQ, ], dtype='int32')
embed_seq = Embedding(num_words + 1,
                      EMBED_DIM,
                      embeddings_initializer = 'glorot_uniform',
                      input_length = MAX_WORDS_IN_SEQ)

sequence_input = Input(shape=(MAX_WORDS_IN_SEQ,), dtype='int32')
embedded_sequences = embed_seq(sequence_input)
l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
preds = Dense(2, activation='softmax')(l_lstm)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
print("Bidirectional LSTM")
# model.summary()

MODEL_PATH = "/home/mgs/PycharmProjects/NLP_Final/spam-detection-using-deep-learning/spam_detect_char"
cp = ModelCheckpoint(MODEL_PATH, monitor='val_acc',verbose=1,save_best_only=True)
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=5, batch_size=2,callbacks=[cp])
