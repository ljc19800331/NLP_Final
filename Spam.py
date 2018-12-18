'''
# All the codes and data refer to the other github repository
# This code is only used for education and not for other application -- not for other purpose
# Referece:
1. https://github.com/MoAbd/Spam-detection/blob/master/Spam%20detection.ipynb
2. https://cloud.tencent.com/developer/news/217727
# Naiev Bayes technique
# For spam classification
# Ref:
# Naiev Bayesian: https://towardsdatascience.com/spam-classifier-in-python-from-scratch-27a98ddd8e73
                + https://github.com/agarwalgaurav811/Spam-classifier
# CNN + HAN + RNN:
            + https://medium.com/jatana/report-on-text-classification-using-cnn-rnn-han-f0e887214d5f
            + https://github.com/jatana-research/Text-Classification (this is im)
            + https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
            + https://medium.com/emergent-future/spam-detection-using-neural-networks-in-python-9b2b2a062272
# CNN for text classification: http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
# CNN: https://github.com/dennybritz/cnn-text-classification-tf + http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

# Comprehensive guide: https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
# Machine learning based method:
# Feature selection:
# SVM: Implement by myself
# Why CNN
# Translation system from scratch: https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/

# Goal: Spam detection with different classifiers
# Title: Comparison of different classifiers on spam identification
# Data set: This is a good idea
# Result 1: Testing and training accuracy (accuracy and epoches)
# Result 2: ROC? Maybe
# Result 3: Final accuracy
# Before and after TF-IDF operation
'''

# Spam detection
import pandas as pd
import re
from keras.models import Sequential
from keras.preprocessing import sequence, text
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import glob
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.sparse import csr_matrix
from sklearn.preprocessing import Binarizer
from keras.layers import Conv1D, GlobalMaxPooling1D, Dropout, Dense, Input, Embedding, MaxPooling1D, Flatten
from sklearn import preprocessing
from keras.models import Model, load_model
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
import matplotlib.pyplot as plt
import time

# Algorithm:
# 1. Training method
# 2, Testing method

class SPAM:

    def __init__(self):
        self.A = 1
        self.B = 1
        self.MAX_WORDS_IN_SEQ = 1000
        self.EMBED_DIM = 100
        self.test_size = 0.2

    def Data_0(self, flag_deep = 0):

        # Goal: Convert raw data to the word feature vector
        PATH = "/home/mgs/PycharmProjects/NLP_Final/"
        data = []
        for verdict in ['spam', 'not_spam']:
            for files in glob.glob(PATH + verdict + "/*")[:500]:
                is_spam = True if verdict == 'spam' else False
                with open(files, "r", encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        # The title begins with "subject" -- this is im
                        if line.startswith("Subject:"):
                            subject = re.sub("^Subject: ", "", line).strip()
                            data.append((subject, is_spam))

        # Collect the spam and non spam details and calculate the labels
        emails = []
        label = np.zeros((len(data), 1))
        for idx, (message, verdict) in enumerate(data):
            emails.append(message)
            if verdict == True:     # This is spam
                label[idx] = 1
            elif verdict == False:  # This is not spam
                label[idx] = 0

        # Post processing of the data
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(emails)
        sequences = tokenizer.texts_to_sequences(emails)
        word2index = tokenizer.word_index
        num_words = len(word2index)

        # Split the data
        count_vector = CountVectorizer()
        data_unpad = count_vector.fit_transform(emails)
        if flag_deep == 1:      # deep learning
            label_cate = to_categorical(label)
        elif flag_deep == 0:    # non deep learning
            label_cate = label
        x_train_unpad, x_test_unpad, y_train_unpad, y_test_unpad = train_test_split(data_unpad, label_cate, test_size=self.test_size)
        x_train_unpad = x_train_unpad.toarray()
        x_train_unpad = x_train_unpad.astype(np.float32)
        x_test_unpad = x_test_unpad.toarray()
        x_test_unpad = x_test_unpad.astype(np.float32)
        data_unpad = [x_train_unpad, x_test_unpad, y_train_unpad, y_test_unpad]

        # Pad the data set
        # self.MAX_WORDS_IN_SEQ = x_train_unpad.shape[1]
        # print(self.MAX_WORDS_IN_SEQ)
        data = sequence.pad_sequences(sequences, maxlen=self.MAX_WORDS_IN_SEQ, padding='post', truncating='post')
        _, _, y_train_vec, y_test_vec = train_test_split(data, label, test_size=self.test_size)
        labels = to_categorical(label)
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=self.test_size)
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        data_use = [x_train, x_test, y_train, y_test]

        return data_use, data_unpad, y_train_vec, y_test_vec, num_words

    def Data_1(self, flag_deep = 0):

        # The first data set
        train_data = []
        data_1 = '/home/mgs/PycharmProjects/NLP_Final/Youtube-Comments-Spam-Detection/Youtube01-Psy.csv'
        data_2 = '/home/mgs/PycharmProjects/NLP_Final/Youtube-Comments-Spam-Detection/Youtube02-KatyPerry.csv'
        data_3 = '/home/mgs/PycharmProjects/NLP_Final/Youtube-Comments-Spam-Detection/Youtube03-LMFAO.csv'
        data_4 = '/home/mgs/PycharmProjects/NLP_Final/Youtube-Comments-Spam-Detection/Youtube04-Eminem.csv'
        data_5 = '/home/mgs/PycharmProjects/NLP_Final/Youtube-Comments-Spam-Detection/Youtube05-Shakira.csv'
        data_files = [data_1, data_2, data_3, data_4, data_5]
        for file in data_files:
            data = pd.read_csv(file)
            train_data.append(data)
        train_data = pd.concat(train_data)
        train_data.info()

        # Get the target features
        def drop_fectures(features, data):
            data.drop(features, axis=1, inplace = True)
        def process_content(content):
            return " ".join(re.findall("[A-Za-z]+", content.lower()))
        drop_fectures(['COMMENT_ID', 'AUTHOR', 'DATE'], train_data)
        train_data['processed_content'] = train_data['CONTENT'].apply(process_content)
        drop_fectures(['CONTENT'], train_data)
        emails = train_data['processed_content'].values
        labels = train_data['CLASS'].values
        max_len = max(map(lambda x: len(x), emails))

        # Case 1: For ANN
        count_vector = CountVectorizer()
        data_unpad = count_vector.fit_transform(emails)

        ''' Identify if this is used for ANN '''
        if flag_deep == 1:  # deep learning
            label_cate = to_categorical(labels)
        elif flag_deep == 0:    # non deep learning
            label_cate = labels

        x_train_unpad, x_test_unpad, y_train_unpad, y_test_unpad = train_test_split(data_unpad, label_cate, test_size = self.test_size)
        x_train_unpad = x_train_unpad.toarray()
        x_train_unpad = x_train_unpad.astype(np.float32)
        x_test_unpad = x_test_unpad.toarray()
        x_test_unpad = x_test_unpad.astype(np.float32)
        data_unpad = [x_train_unpad, x_test_unpad, y_train_unpad, y_test_unpad]

        # Case 2: For CNN or RNN
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(emails)
        sequences = tokenizer.texts_to_sequences(emails)
        word2index = tokenizer.word_index
        num_words = len(word2index)

        # self.MAX_WORDS_IN_SEQ = x_train_unpad.shape[1]
        # print(self.MAX_WORDS_IN_SEQ)
        # Pad the sequences in the same length -- try to change the parameter of MAX_WORDS_IN_SEQ
        data = sequence.pad_sequences(sequences, maxlen = self.MAX_WORDS_IN_SEQ, padding='post', truncating='post')
        # Only the one-label vector -- must before categorical
        _, _, y_train_vec, y_test_vec = train_test_split(data, labels, test_size=self.test_size)
        labels = to_categorical(labels)
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = self.test_size)
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        data_use = [x_train, x_test, y_train, y_test]

        return data_use, data_unpad, y_train_vec, y_test_vec, num_words

    # def Data_2(self):
    #
    #     # Load the data
    #     x_train = csv_to_numpy_array("/home/mgs/PycharmProjects/NLP_Final/Spam-detection-ANN/data/trainX.csv", delimiter=",")
    #     y_train = csv_to_numpy_array("/home/mgs/PycharmProjects/NLP_Final/Spam-detection-ANN/data/trainY.csv", delimiter=",")
    #     x_test = csv_to_numpy_array("/home/mgs/PycharmProjects/NLP_Final/Spam-detection-ANN/data/testX.csv", delimiter=",")
    #     y_test = csv_to_numpy_array("/home/mgs/PycharmProjects/NLP_Final/Spam-detection-ANN/data/testY.csv", delimiter=",")
    #
    #     # Convert the Cate data to the label type
    #     y_train_vec = CateDecode(y_train)
    #     y_test_vec = CateDecode(y_test)
    #
    #     # self.MAX_WORDS_IN_SEQ = x_train.shape[1]
    #     # print(self.MAX_WORDS_IN_SEQ)
    #     data_train = sequence.pad_sequences(x_train, maxlen = self.MAX_WORDS_IN_SEQ, padding='post', truncating='post')
    #     data_test = sequence.pad_sequences(x_test, maxlen = self.MAX_WORDS_IN_SEQ, padding='post', truncating='post')
    #
    #     num_words = x_train.shape[0]
    #
    #     return x_train, y_train, x_test, y_test, y_train_vec, y_test_vec, num_words

    def Data_3(self, flag_deep = 0):

        data = '/home/mgs/PycharmProjects/NLP_Final/UDACITY-MLND-Spam-Detection/SMSSpamCollection'
        df = pd.read_table(data, names=['label', 'sms_message'])
        df.loc[:, 'label'] = df.label.map({'ham': 0, 'spam': 1})

        emails = df['sms_message'].values
        label = df['label'].values

        # Token the words
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(emails)
        sequences = tokenizer.texts_to_sequences(emails)
        word2index = tokenizer.word_index
        # print("The sequences is ", len(sequences))
        num_words = len(word2index)

        # Case 1 for the original data
        count_vector = CountVectorizer()
        data_unpad = count_vector.fit_transform(emails)
        if flag_deep == 1:      # deep learning
            label_cate = to_categorical(label)
        elif flag_deep == 0:    # non deep learning
            label_cate = label
        x_train_unpad, x_test_unpad, y_train_unpad, y_test_unpad = train_test_split(data_unpad, label_cate, test_size = self.test_size)
        x_train_unpad = x_train_unpad.toarray()
        x_train_unpad = x_train_unpad.astype(np.float32)
        x_test_unpad = x_test_unpad.toarray()
        x_test_unpad = x_test_unpad.astype(np.float32)
        data_unpad = [x_train_unpad, x_test_unpad, y_train_unpad, y_test_unpad]

        # Pad the sequences
        # self.MAX_WORDS_IN_SEQ = x_train_unpad.shape[1]
        # print(self.MAX_WORDS_IN_SEQ)
        data = sequence.pad_sequences(sequences, maxlen = self.MAX_WORDS_IN_SEQ, padding = 'post', truncating = 'post')
        # print("The data is ", data.shape)
        _, _, y_train_vec, y_test_vec = train_test_split(data, label, test_size = self.test_size)
        label = to_categorical(label)
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = self.test_size)
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        data_use = [x_train, x_test, y_train, y_test]

        return data_use, data_unpad, y_train_vec, y_test_vec, num_words

    def Test_1(self):

        # Data 0
        # data_use, data_unpad, y_train_vec, y_test_vec, num_words = self.Data_0(flag_deep = 0)

        # Data 1
        # data_use, data_unpad, y_train_vec, y_test_vec, num_words = self.Data_1(flag_deep = 0)

        # Data 2 -- not to use this dataset
        # x_train, y_train, x_test, y_test, y_train_vec, y_test_vec = self.Data_2()

        # Data 3
        data_use, data_unpad, y_train_vec, y_test_vec, num_words = self.Data_3(flag_deep=0)

        # Define the data
        x_train = data_unpad[0]
        x_test = data_unpad[1]
        y_train = data_unpad[2]
        y_test = data_unpad[3]

        # print("The data size is ", len(x_train) + len(x_test))
        # print("The training data size is ", len(x_train))
        # print("The testing data size is ", len(x_test))
        # print("The number of words are ", num_words)

        # NB
        print("The result of NB is ")
        naive_bayes = MultinomialNB()
        naive_bayes.fit(x_train, y_train)
        predictions_NB = naive_bayes.predict(x_test)
        ShowScore(y_test, predictions_NB)

        # SVM
        print("The result of SVM is ")
        svc = SVC(kernel = 'linear', gamma = 1)
        svc.fit(x_train, y_train)
        predictions_SVM = svc.predict(x_test)
        ShowScore(y_test, predictions_SVM)

        # Random forest
        print("The result of random forest is ")
        clf = RandomForestClassifier()
        clf.fit(x_train, y_train)
        predictions_RF = clf.predict(x_test)
        ShowScore(y_test, predictions_RF)

        return y_test, predictions_NB, predictions_SVM, predictions_RF

    def Test_2(self):

        # Data 0 -- Randomize the data
        # data_use, data_unpad, y_train_vec, y_test_vec, num_words = self.Data_0(flag_deep = 1)

        # Data 1
        # data_use, data_unpad, y_train_vec, y_test_vec, num_words = self.Data_1(flag_deep=1)

        # Data 2 -- not working well
        # x_train, y_train, x_test, y_test, y_train_vec, y_test_vec, num_words = self.Data_2()

        # Data 3
        data_use, data_unpad, y_train_vec, y_test_vec, num_words = self.Data_3(flag_deep = 1)

        # Define the data
        x_train = data_use[0]
        x_test = data_use[1]
        y_train = data_use[2]
        y_test = data_use[3]

        # x_train = data_unpad[0]
        # x_test = data_unpad[1]
        # y_train = data_unpad[2]
        # y_test = data_unpad[3]

        # ANN keras
        self.ANN_keras(x_train, y_train, x_test, y_test)

        # ANN
        # self.ANN(x_train, y_train, x_test, y_test)

        # CNN
        # self.CNN(x_train, y_train, x_test, y_test, num_words)

        # RNN-LSTM
        # self.RNN(x_train, y_train, x_test, y_test, num_words)

    def ANN_keras(self, x_train, y_train, x_test, y_test):

        # ANN implementation with Keras
        # Design the model -- model the problem
        num_hidden_nodes1 = 32
        num_hidden_nodes2 = 16
        num_hidden_nodes3 = 8

        keep_prob = 0.5
        num_features = x_train.shape[1]
        # print(num_features)

        # create model
        model = Sequential()
        model.add(Dense(num_hidden_nodes1, input_dim=num_features, activation='relu'))
        model.add(Dense(num_hidden_nodes2, activation='relu'))
        model.add(Dense(num_hidden_nodes3, activation='relu'))
        model.add(Dense(2, activation='sigmoid'))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        history = model.fit(x_train, y_train, epochs=200, batch_size=256, validation_data=[x_test, y_test])

        # evaluate the model
        scores = model.evaluate(x_test, y_test)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        # Prediction
        predictions = model.predict(x_test)
        rounded = [round(x[0]) for x in predictions]
        print(rounded)

        # Show the figure
        fig1 = plt.figure()
        plt.plot(history.history['loss'], 'r', linewidth=3.0)
        plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
        plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title('Loss Curves :CNN', fontsize=16)
        fig1.savefig('loss_cnn.png')
        plt.show()

        fig2 = plt.figure()
        plt.plot(history.history['acc'], 'r', linewidth=3.0)
        plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Accuracy Curves : CNN', fontsize=16)
        fig2.savefig('accuracy_cnn.png')
        plt.show()

    def ANN(self, trainX, trainY, testX, testY):

        print("The training data is ", trainX.shape)

        # Basic ANN
        num_hidden_nodes1 = 1500    # 2000
        num_hidden_nodes2 = 800
        num_hidden_nodes3 = 256
        keep_prob = 0.5

        # numFeatures = the number of words extracted from each email
        numFeatures = trainX.shape[1]
        print("The number of words extracted from each email ", numFeatures)

        # numLabels = number of classes we are predicting (here just 2: Spam or Ham)
        numLabels = trainY.shape[1]

        graph = tf.Graph()

        with graph.as_default():

            tf_train_dataset = tf.constant(trainX)
            tf_train_labels = tf.constant(trainY)
            tf_test_dataset = tf.constant(testX)

            # Single mail input.
            tf_mail = tf.placeholder(tf.float32, shape=(1, numFeatures))

            # Variables.
            weights1 = tf.Variable(
                tf.truncated_normal([numFeatures, num_hidden_nodes1], stddev=np.sqrt(2.0 / (numFeatures))), name="v1")

            biases1 = tf.Variable(tf.zeros([num_hidden_nodes1]), name="v2")

            weights2 = tf.Variable(
                tf.truncated_normal([num_hidden_nodes1, num_hidden_nodes2], stddev=np.sqrt(2.0 / num_hidden_nodes1)),
                name="v3")

            biases2 = tf.Variable(tf.zeros([num_hidden_nodes2]), name="v4")

            weights3 = tf.Variable(
                tf.truncated_normal([num_hidden_nodes2, num_hidden_nodes3], stddev=np.sqrt(2.0 / num_hidden_nodes2)),
                name="v5")

            biases3 = tf.Variable(tf.zeros([num_hidden_nodes3]), name="v6")

            weights4 = tf.Variable(
                tf.truncated_normal([num_hidden_nodes3, numLabels], stddev=np.sqrt(2.0 / num_hidden_nodes3)), name="v7")

            biases4 = tf.Variable(tf.zeros([numLabels]), name="v8")

            # Add ops to save and restore all the variables. -- im
            saver = tf.train.Saver()

            # Training computation.
            layer1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
            drop1 = tf.nn.dropout(layer1_train, keep_prob)
            layer2_train = tf.nn.relu(tf.matmul(drop1, weights2) + biases2)
            drop2 = tf.nn.dropout(layer2_train, keep_prob)
            layer3_train = tf.nn.relu(tf.matmul(drop2, weights3) + biases3)
            drop3 = tf.nn.dropout(layer3_train, keep_prob)
            logits = tf.matmul(drop3, weights4) + biases4
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

            # Optimizer.
            optimizer = tf.train.AdamOptimizer(learning_rate = 0.1,
                                               beta1 = 0.9, beta2 = 0.999,
                                               epsilon = 1e-08).minimize(loss)

            # Predictions for the training, test data, and single mail.
            train_prediction = tf.nn.sigmoid(logits)

            layer1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
            layer2_test = tf.nn.relu(tf.matmul(layer1_test, weights2) + biases2)
            layer3_test = tf.nn.relu(tf.matmul(layer2_test, weights3) + biases3)
            test_prediction = tf.nn.sigmoid(tf.matmul(layer3_test, weights4) + biases4)

            layer1_mail = tf.nn.relu(tf.matmul(tf_mail, weights1) + biases1)
            layer2_mail = tf.nn.relu(tf.matmul(layer1_mail, weights2) + biases2)
            layer3_mail = tf.nn.relu(tf.matmul(layer2_mail, weights3) + biases3)
            prediction_mail = tf.nn.sigmoid(tf.matmul(layer3_mail, weights4) + biases4)

        num_steps = 151
        start = time.time()
        N_epoch = 5

        # Train the model -- iteration within this model
        with tf.Session(graph=graph) as session:

            tf.initialize_all_variables().run()

            print("Initialized")

            for epoch in range(N_epoch):

                print("The current epoch is ", epoch + 1)

                for step in range(num_steps):
                    _, l, predictions = session.run([optimizer, loss, train_prediction])
                    acc = accuracy(predictions, trainY)
                    if (step % 100 == 0):
                        print("Loss at step %d: %f" % (step, l))
                        print("Accuracy: %.1f%%" % acc)

                print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), testY))

            # Save the variables to disk -- This is im
            # save_path = saver.save(session, "./model.ckpt")
            # print("Model saved in file: %s" % save_path)

        end = time.time()
        duration = end - start
        print("time consumed in training: %f seconds" % duration)

    def CNN(self, x_train, y_train, x_test, y_test, num_words):

        # The code for CNN detection
        # The input and output operation -- this is im as well
        input_seq = Input(shape=[self.MAX_WORDS_IN_SEQ, ], dtype='int32')
        embed_seq = Embedding(num_words + 1, self.EMBED_DIM, embeddings_initializer='glorot_uniform',
                              input_length = self.MAX_WORDS_IN_SEQ)(input_seq)

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
        fc2 = Dense(2, activation='softmax')(fc1)

        model = Model(input_seq, fc2)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        model.summary()

        MODEL_PATH = "/home/mgs/PycharmProjects/NLP_Final/spam-detection-using-deep-learning/spam_detect_char"

        history = model.fit(
            x_train,
            y_train,
            batch_size = 128,    # 128
            epochs = 5,    # 5
            callbacks=[ModelCheckpoint(MODEL_PATH, save_best_only=True)],
            validation_data = [x_test, y_test]
        )

        # Show the figure
        fig1 = plt.figure()
        plt.plot(history.history['loss'], 'r', linewidth=3.0)
        plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
        plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title('Loss Curves :CNN', fontsize=16)
        fig1.savefig('loss_cnn.png')
        plt.show()

        fig2 = plt.figure()
        plt.plot(history.history['acc'], 'r', linewidth=3.0)
        plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Accuracy Curves : CNN', fontsize=16)
        fig2.savefig('accuracy_cnn.png')
        plt.show()

    def RNN(self, x_train, y_train, x_test, y_test, num_words):

        input_seq = Input(shape=[self.MAX_WORDS_IN_SEQ, ], dtype='int32')
        embed_seq = Embedding(num_words + 1,
                              self.EMBED_DIM,
                              embeddings_initializer='glorot_uniform',
                              input_length=self.MAX_WORDS_IN_SEQ)

        sequence_input = Input(shape=(self.MAX_WORDS_IN_SEQ,), dtype='int32')
        embedded_sequences = embed_seq(sequence_input)
        l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
        preds = Dense(2, activation='softmax')(l_lstm)
        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
        print("Bidirectional LSTM")
        model.summary()

        MODEL_PATH = "/home/mgs/PycharmProjects/NLP_Final/spam-detection-using-deep-learning/spam_detect_char"
        cp = ModelCheckpoint(MODEL_PATH, monitor='val_acc', verbose=1, save_best_only=True)
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=2, callbacks=[cp])

        # Show the figure
        fig1 = plt.figure()
        plt.plot(history.history['loss'], 'r', linewidth=3.0)
        plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
        plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title('Loss Curves :CNN', fontsize=16)
        fig1.savefig('loss_cnn.png')
        plt.show()

        fig2 = plt.figure()
        plt.plot(history.history['acc'], 'r', linewidth=3.0)
        plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Accuracy Curves : CNN', fontsize=16)
        fig2.savefig('accuracy_cnn.png')
        plt.show()

def ShowScore(y_test, prediction):

    print('Accuracy score: {}'.format(accuracy_score(y_test, prediction)))
    print('Precision score: {}'.format(precision_score(y_test, prediction)))
    print('Recall score: {}'.format(recall_score(y_test, prediction)))
    print('F1 score: {}'.format(f1_score(y_test, prediction)))

def csv_to_numpy_array(filePath, delimiter):

    return np.genfromtxt(filePath, delimiter=delimiter, dtype='float32')

def Tokenize(message):

    message = message.lower()
    all_words = re.findall("[a-z0-9]+", message)
    return set(all_words)

def accuracy(predictions, labels):

  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def CateDecode(data_cate):

    data_label = np.zeros((len(data_cate), 1))

    for idx, item in enumerate(data_cate):

        # print(item)
        # print(np.int(item[0]) == 1 and np.int(item[1]) == 0)
        if np.int(item[0]) == 0 and np.int(item[1]) == 1:       # (0,1) -- 0
            data_label[idx] = 0
        elif np.int(item[0]) == 1 and np.int(item[1]) == 0:     # (1,0) -- 1
            data_label[idx] = 1
    return data_label

if __name__ == "__main__":

    test = SPAM()
    # test.Test_1()
    # test.Data_0()
    test.Test_2()

    # Show the first figure -- NB + SVM + RF
    # N = 10
    # accuracy = np.zeros((10, 3))
    # prediction = np.zeros((10, 3))
    # recall = np.zeros((10, 3))
    # f1= np.zeros((10, 3))
    #
    # for i in range(N):
    #
    #     print("The i epoch ")
    #
    #     y_test, predictions_NB, predictions_SVM, predictions_RF = test.Test_1()
    #
    #     accuracy[0] = accuracy_score(y_test, predictions_NB)
    #     accuracy[1] = accuracy_score(y_test, predictions_SVM)
    #     accuracy[2] = accuracy_score(y_test, predictions_RF)
    #
    #     prediction[0] = precision_score(y_test, predictions_NB)
    #     prediction[1] = precision_score(y_test, predictions_SVM)
    #     prediction[2] = precision_score(y_test, predictions_RF)
    #
    #     recall[0] = recall_score(y_test, predictions_NB)
    #     recall[1] = recall_score(y_test, predictions_SVM)
    #     recall[2] = recall_score(y_test, predictions_RF)
    #
    #     f1[0] = f1_score(y_test, predictions_NB)
    #     f1[1] = f1_score(y_test, predictions_SVM)
    #     f1[2] = f1_score(y_test, predictions_RF)
    #
    # print(np.average(accuracy, axis=1))
    # print(np.average(prediction, axis=1))
    # print(np.average(recall, axis=1))
    # print(np.average(f1, axis=1))