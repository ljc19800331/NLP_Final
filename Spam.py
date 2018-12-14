# Spam detection
import pandas as pd
import re
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

    def Data_0(self):

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

        # print("The data is ", data)
        # print("The shape of the data is ", len(data))

        # Collect the spam and non spam details and calculate the labels
        emails = []
        label = np.zeros((len(data), 1))
        for idx, (message, verdict) in enumerate(data):
            emails.append(message)
            if verdict == True:     # This is spam
                label[idx] = 1
            elif verdict == False:  # This is not spam
                label[idx] = 0

        # print(emails)
        # print(label)
        # print(np.where(label == 0))

        # Post processing of the data
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(emails)
        sequences = tokenizer.texts_to_sequences(emails)
        word2index = tokenizer.word_index
        num_words = len(word2index)

        # print(emails)
        # print(word2index)
        # print("The shape of the training data is ", num_words)

        # Randomize the data
        # Case 1: For ANN
        count_vector = CountVectorizer()
        data_others = count_vector.fit_transform(emails)
        # print("The data_others is ", data_others.toarray())

        label_cate = to_categorical(label)

        # label_cate = label

        x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(data_others, label_cate, test_size=self.test_size)
        x_train_1 = x_train_1.toarray()
        x_train_1 = x_train_1.astype(np.float32)
        x_test_1 = x_test_1.toarray()
        x_test_1 = x_test_1.astype(np.float32)
        print("The x_train_original is ", x_train_1)
        print("The index distribution is ", np.where(y_train_1 == 1))
        np.savetxt('check.txt', y_train_1)

        # Split the data with the train-test split size
        data = sequence.pad_sequences(sequences, maxlen=self.MAX_WORDS_IN_SEQ, padding='post', truncating='post')

        # normal output data
        _, _, y_train_vec, y_test_vec = train_test_split(data, label, test_size=self.test_size)

        # categorical data
        labels = to_categorical(label)  # Binary operation to the input -- Analyze why?
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=self.test_size)

        return x_train, y_train, x_test, y_test, y_train_vec, y_test_vec, num_words, x_train_1, x_test_1, y_train_1, y_test_1

    def Data_1(self):

        # The first data set
        # Load the first dataset
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
        def drop_fectures(features, data):
            data.drop(features, axis=1, inplace = True)
        def process_content(content):
            return " ".join(re.findall("[A-Za-z]+", content.lower()))
        drop_fectures(['COMMENT_ID', 'AUTHOR', 'DATE'], train_data)

        # Lower case
        train_data['processed_content'] = train_data['CONTENT'].apply(process_content)
        drop_fectures(['CONTENT'], train_data)
        # print(train_data.head())

        # Set the target training data and label
        emails = train_data['processed_content'].values
        # print("The shape of the training data is ", len(emails))
        labels = train_data['CLASS'].values
        max_len = max(map(lambda x: len(x), emails))

        # Case 1: For ANN
        count_vector = CountVectorizer()
        data_others = count_vector.fit_transform(emails)
        print("The data_others is ", data_others.toarray())

        # label_cate = to_categorical(labels)

        label_cate = labels

        x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(data_others, label_cate, test_size=self.test_size)
        x_train_1 = x_train_1.toarray()
        x_train_1 = x_train_1.astype(np.float32)
        x_test_1 = x_test_1.toarray()
        x_test_1 = x_test_1.astype(np.float32)

        # Case 2: For CNN or RNN
        # Token the words
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(emails)
        sequences = tokenizer.texts_to_sequences(emails)
        word2index = tokenizer.word_index
        num_words = len(word2index)
        print(sequences)
        print(emails)
        print(word2index)
        print("The shape of the training data is ", num_words)

        # Split the data with the train-test split size
        data = sequence.pad_sequences(sequences, maxlen = self.MAX_WORDS_IN_SEQ, padding='post', truncating='post')

        # normal output data
        _, _, y_train_vec, y_test_vec = train_test_split(data, labels, test_size = self.test_size)

        # categorical data
        labels = to_categorical(labels)         # Binary operation to the input -- Analyze why?
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = self.test_size)

        # le = preprocessing.LabelEncoder()

        # TF-IDF Processing?
        # print("The shape of the training data is ", x_train.shape)
        # print("The training data is ", x_train)

        return x_train, y_train, x_test, y_test, y_train_vec, y_test_vec, num_words, x_train_1, x_test_1, y_train_1, y_test_1

    def Data_2(self):

        print("loading training data")
        x_train = csv_to_numpy_array("/home/mgs/PycharmProjects/NLP_Final/Spam-detection-ANN/data/trainX.csv", delimiter=",")
        y_train = csv_to_numpy_array("/home/mgs/PycharmProjects/NLP_Final/Spam-detection-ANN/data/trainY.csv", delimiter=",")
        print("loading test data")
        x_test = csv_to_numpy_array("/home/mgs/PycharmProjects/NLP_Final/Spam-detection-ANN/data/testX.csv", delimiter=",")
        y_test = csv_to_numpy_array("/home/mgs/PycharmProjects/NLP_Final/Spam-detection-ANN/data/testY.csv", delimiter=",")

        print("The shape of the training data is ", x_train.shape)

        # Convert the Cate data to the label type
        y_train_vec = CateDecode(y_train)
        y_test_vec = CateDecode(y_test)

        data_train = sequence.pad_sequences(x_train, maxlen=self.MAX_WORDS_IN_SEQ, padding='post', truncating='post')
        data_test = sequence.pad_sequences(x_test, maxlen=self.MAX_WORDS_IN_SEQ, padding='post', truncating='post')

        # x_train = data_train
        # x_test = data_test

        # print("The current y_train_vec is ", y_train_vec)
        # print("The current y_test_vec is ", y_test_vec)
        print("The shape of the training data set is ", x_train.shape)
        print("The shape of the testing data set is ", y_train.shape)

        num_words = x_train.shape[0]

        return x_train, y_train, x_test, y_test, y_train_vec, y_test_vec, num_words

    def Data_3(self):

        data = '/home/mgs/PycharmProjects/NLP_Final/UDACITY-MLND-Spam-Detection/SMSSpamCollection'
        df = pd.read_table(data, names=['label', 'sms_message'])
        df.loc[:, 'label'] = df.label.map({'ham': 0, 'spam': 1})

        emails = df['sms_message'].values
        label = df['label'].values

        # print("The data of x_train is ", type(emails))
        # print("The shape of x_train is ", emails)
        # print("The data of x_train is ", type(label))
        # print("The shape of x_train is ", label)

        # Token
        # Token the words
        tokenizer = text.Tokenizer()
        tokenizer.fit_on_texts(emails)
        sequences = tokenizer.texts_to_sequences(emails)
        word2index = tokenizer.word_index
        num_words = len(word2index)

        # print(emails)
        # print(word2index)
        # print("The shape of the training data is ", num_words)

        # Case 1 for the original data
        count_vector = CountVectorizer()
        data_others = count_vector.fit_transform(emails)
        print("The data_others is ", data_others.toarray())
        x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(data_others, label, test_size=self.test_size)

        # Split the data with the train-test split size
        data = sequence.pad_sequences(sequences, maxlen = self.MAX_WORDS_IN_SEQ, padding='post', truncating='post')

        # normal output data
        _, _, y_train_vec, y_test_vec = train_test_split(data, label, test_size = self.test_size)

        # categorical data
        label = to_categorical(label)  # Binary operation to the input -- Analyze why?
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = self.test_size)

        # print("x_train is ", x_train)
        # print("The x_train's shape is ", x_train.shape)

        return x_train, y_train, x_test, y_test, y_train_vec, y_test_vec, num_words, x_train_1, x_test_1, y_train_1, y_test_1

    def Test_1(self):

        # Data 0
        # x_train, y_train, x_test, y_test, y_train_vec, y_test_vec, num_words = self.Data_0()
        # x_train = x_train.astype(np.float32)
        # x_test = x_test.astype(np.float32)

        # Data 1
        x_train, y_train, x_test, y_test, y_train_vec, y_test_vec, num_words, x_train_original, x_test_original, y_train_original, y_test_original = self.Data_1()
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)

        # Data 2 -- not to use this dataset
        # x_train, y_train, x_test, y_test, y_train_vec, y_test_vec = self.Data_2()   # x_train: float32 y_train: float32

        # Data 3
        # x_train, y_train, x_test, y_test, y_train_vec, y_test_vec, num_words, x_train_original, x_test_original, y_train_original, y_test_original = self.Data_3()
        # x_train = x_train_original.astype(np.float32)
        # x_test = x_test_original.astype(np.float32)
        # y_train = y_train_vec
        # y_test = y_test_vec

        # All ok with Data 1 and 3
        # NB -- for this problem
        naive_bayes = MultinomialNB()
        naive_bayes.fit(x_train_original, y_train_original)
        predictions_NB = naive_bayes.predict(x_test_original)
        print("The result of NB is ")
        ShowScore(y_test_original, predictions_NB)

        # SVM
        svc = SVC(kernel = 'linear', gamma = 1) # why linear kernel ? -- to discuss this problem
        svc.fit(x_train_original, y_train_original)
        predictions_SVM = svc.predict(x_test_original)
        print("The result of SVM is ")
        ShowScore(y_test_original, predictions_SVM)

        # Random forest
        clf = RandomForestClassifier()
        clf.fit(x_train_original, y_train_original)
        predictions_RF = clf.predict(x_test_original)
        print("The result of random forest is ")
        ShowScore(y_test_original, predictions_RF)

    def Test_2(self):

        # Data 0 -- Randomize the data
        x_train, y_train, x_test, y_test, y_train_vec, y_test_vec, num_words, x_train_original, x_test_original, y_train_original, y_test_original = self.Data_0()
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)

        # Data 1
        # x_train, y_train, x_test, y_test, y_train_vec, y_test_vec, num_words, x_train_original, x_test_original, y_train_original, y_test_original = self.Data_1()
        # x_train = x_train.astype(np.float32)
        # x_test = x_test.astype(np.float32)

        # Data 2 -- not to use this dataset
        # x_train, y_train, x_test, y_test, y_train_vec, y_test_vec, num_words = self.Data_2()   # x_train: float32 y_train: float32

        # Data 3
        # x_train, y_train, x_test, y_test, y_train_vec, y_test_vec, num_words, x_train_original, x_test_original, y_train_original, y_test_original = self.Data_3()
        # x_train = x_train.astype(np.float32)
        # x_test = x_test.astype(np.float32)

        # x_train = np.asarray(x_train.toarray())
        # x_train = x_train.astype(np.float32)
        # x_test = np.asarray(x_test.toarray())
        # x_test = x_test.astype(np.float32)
        # y_train = np.asarray(to_categorical(y_train))
        # y_test = np.asarray(to_categorical(y_test))

        # Check the data type
        print(x_train.shape)
        print(y_train.shape)
        print(y_train.dtype)
        print(x_test.shape)
        print(y_test.shape)

        # ANN -- This is an possible idea
        # Data 0 is ok
        # Data 1 is ok
        # Data 2 is ok -- original data
        # Data 3 is ok -- use data 3
        # self.ANN(x_train, y_train, x_test, y_test)
        # self.ANN(x_train_original, y_train_original, x_test_original, y_test_original)

        # CNN -- This is an idea
        # Data 0 is not so ok -- it is fine but not perfect
        # Data 1 is ok
        # Data 2 is not ok
        # Data 3 is ok -- use data 3
        # self.CNN(x_train, y_train, x_test, y_test, num_words)

        # RNN network -- LSTM -- This is pending
        self.RNN(x_train, y_train, x_test, y_test, num_words)

    def ANN(self, trainX, trainY, testX, testY):

        print("The training data is ", trainX.shape)

        # Basic ANN
        num_hidden_nodes1 = 2000    # 2000
        num_hidden_nodes2 = 1000
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
            optimizer = tf.train.AdamOptimizer(learning_rate=0.1,
                                               beta1=0.9, beta2=0.999,
                                               epsilon=1e-08).minimize(loss)

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

        # Train the model -- iteration within this model
        with tf.Session(graph=graph) as session:

            tf.initialize_all_variables().run()

            print("Initialized")

            for step in range(num_steps):
                _, l, predictions = session.run([optimizer, loss, train_prediction])
                acc = accuracy(predictions, trainY)
                if (step % 10 == 0):
                    print("Loss at step %d: %f" % (step, l))
                    print("Accuracy: %.1f%%" % acc)

            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), testY))

            # Save the variables to disk -- This is im
            save_path = saver.save(session, "./model.ckpt")
            print("Model saved in file: %s" % save_path)

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