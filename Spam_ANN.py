
# Ref: https://github.com/MoAbd/Spam-detection/blob/master/Spam%20detection.ipynb
from __future__ import division
import tensorflow as tf
import numpy as np
import tarfile
import os
import matplotlib.pyplot as plt
import time

def csv_to_numpy_array(filePath, delimiter):

    return np.genfromtxt(filePath, delimiter=delimiter, dtype='float32')

def accuracy(predictions, labels):

  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def import_data():

    print("loading training data")
    trainX = csv_to_numpy_array("/home/mgs/PycharmProjects/NLP_Final/Spam-detection-ANN/data/trainX.csv", delimiter=",")
    trainY = csv_to_numpy_array("/home/mgs/PycharmProjects/NLP_Final/Spam-detection-ANN/data/trainY.csv", delimiter=",")

    print("loading test data")
    testX = csv_to_numpy_array("/home/mgs/PycharmProjects/NLP_Final/Spam-detection-ANN/data/testX.csv", delimiter=",")
    testY = csv_to_numpy_array("/home/mgs/PycharmProjects/NLP_Final/Spam-detection-ANN/data/testY.csv", delimiter=",")

    return trainX, trainY, testX, testY

trainX, trainY, testX, testY = import_data()

print("The training data is ", trainX.shape)

# Basic ANN
num_hidden_nodes1 = 2000
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
    weights1 = tf.Variable(tf.truncated_normal([numFeatures, num_hidden_nodes1], stddev=np.sqrt(2.0 / (numFeatures))), name="v1")

    biases1 = tf.Variable(tf.zeros([num_hidden_nodes1]), name="v2")

    weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes1, num_hidden_nodes2], stddev=np.sqrt(2.0 / num_hidden_nodes1)), name="v3")

    biases2 = tf.Variable(tf.zeros([num_hidden_nodes2]), name="v4")

    weights3 = tf.Variable(tf.truncated_normal([num_hidden_nodes2, num_hidden_nodes3], stddev=np.sqrt(2.0 / num_hidden_nodes2)), name="v5")

    biases3 = tf.Variable(tf.zeros([num_hidden_nodes3]), name="v6")

    weights4 = tf.Variable(tf.truncated_normal([num_hidden_nodes3, numLabels], stddev=np.sqrt(2.0 / num_hidden_nodes3)), name="v7")

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
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = tf_train_labels))

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
with tf.Session(graph = graph) as session:

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
