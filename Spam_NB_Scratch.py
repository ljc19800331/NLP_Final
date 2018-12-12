'''
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

import re
import math
import glob
import re
import random
from collections import Counter
from collections import defaultdict, Counter

class NB():
    def __init__(self, k=0.5):
        self.k = k
        self.word_prob = []

    def train(self, trainset):

        # Calculate the total spam words
        total_spams = total_non_spams = 0

        # The spam or non spam by verdict
        "Trainig set consist of pairs (message, is_spam)"

        for message, verdict in trainset:
            if verdict:
                total_spams += 1
            else:
                total_non_spams += 1

        # Total number of the words
        counts = self.count_words(trainset)

        # Return the word probability
        self.word_prob = self.word_probabilities(counts, total_spams, total_non_spams)

    def classify(self, message):

        # The spam probability
        p_spam = self.spam_probability(message)

        return p_spam

    def tokenize(self, message):
        message = message.lower()
        all_words = re.findall("[a-z0-9]+", message)
        return set(all_words)

    def count_words(self, training_set):

        "Trainig set consist of pairs (message, is_spam)"
        # Count the spam and non spam words together
        counts = defaultdict(lambda: [0, 0])
        for message, is_spam in training_set:
            for word in self.tokenize(message):
                counts[word][0 if is_spam else 1] += 1
        return counts

    def word_probabilities(self, counts, total_spams, total_non_spams):

        """turns the word_count into list of triplets [w, P(w/spam), P(w/~spam)]
        It gives the probability of word being in spam and not being in spam
        P(w|S) = (k + number of spam containing w)/(2*k + total number of spams)
        P(w |~S) = (k + number of ~spam containing w)/(2*k + total number of ~spams)"""

        return [(w,
                 (self.k + spam) / (2 * self.k + total_spams),
                 (self.k + not_spam) / (2 * self.k + total_non_spams))
                for w, (spam, not_spam) in counts.items()]

    def spam_probability(self, message):

        # Token the message -- this is a new message given by the user
        message_words = self.tokenize(message)
        log_prob_if_spam = log_prob_if_not_spam = 0.0

        for word, prob_if_spam, prob_if_not_spam in self.word_prob:
            if word in message_words:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_not_spam += math.log(prob_if_not_spam)

            # if *word* doesn't appear in the message
            # add the log probability of _not_ seeing it
            # which is log(1 - probability of seeing it)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

        # Nice operation
        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_not_spam = math.exp(log_prob_if_not_spam)

        return prob_if_spam / (prob_if_spam + prob_if_not_spam)

# The path planning for this dataset
PATH = "/home/mgs/PycharmProjects/NLP_Final/"
TP = (True, True)
TN = (False, False)
FP = (True, False)
FN = (False, True)

def split_data(data, prob):
    random.shuffle(data)
    data_len = len(data)
    split = int(prob * data_len)
    train_data = data[:split]
    test_data = data[split:]

    return train_data, test_data

def spammiest_word(classifier):
    words_with_prob = classifier.word_prob

    words_with_prob.sort(key=lambda row: row[1] / (row[1] + row[2]))

    return [w[0] for w in words_with_prob][-5:]

def accuracy(result):
    return (result[TP] + result[TN]) / (result[TP] + result[TN] + result[FP] + result[FN])

def precision(result):
    return result[TP] / (result[TP] + result[FP])

def recall(result):
    return (result[TP] / (result[TP] + result[FN]))

def most_misclassified(result):
    # sorting in acending order of spam probability
    result.sort(key=lambda row: row[2])

    # have high probability of being classified as spam while it is not spam
    spammiest_hams = list(filter(lambda row: not row[1], result))[-5:]

    # have lowest probability of beign classified as spam while it is spam
    hammiest_spams = list(filter(lambda row: row[1], result))[:5]

    return spammiest_hams, hammiest_spams

def main():

    data = []

    # Preprocess the data
    for verdict in ['spam', 'not_spam']:
        for files in glob.glob(PATH + verdict + "/*")[:500]:

            # Read each files within the spam and not spam folder
            # print("The current files is ", files)
            # Assign the spam property to the list or string

            is_spam = True if verdict == 'spam' else False

            with open(files, "r", encoding='utf-8', errors='ignore') as f:
                for line in f:

                    # The title begins with "subject" -- this is im
                    if line.startswith("Subject:"):
                        subject = re.sub("^Subject: ", "", line).strip()
                        # print("The current subject is ", subject)
                        data.append((subject, is_spam))

    print("The data is ", data)

    # Split the data
    random.seed(0)
    train_data, test_data = split_data(data, 0.75)

    print("The train data is ", train_data)
    print("The length of training data is ", len(train_data))
    print("The testing data is ", test_data)
    print("The length of testing data is ", len(test_data))

    # Train the model
    classifier = NB()
    classifier.train(train_data)

    # Show the data demo
    print("Spam" if classifier.classify("Get free laptops now!") > 0.5 else "Not Spam")

    # Notice: The Bayes Naive is based on the fact whether a probability of a word is already calculated
    classified = [(subject, is_spam, classifier.classify(subject))
                  for subject, is_spam in test_data]
    count = Counter((is_spam, spam_probability > 0.5)
                    for _, is_spam, spam_probability in classified)
    spammiest_hams, hammiest_spams = most_misclassified(classified)

    print("Accuracy: ", accuracy(count))
    print("Precision: ", precision(count))

    # print("Recall: ", recall(count))
    # print("\nTop 5 falsely classified as spam:\n\n", spammiest_hams)
    # print("\nTop 5 falsely classified as not spam:\n\n", hammiest_spams)
    # print("\nMost spammiest words: ", spammiest_word(classifier))

if __name__ == "__main__":
    main()
