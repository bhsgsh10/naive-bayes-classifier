import sys
import string
import math
from collections import Counter
import re

"""
Results obtained for different values of m and k
-------------------------------------------------
train.txt
m   k     Accuracy  
1   1     98.7668
1  0.1    99.148

dev.txt
m   k       Accuracy   
1   1       97.486
1   0.05    97.486
1   0.1     97.307

test.txt
m   k       Accuracy
1   1       98.205
1   0.05    97.846
1   0.1     97.846

Removing stopwords,
train.txt
m   k       Accuracy
1   1       98.4977
1   0.1     99.0807
1   0.05    99.0134

dev.txt
m   k       Accuracy
1   1       96.9479
1   0.1     96.9479
2   0.1     97.127
1   0.05    97.127

test.txt
m   k       Accuracy
1   1       98.5637
1   0.1     98.2047
1   0.05    98.2047
-------------------------------------------------
"""

class NbClassifier(object):

    ham = "ham"
    spam = "spam"

    """
    A Naive Bayes classifier object has three parameters, all of which are populated during initialization:
    - a set of all possible attribute types
    - a dictionary of the probabilities P(Y), labels as keys and probabilities as values
    - a dictionary of the probabilities P(F|Y), with (feature, label) pairs as keys and probabilities as values
    """
    def __init__(self, training_filename, stopword_file):
        self.attribute_types = set()
        self.label_prior = {}    
        self.word_given_label = {}   

        self.collect_attribute_types(training_filename)
        if stopword_file is not None:
            self.remove_stopwords(stopword_file)
        self.train(training_filename)


    """
    A helper function to transform a string into a list of word strings.
    You should not need to modify this unless you want to improve your classifier in the extra credit portion.
    """
    def extract_words(self, text):
        no_punct_text = "".join([x for x in text.lower() if not x in string.punctuation])
        return [word for word in no_punct_text.split()]


    """
    Given a stopword_file, read in all stop words and remove them from self.attribute_types
    Implement this for extra credit.
    """
    def remove_stopwords(self, stopword_file):

        file_object = open(stopword_file)
        file_contents = file_object.read()
        stopword_list = file_contents.split('\n')
        print(len(self.attribute_types))
        self.attribute_types = self.attribute_types.difference(set(stopword_list))
        print(len(self.attribute_types))




    """
    Given a training datafile, add all features that appear at least m times to self.attribute_types
    """
    def collect_attribute_types(self, training_filename, m=1):
        file_object = open(training_filename)
        file_contents = file_object.read()
        contents_without_punctuation = self.replace_punc_space(file_contents)
        all_words = self.extract_words(contents_without_punctuation)
        counts_dict = Counter(all_words)
        most_recurring_words = {k:v for (k,v) in counts_dict.items() if v >= m}
        self.attribute_types = set(list(most_recurring_words.keys()))
        print(len(self.attribute_types))
        file_object.close()


    """
    Given a string, this function removes all punctuation marks and replaces them with whitespaces 
    """
    def replace_punc_space(self, text):
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        modified_text = text.translate(translator)
        return modified_text

    """
    Given a training datafile, estimate the model probability parameters P(Y) and P(F|Y).
    Estimates should be smoothed using the smoothing parameter k.
    """
    def train(self, training_filename, k=1):
        self.label_prior = {}
        self.word_given_label = {}
        ham_word_count = 0
        spam_word_count = 0
        list_ham = []
        list_spam = []

        with open(training_filename) as file_object:
            line = file_object.readline()
            while line:
                values = line.split("\t")
                if values[0] == self.ham:
                    message = self.replace_punc_space(values[1].lower())
                    word_list = self.extract_words(message)
                    list_ham.append(message)
                    #ham_word_count += len(line_without_newline.split())
                    ham_word_count += len(word_list)
                elif values[0] == self.spam:
                    message = self.replace_punc_space(values[1].lower())
                    word_list = self.extract_words(message)
                    list_spam.append(message)
                    spam_word_count += len(word_list)
                line = file_object.readline()

        file_object.close()

        self.label_prior[self.ham] = len(list_ham) / (len(list_ham) + len(list_spam))
        self.label_prior[self.spam] = len(list_spam) / (len(list_ham) + len(list_spam))

        #print(self.label_prior)

        smoothing_denominator_ham = ham_word_count + (k * len(self.attribute_types))
        smoothing_denominator_spam = spam_word_count + (k * len(self.attribute_types))
        #print(ham_word_count, spam_word_count)

        for attribute in self.attribute_types:
            counter = 0
            for line in list_ham:
                counter += line.count(attribute)

            self.word_given_label[(attribute, self.ham)] = (counter + k) / smoothing_denominator_ham
            counter = 0
            for line in list_spam:
                counter += line.count(attribute)
            self.word_given_label[(attribute, self.spam)] = (counter + k) / smoothing_denominator_spam
            counter = 0

        #print(len(self.word_given_label))

    """
    Given a piece of text, return a relative belief distribution over all possible labels.
    The return value should be a dictionary with labels as keys and relative beliefs as values.
    The probabilities need not be normalized and may be expressed as log probabilities. 
    """
    def predict(self, text):
        log_p_ham = math.log(self.label_prior[self.ham])
        log_p_spam = math.log(self.label_prior[self.spam])

        sum_log_ham = 0
        sum_log_spam = 0
        ham_dict = {k: v for (k, v) in self.word_given_label.items() if self.ham in k[1]}
        spam_dict = {k: v for (k, v) in self.word_given_label.items() if self.spam in k[1]}

        words_in_text = self.extract_words(text)
        for word in words_in_text:
            if (word, self.ham) in ham_dict:
                sum_log_ham += math.log(ham_dict[(word, self.ham)])

            if (word, self.spam) in spam_dict:
                sum_log_spam += math.log(spam_dict[(word, self.spam)])


        prob_ham = log_p_ham + sum_log_ham
        prob_spam = log_p_spam + sum_log_spam

        predicted_pr = {self.ham: prob_ham, self.spam: prob_spam}

        return predicted_pr


    """
    Given a datafile, classify all lines using predict() and return the accuracy as the fraction classified correctly.
    """
    def evaluate(self, test_filename):

        fp = open(test_filename)
        line = fp.readline()
        total_lines = 0
        accurate_count = 0
        wrong_ham = 0
        wrong_spam = 0
        while line:
            split_line = line.split('\t')
            if len(split_line) >= 2:
                total_lines += 1
                # split sentence between ham/spam and the message
                classified_as = split_line[0]
                message = self.replace_punc_space(split_line[1])
                predicted_probabilities = self.predict(message)
                if classified_as == self.ham:
                    if predicted_probabilities[self.ham] > predicted_probabilities[self.spam]:
                        accurate_count += 1
                    else:
                        wrong_ham += 1
                elif classified_as == self.spam:
                    if predicted_probabilities[self.spam] > predicted_probabilities[self.ham]:
                        accurate_count += 1
                    else:
                        wrong_spam += 1

            line = fp.readline()

        fp.close()
        classification_accuracy = (accurate_count / total_lines) * 100
        return classification_accuracy

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("\nusage: ./hmm.py [training data file] [test or dev data file] [(optional) stopword file]")
        exit(0)

    elif len(sys.argv) == 3:
        classifier = NbClassifier(sys.argv[1], None)
    else:
        classifier = NbClassifier(sys.argv[1], sys.argv[3])
    print(classifier.evaluate(sys.argv[2]))
    temp = sorted(classifier.word_given_label, key=classifier.word_given_label.get)

    temp.reverse()

    if temp[0:3] == [('i', 'ham'), ('you', 'ham'), ('to', 'ham')]:
        print("True")
    else:
        print("False")

    print(temp[0:3])