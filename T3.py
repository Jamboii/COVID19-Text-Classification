import json
import json
import csv
import os
import random
import math
import string
from collections import namedtuple, Counter

# Create a dictionary of probabilities for each class (before/after median date),
def createUnigrams():
    ### Create Naive Bayes Text Classification with Add-1 smoothing for each class
    # Training and test directories - make into command line args later
    basePathTrain = "./train/"
    basePathTest  = "./test/"

    unigram_probabilities = dict()
    V, N, C = findParams(basePathTrain)
    # combine vocab of training data with that of test data
    V_test, _, test_token_counts = findParams(basePathTest)
    V = V.union(V_test)

    for className in os.listdir(basePathTest):
        # set up variables
        if className not in list(C.keys()):
            continue

        print(className)
        word_freq = C[className]
        total = N[className] # all term frequencies from training class
        vocab = len(V) # size of vocabulary across entire dataset

        # create Naive Bayes probability dictionary for file
        prob_dict = dict()

        # calculate add-1 class conditional probabilities and add
        # Naive Bayes Model: P^hat(w_i|c) = (count(w_i,c) + 1) / (sum(count(w,c)) + |V|)
        for word in V:
            # calculate add-1 prob for word
            freq = word_freq[word] if word in word_freq.keys() else 0 # word frequencies from all training class documents
            prob = (freq + 1) / (total + vocab) # add-1 smoothing probability
            # add prob to dict
            prob_dict[word] = prob

        # add probability dict to probability set
        unigram_probabilities[className] = prob_dict
    print("Conditional Class Probabilities created...")


### Convert all text to a BOW representation
# Although this isn't a "BOW Representation" really, just the frequency dictionary
def findParams(basePath):
    classes = ["before","after"]
    V = set()  # V holds the length of the vocabulary (types)
    N = dict() # N holds length of corpus PER class (tokens)
    C = dict() # C holds word frequencies amongst all articles PER class

    # load articles
    print("Calculating unigram parameters for each class in {} ...".format(basePath))

    for classDir in classes:
        # create count of tokens for this class
        N_class = 0
        # create counter of frequencies for this class
        C_class = Counter()

        # parse through each file
        print("Class: {}".format(classDir))
        for file in os.listdir(basePath + classDir):
            # article file name
            article = os.fsdecode(file)

            # tokenize each article
            with open(basePath + classDir + "/" + article, "r", encoding="UTF-8") as fsource:
                # within each file there is ONE "line" string composed of the entire article
                for line in fsource:
                    # get rid of all punctuation and extraneous tokens, make all lower case, create list
                    preprocessLine = line.translate(str.maketrans('','',string.punctuation)).lower().split()

                    # add article to Vocabulary Set
                    V.update(preprocessLine)

                    # add article to token count for this class
                    N_class += len(preprocessLine)

                    # create Counter of all words in article
                    C_article = Counter(preprocessLine)
                    # update frequencies for class
                    C_class += C_article

        # update N
        N[classDir] = N_class
        # update C
        C[classDir] = C_class

    print("Calculations created...")
    return V, N, C

'''
### Test model on testing data created in T2 and compute the accuracy of the model
def predict(testPath):
    likelihood_prob = dict()
    classes = ["before","after"]

    # loop through each class to determine the class likelihood of the article

    # look through each class directory
    for classDir in classes:
        print("Class: {}".format(classDir))
        # parse through each file in class directory
        for file in os.listdir(testPath + classDir):
            # create a predicted probability for each class 'className' for true class 'classDir'
            article_prob = dict()
            # article file name
            article = os.fsdecode(file)

            for className in classes:
                # tokenize each article
                with open(testPath + classDir + "/" + article, "r", encoding="UTF-8") as fsource:
                    # within each file there is ONE "line" string composed of the entire article
                    for line in fsource:
                        # get rid of all punctuation and extraneous tokens, make all lower case, create list
                        preprocessLine = line.translate(str.maketrans('','',string.punctuation)).lower().split()
                        n = len(preprocessLine)

                        # grab training prob from each word in test article
                        for word in preprocessLine:
                            test_token_prob = unigram_probabilities[className][word]
                            # Add onto a score???? for the article's class
                            article_prob[className] = article_prob.get(className, 0) + math.log(test_token_prob)
                # Create a probability for that article class (before or after median date)
                article_prob[className] = (-1/n)*article_prob.get(className, 0)

            # assign article probabilities for each class
            likelihood_prob[article[:-4]] = article_prob
'''
