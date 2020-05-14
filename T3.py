#================================================================================================================
#Run the code with two command line arguments:
#   1. the path to the directory containing the training set: /path/to/train
#   2. the path to the directory containing the test set: /path/to/test
#================================================================================================================

import json
import json
import csv
import os
import random
import math
import string
import sys
from collections import namedtuple, Counter

#================================================================================================================

# Return a dictionary of probabilities for each class (before/after median date)
#First parameter is the path to the training directory, and second parameter is the path to the test directory
def createUnigrams(basePathTrain, basePathTest):
    ### Create Naive Bayes Text Classification with Add-1 smoothing for each class

    unigram_probabilities = dict()

    V, N, C = findParams(basePathTrain)
    # combine vocab of training data with that of test data
    V_test, _, test_token_counts = findParams(basePathTest)
    V = V.union(V_test)

    for className in os.listdir(basePathTest):
        # set up variables
        if className not in list(C.keys()):
            continue

        # print(className)
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

        # add probability dict for current class to probability set
        unigram_probabilities[className] = prob_dict
    
    print("Conditional Class Probabilities created...")
    return unigram_probabilities

#================================================================================================================

### Convert all text to a BOW representation
# Although this isn't a "BOW Representation" really, just the frequency dictionary
# Parameter is the path to the directory containing the training data
# Returns V, N, C - detailed in the function below
def findParams(basePath):
    classes = ["before","after"]
    V = set()  # V all the vocab types in basePath, so len(V) is the length of the vocabulary (types)
    N = dict() # N the length of corpus PER class in basePath (token counts)
    C = dict() # C holds word frequencies amongst all articles PER class.
               # Presented as {before: Counter({"word": count}), after: Counter({"word", count})}

    # load articles
    print("Calculating unigram parameters for each class in {} ...".format(basePath))

    for classDir in classes:
        # create count of tokens for this class
        N_class = 0
        # create counter of frequencies for this class
        C_class = Counter()

        # parse through each file
        print("Class: {}".format(classDir))
        for file in os.listdir(os.path.join(basePath, classDir)): 
            # article file name
            article = os.fsdecode(file)

            # tokenize each article
            with open(os.path.join(basePath, classDir, article), "r", encoding="UTF-8") as fsource:
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

#================================================================================================================

### Test model on testing data created in T2 and compute the accuracy of the model
# Returns a dictionary that details the percentage of correct classifications from the files in the "before" test directory
#   and the "after" test directory 
def predict(testPath, unigram_probabilities):
    
    likelihood_prob = {"test_files_before": {}, "test_files_after": {}}

    classes = ["before","after"]

    #The count of how many test files there are in the before and after directories
    before_count = 0
    after_count = 0

    # loop through each class to determine the class likelihood of the article

    # look through each test file class directory
    for classDir in classes:
        # parse through each file in class directory
        for file in os.listdir(os.path.join(testPath, classDir)):
            
            if classDir == "before":
                before_count += 1
            elif classDir == "after":
                after_count += 1
            # article file name
            article = os.fsdecode(file)
            # create a predicted probability for each class 'className' for true class 'classDir'
            article_prob = dict()
            
            #create prediction of each className ("before" and "after") for the current test file
            for className in classes:
                # tokenize each article
                with open(os.path.join(testPath, classDir, article), "r", encoding="UTF-8") as fsource:
                    # within each file there is ONE "line" string composed of the entire article
                    for line in fsource:
                        # get rid of all punctuation and extraneous tokens, make all lower case, create list
                        preprocessLine = line.translate(str.maketrans('','',string.punctuation)).lower().split()
                        n = len(preprocessLine)

                        #Used in the formula for calculating the probabilities that the current article belongs in each class
                        logsum = 0.0

                        # grab training prob from each word in test article
                        for word in preprocessLine:
                            test_token_prob = unigram_probabilities[className][word]
                            # print(test_token_prob)
                            # Add onto a score???? for the article's class
                            # print(float(math.log(test_token_prob)))
                            logsum = logsum + float(math.log(test_token_prob))
                # Create a probability for that article's class (before or after median date)
                article_prob[className] = logsum
            
            correct_class = "test_files_" + classDir
            #If logsum of this article for "before" is greater than "after", then this article is more likely to belong in "before"
            if article_prob["before"] >= article_prob["after"]:
                likelihood_prob[correct_class][article] = "before"
            #If logsum of this article for "before" is less than "after", then this article is more likely to belong in "after"
            if article_prob["before"] < article_prob["after"]:
                likelihood_prob[correct_class][article] = "after"

    #This dict will hold the percentages of the total test articles that we correctly classified
    correct_classification_percentages = {"before": 0, "after": 0}

    #First we will increment the classification dict as counts, and then convert to percentages afterwards
    for classification in likelihood_prob["test_files_before"].values():
        if classification == "before":
            correct_classification_percentages["before"] += 1 
    for classification in likelihood_prob["test_files_after"].values():
        if classification == "after":
            correct_classification_percentages["after"] += 1
    # print(correct_classification_percentages)
    
    #Now convert to percentages:
    correct_classification_percentages["before"] = float(correct_classification_percentages["before"]) / before_count
    correct_classification_percentages["after"] = float(correct_classification_percentages["after"]) / after_count
    # print(correct_classification_percentages)

    return correct_classification_percentages

#================================================================================================================

if __name__ == '__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    # print(train_path)
    # findParams(train_path)
    unigram_probabilities = createUnigrams(train_path, test_path)
    # print(unigram_probabilities)
    correct_classification_percentages = predict(test_path, unigram_probabilities)
    print("The percentage of the \"before\" test files that were correctly predicted as \"before\": " + str(correct_classification_percentages["before"]))
    print("The percentage of the \"after\" test files that were correctly predicted as \"after\": " + str(correct_classification_percentages["after"]))
    