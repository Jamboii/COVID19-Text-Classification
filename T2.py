import json
import json
import csv
import os
import random
import math
import string
from collections import namedtuple, Counter

# T2 Deliverable
# Assuming dataset is downloaded, open each json and extract text data
# Assuming metadata is downloaded, open and extract publish time per json article
def createDataset():

    # Create training and test folder directories
    trainBeforePath = "train/before"
    trainAfterPath  = "train/after"
    testBeforePath  = "test/before"
    testAfterPath   = "test/after"

    if not os.path.exists(trainBeforePath):
        os.makedirs(trainBeforePath)
    if not os.path.exists(trainAfterPath):
        os.makedirs(trainAfterPath)
    if not os.path.exists(testBeforePath):
        os.makedirs(testBeforePath)
    if not os.path.exists(testAfterPath):
        os.makedirs(testAfterPath)

    ### Place all publish_time dates into sorted order (earliest -> latest)
    dataset_dir = "./biorxiv_medrxiv/pdf_json"
    dataset_list = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".json"):
            path = os.path.join(dataset_dir, filename)
            # print(path)
            dataset_list.append(path)
        else:
            continue

    # holds text per paper id
    paperId_text = {}
    # open and parse through all files
    for dataset_path in dataset_list:
        dataset_file = open(dataset_path,"r")
        dataset = json.load(dataset_file)

        # iterate through json
        paper_id = dataset["paper_id"]
        paperId_text[paper_id] = ""

        # get each piece of text from the article
        for section in ["abstract","body_text"]:
            text_section = dataset[section]
            for ref in text_section:
                paperId_text[paper_id] += ref["text"] + " "

    # Start to look through metadata for publish times
    data_by_time = dict()
    # named tuple for article which holds a publish time and its text
    Article = namedtuple('Article', ['publish_time', 'text'])
    print("Beginning to read metadata...")
    with open("metadata.csv","r") as csvfile:
        reader = csv.DictReader(csvfile)
        for article in reader:
            paper_id = article["sha"]
            publish_time = article["publish_time"]

            # place article in data if paper_id is in our dataset
            if paper_id in paperId_text:
                data_by_time[paper_id] = Article(publish_time=publish_time, text=paperId_text[paper_id])

    print("Completed reading of metadata...")
    print("Length of dataset: {}".format(len(data_by_time)))

    # Sort our newly acquired data by the publish time
    totalSet = {k: v for k, v in sorted(data_by_time.items(), key=lambda item: item[1].publish_time)}
    # print("Printing sorted data...")
    # print(sorted_data_by_time)

    ### let median date -> medianDate, half articles published before medianDate (beforeSet), half on or after (afterSet)
    medianDateIdx = len(totalSet) // 2

    beforeSet = dict(list(totalSet.items())[:medianDateIdx])
    afterSet = dict(list(totalSet.items())[medianDateIdx:])
    medianDate = list(afterSet.items())[0][1].publish_time
    print("Median date: {}".format(medianDate))

    ### Randomly select 90% of beforeSet (trainBefore) and 90% of afterSet (trainAfter)
    train_test_before = random.sample(range(len(beforeSet)), len(beforeSet))
    train_test_after = random.sample(range(len(afterSet)), len(afterSet))
    # create split indices to determing
    trainBefore_split = math.ceil(0.9 * len(train_test_before))
    trainAfter_split = math.ceil(0.9 * len(train_test_after))
    # create set lists for random indexing
    beforeSet_list = list(beforeSet.items())
    afterSet_list  = list(afterSet.items())

    # trainBefore
    print("Exporting 'before' training text...")
    for i in train_test_before[:trainBefore_split]:
        idx = train_test_before[i]        # index of the paper ID in beforeSet that we're using for training

        paper_id = beforeSet_list[idx][0] # get paper ID from before list
        text = beforeSet[paper_id].text   # get text using paper_id

        # write text to file
        ftrain = open(trainBeforePath + "/" + paper_id + ".txt", "a+", encoding="UTF-8")
        ftrain.writelines(text)
        ftrain.close()
    print("Finished exporting 'before' training text...")

    # trainAfter
    print("Exporting 'after' training text...")
    for i in train_test_after[:trainAfter_split]:
        idx = train_test_before[i]       # index of the afterSet that we're using for training

        paper_id = afterSet_list[idx][0] # get paper ID from after list
        text = afterSet[paper_id].text   # get text using paper_id

        # write text to file
        ftrain = open(trainAfterPath + "/" + paper_id + ".txt", "a+", encoding="UTF-8")
        ftrain.writelines(text)
        ftrain.close()
    print("Finished exporting 'after' training text...")

    ### Remaining 10% of before and after are testBefore, testAfter, respectively

    # testBefore
    print("Exporting 'before' test text...")
    for i in train_test_before[trainBefore_split:]:
        idx = train_test_before[i]        # index of the beforeSet that we're using for training

        paper_id = beforeSet_list[idx][0] # get paper ID from before list
        text = beforeSet[paper_id].text   # get text using paper_id

        # write text to file
        ftrain = open(testBeforePath + "/" + paper_id + ".txt", "a+", encoding="UTF-8")
        ftrain.writelines(text)
        ftrain.close()
    print("Finished exporting 'before' test text...")

    # testAfter
    print("Exporting 'after' test text...")
    for i in train_test_after[trainAfter_split:]:
        idx = train_test_before[i]       # index of the afterSet that we're using for training

        paper_id = afterSet_list[idx][0] # get paper ID from after list
        text = afterSet[paper_id].text   # get text using paper_id

        # write text to file
        ftrain = open(testAfterPath + "/" + paper_id + ".txt", "a+", encoding="UTF-8")
        ftrain.writelines(text)
        ftrain.close()
    print("Finished exporting 'after' test text...")

createDataset()
