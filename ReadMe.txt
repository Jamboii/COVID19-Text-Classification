CSC470 Natural Language Processing - Final Project Report
Text Classification using the COVID-19 Open Research Dataset (CORD-19)
Alex Benasutti, Joe Candiano, Mike Knothe
Final Project ReadMe
--------------------------------------------------------

The submission for this final project will be named final_project.tar.gz and it will contain seven files/folders. Upon opening the .tar file, there will be a folder named COVID19-Text-Classification that contains all the project files. The files will be:

 - T2.py, contains the source code for T2 for creating the train and test sets to be used (partially fulfills D1). Please note: Do not run T2. It is simply provided to see the source code for creating the train and test sets. The “train” and “test” folders explained below already have the correct files in them, and running T2.py again may cause the distribution of files to be altered.
 - T3.py, contains the rest of the source code for T3 and T4 to create the Naive-Bayes text classifier and computes the cross entropies of the train and test sets (also fulfills D1)
 - This ReadMe.txt file to explain the project contents and usage instructions (fulfills D1)
 - The ‘train’ and ‘test’ folders that contain the files of the before and after sets of the corpus’ medianDate (these folders fulfill D2)
 - analysis.txt which is a writeup and analysis of the output from T3 and T4 (this fulfills D3)
 - questions.txt which is a file containing our answers to the five questions asked at the bottom of the project file (this fulfills D4)

--------------------------------------------------------

To run our project in the terminal, you must be within the same directory where you downloaded our .tar file, then run the command: 

python (or python3) T3.py /your/unique/path/to/train/ /your/unique/path/to/test/

The output of this program will first be showing the unigrams and their probabilities being calculated for the whole dataset and the training sets of the dataset. The program will then output the percentage of the correctly predicted ‘before’ and ‘after’ test set files with the Naive-Bayes text classifier. The last of the program’s output will be the four cross entropy calculations, beforeTrain-beforeTest, beforeTrain-afterTest, afterTrain-beforeTest, and afterTrain-afterTest. 

If you do wish to run the script for T2, you must delete the current test and train directories first. To run the script, run the following command:

python (or python3) T2.py datasetDir metadataPath basePath

where datasetDir is the directory of json article files, metadataPath is the path to the metadata.csv file, and basePath is the directory you wish to export the training and test sets to. Running this script will output both a test/ and train/ directory, both containing a randomly selected before/ and after/ directory, pertaining to the categories “published before medianDate” and “published after medianDate,” respectively.