# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:53:18 2019

@author: Joseph
Using a dataset provided by Almeida, T.A. and Gomez Hidalgo, with due credit below.

Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. Contributions to the Study of SMS Spam Filtering: New Collection and Results.  Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11), Mountain View, CA, USA, 2011. [preprint]

Gómez Hidalgo, J.M., Almeida, T.A., Yamakami, A. On the Validity of a New SMS Spam Collection.  Proceedings of the 11th IEEE International Conference on Machine Learning and Applications (ICMLA'12), Boca Raton, FL, USA, 2012. [preprint]

Almeida, T.A., Gómez Hidalgo, J.M., Silva, T.P.  Towards SMS Spam Filtering: Results under a New Dataset.   International Journal of Information Security Science (IJISS), 2(1), 1-18, 2013. [Invited paper - full version]

http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/

@article{scikit-learn,
 title={Scikit-learn: Machine Learning in {P}ython},
 author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
 journal={Journal of Machine Learning Research},
 volume={12},
 pages={2825--2830},
 year={2011}
}
"""
import os
import sys
import math
import time
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
start_time = time.time()


def clean(handle):
    directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    name = os.path.join(directory, handle)
    dataset = open(name, "r")
    return dataset


def entry(line):
    category, point = line.split(None,1)
    return category, point.strip('\n').lower()

def setconstruct(dataset, p):
    datapoint = 0
    lines = list()
    for line in dataset:
        lines.append(entry(line))
        datapoint += 1
    random.shuffle(lines)
    trainingset = lines[0:int(p * datapoint)]
    testset = lines[int(p * datapoint):]
    return trainingset, testset

def dictconstruct(trainingset):
    bayesdict=[{},{}]
    independent = 0
    spam = 0
    ham = 0
    spambool = True
    for line in trainingset:
        if line[0] == 'spam':
            spambool = True
        else:
            spambool = False
        for word in line[1].split():
            if spambool:
                spam += 1
                if word not in bayesdict[0]:
                    bayesdict[0][word] = 2
                    independent += 1
                elif word in bayesdict[0]:
                    bayesdict[0][word] += 1
                if word not in bayesdict[1]:
                    bayesdict[1][word] = 1
            else:
                ham += 1
                if word not in bayesdict[1]:
                    bayesdict[1][word] = 2
                    independent += 1
                elif word in bayesdict[1]:
                    bayesdict[1][word] += 1
                if word not in bayesdict[0]:
                    bayesdict[0][word] = 1
    return bayesdict, spam, ham, independent

def prob(dictionary, spam, ham, independent):
    probham = 0
    probspam = 0
    for key in dictionary[0]:
        dictionary[0][key] = math.log(dictionary[0][key] / sum([spam, independent]))
    for key in dictionary[1]:
        dictionary[1][key] = math.log(dictionary[1][key] / sum([ham, independent]))
    probspam = spam / sum([spam, ham])
    probham = 1 - probspam
    return dictionary, probspam, probham

def scoring(line, probdictionary, probspam, classifier):
    probability = 0
    for word in line.split():
        if word in probdictionary[classifier]:
            probability += probdictionary[classifier][word]
        else:
            pass 
    if classifier == 0:
        return math.log(probspam) + probability
    else:
        return math.log(1-probspam) + probability
    
def test(testset, probdictionary, probspam):
    spam = bool
    correct = 0
    incorrect = 0
    for line in testset:
        if line[0] == "spam":
            spam = True
        elif line[0] == "ham":
            spam = False
        spamscore = scoring(line[1], probdictionary, probspam, 0)
        hamscore = scoring(line[1], probdictionary, probspam, 1)
        if spamscore > hamscore and spam:
            correct += 1
        elif spamscore < hamscore and spam:
            incorrect += 1
        elif spamscore > hamscore and not spam:
            incorrect += 1
        elif spamscore < hamscore and not spam:
            correct += 1
    return correct, incorrect
            
#trainingset, testset = setconstruct(clean("SMSSpamCollection"), 1)
#dictionary, spam, ham, independent = dictconstruct(trainingset)
#probdictionary, probspam, probham = prob(dictionary,spam, ham, independent)
#correct, incorrect = test(trainingset, probdictionary, probspam)

#print("The number correctly identified was: " + str(correct))
#print("The number incorrectly idenftified was: " + str(incorrect))
#print("The accuracy rate in sample is: " + str(correct/(correct+incorrect)))
#print("- %s seconds -" % (time.time() - start_time))

#Optimization
def outofsample():
    training = np.array([])
    outsampleaccuracy = np.array([])
    for trainingportion in range (5, 100, 5):
        trainingset, testset = setconstruct(clean("SMSSpamCollection"), trainingportion/100)
        dictionary, spam, ham, independent = dictconstruct(trainingset)
        probdictionary, probspam, probham = prob(dictionary,spam, ham, independent)
        correct, incorrect = test(testset, probdictionary, probspam)
        training = np.append(training, trainingportion/100)
        outsampleaccuracy = np.append(outsampleaccuracy, correct/(correct+incorrect))
    return training, outsampleaccuracy

def insample():
    training = np.array([])
    insampleaccuracy = np.array([])
    for trainingportion in range (5, 100, 5):
        trainingset, testset = setconstruct(clean("SMSSpamCollection"), trainingportion/100)
        dictionary, spam, ham, independent = dictconstruct(trainingset)
        probdictionary, probspam, probham = prob(dictionary,spam, ham, independent)
        correct, incorrect = test(trainingset, probdictionary, probspam)
        training = np.append(training, trainingportion/100)
        insampleaccuracy = np.append(insampleaccuracy, correct/(correct+incorrect))
    return training, insampleaccuracy

def optimize():
    training, outsample = outofsample()
    insampled = insample()[1]
    optimized = abs(outsample - insampled)
    optimizedpoint = training[np.argmin(optimized)]
    plt.plot(training, outsample)
    plt.plot(training, insampled)
    plt.axvline(x=optimizedpoint, color = 'r', linewidth=1)
    plt.xlabel('Portion of Data Used for Training')
    plt.ylabel('Accuracy')
    plt.legend(['out of sample accuracy', 'in sample accuracy', 'optimized'])
    plt.show()
    return optimizedpoint

#trainingset, testset = setconstruct(clean("SMSSpamCollection"), .8)
#dictionary, spam, ham, independent = dictconstruct(trainingset)
#probdictionary, probspam, probham = prob(dictionary, spam, ham, independent)
#print(scoring("big wire transfer", dictionary, probspam, 0))
#print(scoring("big wire transfer", dictionary, probspam, 1))
#print("- %s seconds -" % (time.time() - start_time))
"""
def optimizedresult():
    points = []
    for i in range(50):
        print(i)
        points.append(optimize())
    print(sum(points) / len(points))
"""
#TYPICALLY AROUND .865 of the data
optimize()
#print("- %s seconds -" % (time.time() - start_time))

