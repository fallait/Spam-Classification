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

def makedataframe(dataset):
    SMS = []
    cats = []
    for line in dataset:
        row = entry(line)
        cats.append(int(row[0] == 'spam'))
        SMS.append(row[1])
    data = {'SMS' : SMS,
            'Spam' : cats}
    dataframe = pd.DataFrame(data)
    return dataframe
    
def setup(dataframe):
    entry = dataframe.SMS
    category = dataframe.Spam
    Count_Vector = CountVectorizer()
    entry = Count_Vector.fit_transform(entry)
    Tfidf_Vector = TfidfTransformer()
    entry= Tfidf_Vector.fit_transform(entry)
    #Y_Training = Count_Vector.fit_transform(Y_Training)
    X_Training, X_Testing, Y_Training, Y_Testing = train_test_split(entry, category, test_size = .13, random_state = 1)
    classifier = SVC(kernel='linear')
    classifier.fit(X_Training, Y_Training)
    Y_Predict = classifier.predict(X_Testing)
    print(confusion_matrix(Y_Testing, Y_Predict))
    print(classification_report(Y_Testing, Y_Predict))
    print(accuracy_score(Y_Testing, Y_Predict))
    #accuracies = cross_val_score(clf, features, labels, scoring='accuracy', cv=CV)

#def evaluation():
    
setup(makedataframe(clean("SMSSpamCollection")))

print("- %s seconds -" % (time.time() - start_time))