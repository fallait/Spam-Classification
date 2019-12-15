# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:53:18 2019

@author: Joseph
Using a dataset provided by Almeida, T.A. and Gomez Hidalgo, with due credit below.

Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. Contributions to the Study of SMS Spam Filtering: New Collection and Results.  Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11), Mountain View, CA, USA, 2011. [preprint]

Gómez Hidalgo, J.M., Almeida, T.A., Yamakami, A. On the Validity of a New SMS Spam Collection.  Proceedings of the 11th IEEE International Conference on Machine Learning and Applications (ICMLA'12), Boca Raton, FL, USA, 2012. [preprint]

Almeida, T.A., Gómez Hidalgo, J.M., Silva, T.P.  Towards SMS Spam Filtering: Results under a New Dataset.   International Journal of Information Security Science (IJISS), 2(1), 1-18, 2013. [Invited paper - full version]

http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/
"""
import os
import sys
import math
import time
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

categories = ['rec.sport.baseball', 'rec.sport.hockey']
twenty_train = fetch_20newsgroups(subset='train', categories = categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories = categories, shuffle=True, random_state=42)
Count_Vectorizer = CountVectorizer()

entry = (twenty_train.data[0].split("\n"))
#print(entry)
Count_Vectorizer.fit(twenty_train.data)
print(Count_Vectorizer.vocabulary_)