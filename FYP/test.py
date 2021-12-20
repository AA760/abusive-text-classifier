import pandas as pd
from pandas.io.parsers import TextFileReader
import sklearn
import nltk
import string
string.punctuation
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential,load_model
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten, TextVectorization, Bidirectional, Input
from keras.optimizers import adam_v2
from imblearn import over_sampling, under_sampling
import numpy as np
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest
from numpy import array, asarray, zeros

dataT = pd.read_csv(r"C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\test_X.csv",sep=',')
yT = pd.read_csv(r"C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\test_y.csv",sep=',')


def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


#remove punctuation
dataT.comment_text = dataT.comment_text.apply(lambda x:remove_punctuation(x))

#all lowercase
dataT.comment_text = dataT.comment_text.apply(lambda x: x.lower())

#remove stopwords
dataT.comment_text = dataT.comment_text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#remove numbers
dataT.comment_text = dataT.comment_text.str.replace('\d+', '',regex=True)

#tokenize + lemmatization
dataT.comment_text = dataT.comment_text.apply(lemmatize_text)

#untokenize
dataT.comment_text = dataT.comment_text.apply(lambda x:TreebankWordDetokenizer().detokenize(x))

##testing the model
model = load_model('nn_model/')

prediction = model.predict(dataT.comment_text)
prediction = [1 if p > 0.5 else 0 for p in prediction]

accuracy = accuracy_score(yT, prediction)
print('Accuracy: %f' % accuracy)

precision = precision_score(yT, prediction)
print('Precision: %f' % precision)

recall = recall_score(yT, prediction)
print('Recall: %f' % recall)

f1 = f1_score(yT, prediction)
print('F1 score: %f' % f1)
