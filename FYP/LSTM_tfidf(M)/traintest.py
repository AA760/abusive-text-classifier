import pandas as pd
from pandas.io.parsers import TextFileReader
import sklearn
import nltk
import string
string.punctuation
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix
import pickle
from keras.models import Sequential,load_model
from keras.layers import Embedding, LSTM, Dense,Bidirectional,TextVectorization,Input,Dropout
from keras.optimizers import adam_v2
import numpy as np
import tensorflow as tf
from numpy import array
from keras.metrics import Precision, Recall 
import keras

#importing data
data = pd.read_csv(r"..\Data\Multi\processed_train(M).csv",sep=',')
data.sample(frac=1)
X_train = data.comment_text
y_train = data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

data_test = pd.read_csv(r"..\Data\Multi\test_X(M).csv",sep=',')
X_test = data_test.comment_text
y_test = pd.read_csv(r"..\Data\Multi\test_y(M).csv",sep=',')


#cleaning text
def remove_punctuation(text):
    text = "".join([i for i in text if i not in string.punctuation])
    return text

def lemmatize_text(text):
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in tokenizer.tokenize(text)]

def clean(text):
    #remove ip addresses
    text = text.str.replace('\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', ' ',regex=True)

    #remove words with http
    text = text.str.replace('(?i:http.+)', ' ',regex=True)

    #remove the word article
    text = text.str.replace('(?i:article.+)', ' ',regex=True)

    #remove the word page
    text = text.str.replace('(?i:page.+)', ' ',regex=True)

    #remove the word wikipedia
    text = text.str.replace('(?i:wikipedia.+)', ' ',regex=True)

    #remove the word talk
    text = text.str.replace('(?i:talk.+)', ' ',regex=True)

    #remove punctuation
    text = text.apply(lambda x:remove_punctuation(x))
    text = text.str.replace('’', ' ',regex=True)

    #remove numbers
    text = text.str.replace('\d+', ' ',regex=True)

    #convert to lowercase
    text = text.apply(lambda x: x.lower())

    ##remove stopwords
    #text = text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    #remove multiple spaces
    text = text.str.replace('\s+', ' ',regex=True)

    #remove repeated letters
    text = text.str.replace(
        '''a{3,}|b{3,}|c{3,}|d{3,}|e{3,}|f{3,}|g{3,}|h{3,}|i{3,}|j{3,}|k{3,}|l{3,}|m{3,}|n{3,}|o{3,}|p{3,}|
        q{3,}|r{3,}|s{3,}|t{3,}|u{3,}|v{3,}|w{3,}|x{3,}|y{3,}|z{3,}'''
        , ' ',regex=True)

    #remove words with >20 characters
    text = text.str.replace('^[a-z]{20,}$', '',regex=True)

    #lemmatization
    text = text.apply(lemmatize_text)
    text = text.apply(lambda x:TreebankWordDetokenizer().detokenize(x))

    return text

X_train = clean(X_train)
X_test = clean (X_test)


#initializing vectorizer
vectorize_layer = TextVectorization(max_tokens=20000, standardize=None, output_mode="tf-idf", ngrams=(2))

def adapt_vectorizer():
    with tf.device("CPU"):
        vectorize_layer.adapt(X_train)
        return "Vectorizer adapted"

adapt_vectorizer()


#building, training, and exporting the model
model = Sequential()
model.add(Input(shape=(1,), dtype="string"))
model.add(vectorize_layer)
model.add(tf.keras.layers.Reshape((1, 20000)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(128, dropout=0.1)))
model.add(Dense(6, activation="sigmoid"))

optimizer = adam_v2.Adam(learning_rate=3e-4)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy", Precision(), Recall()])
model.summary()

model.fit(X_train, y_train, epochs=20, batch_size=64)
model.save('model/')


#predict with model
print("Model trained. Testing...")
model = load_model('model/')

preds = model.predict(X_test)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0

preds = np.array(preds)

accuracy = accuracy_score(np.array(y_test), preds)
print('Accuracy: %f' % accuracy)

precision = precision_score(np.array(y_test), preds, average='weighted')
print('Precision: %f' % precision)

recall = recall_score(np.array(y_test), preds, average='weighted')
print('Recall: %f' % recall)

f1 = f1_score(np.array(y_test), preds, average='weighted')
print('F1 score: %f' % f1)

cm = multilabel_confusion_matrix(y_test,preds)
print('Confusion Matrices:')
print(cm)
