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
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten, TextVectorization, Bidirectional, Input, Reshape
from keras.optimizers import adam_v2
from imblearn import over_sampling, under_sampling
import numpy as np
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest
from numpy import array, asarray, zeros
import re
from tensorflow import saved_model
from keras.utils.generic_utils import custom_object_scope
from keras.metrics import Precision, Recall 
from sklearn.utils import class_weight

data = pd.read_csv(r"C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\processed_train.csv",sep=',')
y = data.abusive

dataT = pd.read_csv(r"C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\test_X.csv",sep=',')
yT = pd.read_csv(r"C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\test_y.csv",sep=',')


weights =  list(class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y))
weights.sort()
weights = {0:weights[1],1:weights[0]}
print(weights)


#def remove_punctuation(text):
#    punctuationfree="".join([i for i in text if i not in string.punctuation])
#    return punctuationfree

def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


##remove punctuation
#data.comment_text = data.comment_text.apply(lambda x:remove_punctuation(x))
#dataT.comment_text = dataT.comment_text.apply(lambda x:remove_punctuation(x))

##all lowercase
#data.comment_text = data.comment_text.apply(lambda x: x.lower())
#dataT.comment_text = dataT.comment_text.apply(lambda x: x.lower())

#remove stopwords
#data.comment_text = data.comment_text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#dataT.comment_text = dataT.comment_text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#remove numbers
data.comment_text = data.comment_text.str.replace('\d+', '',regex=True)
dataT.comment_text = dataT.comment_text.str.replace('\d+', '',regex=True)

#tokenize + lemmatization
data.comment_text = data.comment_text.apply(lemmatize_text)
dataT.comment_text = dataT.comment_text.apply(lemmatize_text)

#untokenize
data.comment_text = data.comment_text.apply(lambda x:TreebankWordDetokenizer().detokenize(x))
dataT.comment_text = dataT.comment_text.apply(lambda x:TreebankWordDetokenizer().detokenize(x))

train_sentences = data.comment_text
train_labels = y
#train/test split
#train_size = int(data.shape[0] * 0.8)

#train_sentences = data.comment_text[:train_size]
#train_labels = y[:train_size]

#test_sentences = data.comment_text[train_size:]
#test_labels = y[train_size:]

val_size = int(data.shape[0] * 0.8)

val_sentences = dataT.comment_text[val_size:]
val_labels = yT[val_size:]

pred_sentences = dataT.comment_text[:val_size]
pred_labels = yT[:val_size]



model = Sequential()
model.add(Input(shape=(1,), dtype="string"))

# Create a custom standardization function
#def custom_standardization(input_data):
#  lowercase = tf.strings.lower(input_data)
#  stripped_html = tf.strings.regex_replace(lowercase, '/<[^>]+>/', ' ')
#  stripped_punc = tf.strings.regex_replace(stripped_html,r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', ' ')
#  stripped_nums = tf.strings.regex_replace(stripped_punc, '\d+', ' ')
#  for i in stop:
#      stripped_nums = tf.strings.regex_replace(stripped_nums, f' {i} ', ' ')
#  return tf.strings.regex_replace(stripped_nums, '[%s]' % re.escape(string.punctuation), '')

vectorize_layer = TextVectorization(max_tokens=20000, standardize='lower_and_strip_punctuation', output_mode="tf-idf", ngrams=(2))




def adapt_vectorizer():
    with tf.device("CPU"):
        vectorize_layer.adapt(train_sentences)
        return "Vectorizer adapted"

adapt_vectorizer()


model.add(vectorize_layer)
model.add(tf.keras.layers.Reshape((1, 20000)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(128, dropout=0.2, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64, dropout=0.2, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(32, dropout=0.2)))
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))

optimizer = adam_v2.Adam(learning_rate=3e-4)

model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy", Precision(), Recall()])
model.summary()

model.fit(train_sentences, train_labels, epochs=100, batch_size=64, class_weight=weights, validation_data=(val_sentences, val_labels))

model.save('nn_model/')


#testing the model
model = load_model('nn_model/')

prediction = model.predict(pred_sentences)
prediction = [1 if p > 0.5 else 0 for p in prediction]

accuracy = accuracy_score(pred_labels, prediction)
print('Accuracy: %f' % accuracy)

precision = precision_score(pred_labels, prediction)
print('Precision: %f' % precision)

recall = recall_score(pred_labels, prediction)
print('Recall: %f' % recall)

f1 = f1_score(pred_labels, prediction)
print('F1 score: %f' % f1)
