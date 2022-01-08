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
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten, TextVectorization, Bidirectional, CuDNNLSTM, Flatten, SeparableConv1D, MaxPooling1D, GlobalAveragePooling1D, Input
from keras.optimizers import adam_v2
from imblearn import over_sampling, under_sampling
import numpy as np
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest
from numpy import array, asarray, zeros
from keras.metrics import Precision, Recall 
from sklearn.utils import class_weight
import tensorflowjs as tfjs
from tensorflowjs import quantization



data = pd.read_csv(r"C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\processed_train(M).csv",sep=',')
y = data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

dataT = pd.read_csv(r"C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\test_X(M).csv",sep=',')
yT = pd.read_csv(r"C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\test_y(M).csv",sep=',')

weights = {0:,1:weights[0]}

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

def isEnglish(text):
  return text.isascii()


#remove punctuation
data.comment_text = data.comment_text.apply(lambda x:remove_punctuation(x))
dataT.comment_text = dataT.comment_text.apply(lambda x:remove_punctuation(x))

#all lowercase
data.comment_text = data.comment_text.apply(lambda x: x.lower())
dataT.comment_text = dataT.comment_text.apply(lambda x: x.lower())

##remove stopwords
#data.comment_text = data.comment_text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
#dataT.comment_text = dataT.comment_text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#remove numbers
data.comment_text = data.comment_text.str.replace('\d+', ' ',regex=True)
dataT.comment_text = dataT.comment_text.str.replace('\d+', ' ',regex=True)

#all english alphabet
data.comment_text = data.comment_text.apply(lambda x: ' '.join([word for word in x.split() if isEnglish(word) == True]))
dataT.comment_text = dataT.comment_text.apply(lambda x: ' '.join([word for word in x.split() if isEnglish(word) == True]))

#remove words with http
data.comment_text = data.comment_text.str.replace('(?:http.+)', ' ',regex=True)
dataT.comment_text = dataT.comment_text.str.replace('(?:http.+)', ' ',regex=True)

#remove single characters
data.comment_text = data.comment_text.str.replace('\s+[a-zA-Z]\s+', ' ',regex=True)
dataT.comment_text = dataT.comment_text.str.replace('\s+[a-zA-Z]\s+', ' ',regex=True)

#remove multiple spaces
data.comment_text = data.comment_text.str.replace('\s+', ' ',regex=True)
dataT.comment_text = dataT.comment_text.str.replace('\s+', ' ',regex=True)

#remove words with >20 characters
data.comment_text = data.comment_text.str.replace('^[a-z]{20,}$', '',regex=True)
dataT.comment_text = dataT.comment_text.str.replace('^[a-z]{20,}$', '',regex=True)

#tokenize + lemmatization
data.comment_text = data.comment_text.apply(lemmatize_text)
dataT.comment_text = dataT.comment_text.apply(lemmatize_text)
data.comment_text = data.comment_text.apply(lambda x:TreebankWordDetokenizer().detokenize(x))
dataT.comment_text = dataT.comment_text.apply(lambda x:TreebankWordDetokenizer().detokenize(x))


test_sentences, val_sentences, test_labels, val_labels = train_test_split(dataT.comment_text, yT, test_size=0.5, shuffle= True)

#renaming
train_sentences = data.comment_text
train_labels = y

##initialising keras vectorize layer
#vectorize_layer = TextVectorization(standardize='lower_and_strip_punctuation', max_tokens=20000, output_mode='int', output_sequence_length=200)

#def adapt_vectorizer():
#    with tf.device("CPU"):
#        vectorize_layer.adapt(train_sentences)
#        return "Vectorizer adapted"

#adapt_vectorizer()


#tokenize
tokenizer = Tokenizer(num_words=20000, char_level=False, lower=False)
tokenizer.fit_on_texts(train_sentences)

with open(r'C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\Pickle\keras_tokenizer', 'wb') as picklefile:
    pickle.dump(tokenizer,picklefile)

#text to num
input_length = 100

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=input_length, padding="post", truncating="post")

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=input_length, padding="post", truncating="post")

val_sequences = tokenizer.texts_to_sequences(val_sentences)
val_padded = pad_sequences(val_sequences, maxlen=input_length, padding="post", truncating="post")



## load the embedding into memory
#embeddings_index = dict()
#f = open(r'C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\glove_wiki\glove.6B.50d.txt', encoding='utf-8')
#for line in f:
#	values = line.split()
#	word = values[0]
#	coefs = asarray(values[1:], dtype='float32')
#	embeddings_index[word] = coefs
#f.close()
#print('Loaded %s word vectors.' % len(embeddings_index))


#vocab_size = len(tokenizer.word_index) + 1
dimensions = 50

## create a weight matrix for words in training docs
#embedding_matrix = zeros((vocab_size, dimensions))
#for word, i in tokenizer.word_index.items():
#	embedding_vector = embeddings_index.get(word)
#	if embedding_vector is not None:
#		embedding_matrix[i] = embedding_vector


#compiling, training, and exporting the model
model = Sequential()

#model.add(Input(shape=(1,), dtype="string"))
#model.add(vectorize_layer)
#model.add(Embedding(vocab_size, dimensions, weights=[embedding_matrix], input_length=input_length, trainable=False))
model.add(Embedding(20000, dimensions, input_length=100))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(128, dropout=0.2, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64, dropout=0.2, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(32, dropout=0.2)))
model.add(Dropout(0.3))
model.add(Dense(6, activation="sigmoid"))

#model.add(Embedding(vocab_size, dimensions, weights=[embedding_matrix], input_length=input_length, trainable=False))
#model.add(LSTM(128, dropout=0.3))
#model.add(Dense(1, activation="sigmoid"))

#model.add(Embedding(20000, 64, input_length=200))
#model.add(LSTM(128, dropout=0.2))
#model.add(Dense(1, activation="sigmoid"))

#model.add(Dropout(0.5,input_shape=(20000,)))
#model.add(Dense(64, activation="relu"))
#model.add(Dropout(0.2))
#model.add(Dense(1, activation="sigmoid"))


optimizer = adam_v2.Adam(learning_rate=3e-4	)

model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy", Precision(), Recall()])
model.summary()

model.fit(train_padded, train_labels, epochs=2, batch_size=64, validation_data=(val_padded,val_labels), shuffle=True)

model.save('nn_model_embeddings2/')
tfjs.converters.save_keras_model(model, 'nn_model_embeddings2(js)')

##testing the model
print("Model trained. Testing...")
model = load_model('nn_model_embeddings2/')

preds = model.predict(test_padded)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0

preds = np.array(preds)

accuracy = accuracy_score(np.array(test_labels), preds)
print('Accuracy: %f' % accuracy)

precision = precision_score(np.array(test_labels), preds, average='weighted')
print('Precision: %f' % precision)

recall = recall_score(np.array(test_labels), preds, average='weighted')
print('Recall: %f' % recall)

f1 = f1_score(np.array(test_labels), preds, average='weighted')
print('F1 score: %f' % f1)
