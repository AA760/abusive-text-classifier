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
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten, TextVectorization, Bidirectional
from keras.optimizers import adam_v2
from imblearn import over_sampling, under_sampling
import numpy as np
import tensorflow as tf
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest
from numpy import array, asarray, zeros



data = pd.read_csv(r"C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\processed_train.csv",sep=',')
y = data.abusive

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
data.comment_text = data.comment_text.apply(lambda x:remove_punctuation(x))
dataT.comment_text = dataT.comment_text.apply(lambda x:remove_punctuation(x))

#all lowercase
data.comment_text = data.comment_text.apply(lambda x: x.lower())
dataT.comment_text = dataT.comment_text.apply(lambda x: x.lower())

#remove stopwords
data.comment_text = data.comment_text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
dataT.comment_text = dataT.comment_text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#remove numbers
data.comment_text = data.comment_text.str.replace('\d+', '',regex=True)
dataT.comment_text = dataT.comment_text.str.replace('\d+', '',regex=True)

#tokenize + lemmatization
data.comment_text = data.comment_text.apply(lemmatize_text)
dataT.comment_text = dataT.comment_text.apply(lemmatize_text)

#untokenize
data.comment_text = data.comment_text.apply(lambda x:TreebankWordDetokenizer().detokenize(x))
dataT.comment_text = dataT.comment_text.apply(lambda x:TreebankWordDetokenizer().detokenize(x))


#train/test split
train_size = int(data.shape[0] * 0.8)

train_sentences = data.comment_text[:train_size]
train_labels = y[:train_size]

test_sentences = data.comment_text[train_size:]
test_labels = y[train_size:]


#tokenize
tokenizer = Tokenizer(num_words=20000, char_level=False)
tokenizer.fit_on_texts(train_sentences)

with open(r'C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\Pickle\keras_tokenizer', 'wb') as picklefile:
    pickle.dump(tokenizer,picklefile)

#text to num + oversmapling
oversample = over_sampling.RandomOverSampler(sampling_strategy=1.0)
max_words = 200

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_words, padding="post", truncating="post")
train_padded, train_labels = oversample.fit_resample(train_padded,train_labels)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_words, padding="post", truncating="post")
test_padded, test_labels = oversample.fit_resample(test_padded,test_labels)

pred_sequences = tokenizer.texts_to_sequences(dataT.comment_text)
pred_padded = pad_sequences(pred_sequences, maxlen=max_words, padding="post", truncating="post")


##tokenize + vectorize + oversample
#tfidf = TfidfVectorizer(max_features=20000,min_df=2,ngram_range=(1,2),dtype=np.float32)
#train_padded = tfidf.fit_transform(train_sentences).toarray()
#test_padded = tfidf.transform(test_sentences).toarray()
#pred_padded = tfidf.transform(dataT.comment_text).toarray()

#undersample = under_sampling.RandomUnderSampler(sampling_strategy=1.0)
#train_padded, train_labels = undersample.fit_resample(train_padded,train_labels)
#test_padded, test_labels = undersample.fit_resample(test_padded,test_labels)

selector = SelectKBest(k=min(20000, train_padded.shape[1]))
selector.fit(train_padded, train_labels)
train_padded = selector.transform(train_padded)
test_padded = selector.transform(test_padded)
pred_padded = selector.transform(pred_padded)


# load the whole embedding into memory
dimensions = 200
embeddings_index = dict()
f = open(r'C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\glove_twitter\glove.twitter.27B.200d.txt', encoding='utf-8')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


vocab_size = len(tokenizer.word_index) + 1

# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, dimensions))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector


#compiling, training, and exporting the model
model = Sequential()

model.add(Embedding(vocab_size, dimensions, weights=[embedding_matrix], input_length=max_words, trainable=False))
model.add(Bidirectional(LSTM(128, dropout=0.5,return_sequences=True)))
model.add(Bidirectional(LSTM(64, dropout=0.2)))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

#model.add(Embedding(20000, 64, input_length=200))
#model.add(LSTM(128, dropout=0.2))
#model.add(Dense(1, activation="sigmoid"))

#model.add(Dropout(0.5,input_shape=(20000,)))
#model.add(Dense(64, activation="relu"))
#model.add(Dropout(0.2))
#model.add(Dense(1, activation="sigmoid"))


optimizer = adam_v2.Adam(learning_rate=3e-4)

model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.summary()

model.fit(train_padded, train_labels, epochs=2, validation_data=(test_padded, test_labels), batch_size = 1)

model.save('nn_model/')

model = load_model('nn_model/')

##testing the model
prediction = model.predict(pred_padded)
prediction = [1 if p > 0.5 else 0 for p in prediction]

accuracy = accuracy_score(yT, prediction)
print('Accuracy: %f' % accuracy)

precision = precision_score(yT, prediction)
print('Precision: %f' % precision)

recall = recall_score(yT, prediction)
print('Recall: %f' % recall)

f1 = f1_score(yT, prediction)
print('F1 score: %f' % f1)
