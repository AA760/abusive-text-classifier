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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential,load_model
from keras.layers import Embedding, LSTM, Dense, Dropout, TextVectorization, Bidirectional, Input
from keras.optimizers import adam_v2
import numpy as np
import tensorflow as tf
from numpy import array
from keras.metrics import Precision, Recall 
from sklearn.utils import class_weight
import keras
import matplotlib.pyplot as plt
import seaborn as sns


#importing data
data = pd.read_csv(r"..\Data\Binary\processed_train.csv",sep=',')
X_train=data.comment_text.astype(str)
y_train = data.abusive

X_test = pd.read_csv(r"..\Data\Binary\test_X.csv",sep=',').comment_text
y_test = pd.read_csv(r"..\Data\Binary\test_y.csv",sep=',')


#calculating class weights
weights =  list(class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train))
weights.sort()
weights = {0:weights[1],1:weights[0]}


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


#test/validation split
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle= True)


#initialize vectorize layer
length = 50
vectorize_layer = TextVectorization(standardize=None, max_tokens=20000, output_mode='int', output_sequence_length=length)

def adapt_vectorizer():
    with tf.device("CPU"):
        vectorize_layer.adapt(X_train)
        return "Vectorizer adapted"

adapt_vectorizer()


#building, training, and exporting the model
input_dim = 20000
output_dim = 128

model = Sequential()
model.add(Input(shape=(1,), dtype="string"))
model.add(vectorize_layer)
model.add(Embedding(input_dim, output_dim, input_length=length))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(128, dropout=0.2, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64, dropout=0.2, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(32, dropout=0.2)))
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))

optimizer = adam_v2.Adam(learning_rate=3e-4	)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy", Precision(), Recall()])
model.summary()

callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
model.fit(X_train, y_train, epochs=100, batch_size=128, class_weight=weights,validation_data=(X_val,y_val), shuffle=True, callbacks=[callback])
model.save('model/')


##testing the model
model = load_model('model/')
preds = model.predict(X_test)
preds = [1 if p > 0.5 else 0 for p in preds]


#confusion matrix
cm = confusion_matrix(y_test, preds)
ax = sns.heatmap(cm, annot=True, cmap='Blues',fmt='g')

ax.set_title('Confusion Matrix - Bidirectional LSTM w/ trained embeddings',weight='bold');
ax.set_xlabel('\nPredicted Values',weight='bold')
ax.set_ylabel('Actual Values\n',weight='bold');

ax.xaxis.set_ticklabels(['Not Abusive','Abusive'])
ax.yaxis.set_ticklabels(['Not Abusive','Abusive'])

plt.show()

#f1-scores
f10 = f1_score(y_test, preds, average='binary',pos_label=0)
f11 = f1_score(y_test, preds, average='binary',pos_label=1)
print('F1 score (Not Abusive): %f' % f10)
print('F1 score (Abusive): %f' % f11)
print('')

#other performance metrics
precision = precision_score(np.array(y_test), preds, average='binary')
print('Precision (Abusive): %f' % precision)

recall = recall_score(np.array(y_test), preds, average='binary')
print('Recall (Abusive): %f' % recall)

accuracy = accuracy_score(np.array(y_test), preds)
print('Overall accuracy: %f' % accuracy)