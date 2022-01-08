import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.tokenize.treebank import TreebankWordDetokenizer
from keras.models import load_model
import string
import pickle
from keras.preprocessing.sequence import pad_sequences 
import json
import io


#tokenize
with open(r'C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\Pickle\keras_tokenizer', 'rb') as keras_tokenizer:
    tokenizer = pickle.load(keras_tokenizer)

#word_index = tokenizer.word_index

#with open(r'C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\Pickle\tokenizer_word_index', 'wb') as jsonfile:
#    json.dump(word_index,jsonfile)

tokenizer_json = tokenizer.word_index

with io.open('tokenizer2.json', 'w', encoding='utf-8') as f:  

      f.write(json.dumps(tokenizer_json, ensure_ascii=False))