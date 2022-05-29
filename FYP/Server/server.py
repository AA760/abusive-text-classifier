import nltk
import string
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pickle
from flask import Flask
from flask import request
from flask_cors import CORS
import json
import numpy as np
import csv
from flask import Flask, jsonify, request
import json



with open(r'..\RF_tfidf(B)\tfidf_vectorizer', 'rb') as text_vectorizer:
    tfidf = pickle.load(text_vectorizer)

with open(r'..\RF_tfidf(B)\model', 'rb') as training_model:
    model = pickle.load(training_model)

app = Flask(__name__)
CORS(app)


def remove_punctuation(text):
    text = "".join([i for i in text if i not in string.punctuation])
    return text

def lemmatize_text(text):
    tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in tokenizer.tokenize(text)]

@app.route('/', methods=['GET', 'POST'])

def classifyText():

    if request.method == "POST":

        input = request.data.decode('UTF-8')

        #remove ip addresses
        input = input.replace('\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', ' ')

        #remove words with http
        input = input.replace('(?i:http.+)', ' ')

        #remove the word article
        input = input.replace('(?i:article.+)', ' ')

        #remove the word page
        input = input.replace('(?i:page.+)', ' ')

        #remove the word wikipedia
        input = input.replace('(?i:wikipedia.+)', ' ')

        #remove the word talk
        input = input.replace('(?i:talk.+)', ' ')

        #remove punctuation
        input = remove_punctuation(input)
        input = input.replace('’', ' ')

        #remove numbers
        input = input.replace('\d+', ' ')

        #all lowercase
        input = input.lower()

        #remove multiple spaces
        input = input.replace('\s+', ' ')

        #remove repeated letters
        input = input.replace(
        'a{3,}|b{3,}|c{3,}|d{3,}|e{3,}|f{3,}|g{3,}|h{3,}|i{3,}|j{3,}|k{3,}|l{3,}|m{3,}|n{3,}|o{3,}|p{3,}|q{3,}|r{3,}|s{3,}|t{3,}|u{3,}|v{3,}|w{3,}|x{3,}|y{3,}|z{3,}'
        , ' ')

        #remove words with >20 characters
        input = input.replace('^[a-z]{20,}$', ' ')

        #lemmatization
        input = lemmatize_text(input)
        input = TreebankWordDetokenizer().detokenize(input)

        #TF-IDF vectorization
        print('INPUT TEXT: ', {input})
        features = tfidf.transform([input])

        prediction = model.predict(features)

        print(str(prediction[0]))

        return str(prediction[0]) 

@app.route('/reportFP',methods = ['POST', 'GET'])
def reportTP():
   if request.method == 'POST':
       text = request.data.decode('UTF-8')
       with open(r'.\reportsFP.csv', 'a', newline='') as csvfile1:
            writer = csv.writer(csvfile1, delimiter=',')
            writer.writerow([text])
       return "Report received."

@app.route('/reportFN',methods = ['POST', 'GET'])
def reportFP():
   if request.method == 'POST':
       text = request.data.decode('UTF-8')
       with open(r'.\reportsFN.csv', 'a', newline='') as csvfile2:
            writer = csv.writer(csvfile2, delimiter=',')
            writer.writerow([text])
       return "Report received"

if __name__ == "__main__":
    app.run(debug=True)