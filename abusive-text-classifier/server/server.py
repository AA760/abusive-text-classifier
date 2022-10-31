import nltk
import string
import pickle
from flask import Flask
from flask import request
from flask_cors import CORS
import csv
from flask import Flask, request
tokenizer = nltk.tokenize.WhitespaceTokenizer()
detokenizer = nltk.tokenize.treebank.TreebankWordDetokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


with open(r'..\..\trainscripts\rf_tfidf_b\tfidf_vectorizer.pickle', 'rb') as text_vectorizer:
    tfidf = pickle.load(text_vectorizer)

with open(r'..\..\trainscripts\rf_tfidf_b\model.pickle', 'rb') as training_model:
    model = pickle.load(training_model)

def clean(input):
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
    input = input.translate(str.maketrans('', '', string.punctuation))
    input = input.replace('’', ' ')

    #remove numbers
    input = input.replace('\d+', ' ')

    #all lowercase
    input = input.lower()

    #remove multiple spaces
    input = input.replace('\s+', ' ')

    #remove repeated letters
    input = input.replace(
    'a{3,}|b{3,}|c{3,}|d{3,}|e{3,}|f{3,}|g{3,}|h{3,}|i{3,}|j{3,}|k{3,}|l{3,}|m{3,}|\
    n{3,}|o{3,}|p{3,}|q{3,}|r{3,}|s{3,}|t{3,}|u{3,}|v{3,}|w{3,}|x{3,}|y{3,}|z{3,}'
    , ' ')

    #remove words with >20 characters
    input = input.replace('^[a-z]{20,}$', ' ')

    #lemmatization
    input = [lemmatizer.lemmatize(w) for w in tokenizer.tokenize(input)]
    input = detokenizer.detokenize(input)

    return input


app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET', 'POST'])

def classifyText():

    if request.method == "POST":

        input = request.data.decode('UTF-8')

        input = clean(input)

        #TF-IDF vectorization
        print('INPUT TEXT: ', {input})
        features = tfidf.transform([input])

        prediction = model.predict(features)

        print(str(prediction[0]))

        return str(prediction[0]) 

@app.route('/reportFP',methods = ['POST', 'GET'])
def reportFP():
   if request.method == 'POST':
       text = request.data.decode('UTF-8')
       with open(r'.\reportsFP.csv', 'a', newline='') as csvfile1:
            writer = csv.writer(csvfile1, delimiter=',')
            writer.writerow([text])
       return "Report received."

@app.route('/reportFN',methods = ['POST', 'GET'])
def reportFN():
   if request.method == 'POST':
       text = request.data.decode('UTF-8')
       with open(r'.\reportsFN.csv', 'a', newline='') as csvfile2:
            writer = csv.writer(csvfile2, delimiter=',')
            writer.writerow([text])
       return "Report received."


if __name__ == "__main__":
    app.run(debug=True)