import pandas as pd
from pandas.io.parsers import TextFileReader
import sklearn
import nltk
import string
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.utils import class_weight
import numpy as np
import seaborn as sns
import matplotlib as plt
import psycopg2

#importing data
psgdb = psycopg2.connect(host="localhost", database="datasets", user="postgres", password="atc9310")

c = psgdb.cursor()
c.execute("SELECT comment_text, abusive FROM bTrain")

table = c.fetchall()
c.close()

data = pd.DataFrame(table, columns =['comment_text', 'abusive'])
data['abusive'] = data['abusive'].replace(True,1)
data['abusive'] = data['abusive'].replace(False,0)

data.sample(frac=1)
X = data.comment_text
y = data.abusive


#calculating class weights
weights =  list(class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y))
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

cleanedX = clean(X)
X = cleanedX


#word frequency graph
a = X.str.cat(sep=' ')
words = nltk.tokenize.word_tokenize(a)
word_dist = nltk.FreqDist(words)
word_dist.plot(20);


#tokenize + vectorize
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(3,5), analyzer='char')
X = tfidf.fit_transform(X)


#printing tfidf matrix
feature_names = tfidf.get_feature_names_out()
corpus_index = [n for n in cleanedX]
tfidf_matrix = pd.DataFrame(X.todense(), index=corpus_index, columns=feature_names)
print(tfidf_matrix)


classifier = RandomForestClassifier(n_estimators=100, random_state=0,
                                    n_jobs=-1, verbose=1,  criterion="entropy", class_weight=weights)
classifier.fit(X, y)
print("Model trained. Exporting...")
print("")



with open(r'.\model', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)
    print("Exported model.")
    print("")

with open(r'.\tfidf_vectorizer', 'wb') as picklefile:
    pickle.dump(tfidf,picklefile)
    print("Exported vectorizer.")
    print("")

