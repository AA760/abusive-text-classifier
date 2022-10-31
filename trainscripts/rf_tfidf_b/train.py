import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.utils import class_weight
import numpy as np
import psycopg2
tokenizer = nltk.tokenize.WhitespaceTokenizer()
detokenizer = nltk.tokenize.treebank.TreebankWordDetokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


#import data
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


#calculate class weights
weights =  list(class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y))
weights = {0:weights[1],1:weights[0]}


def clean(text):
    #remove ip addresses
    text = text.str.replace('\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', ' ',regex=True)

    #remove words with http
    text = text.str.replace('(?i:http.+)', ' ',regex=True)

    #remove 'article'
    text = text.str.replace('(?i:article.+)', ' ',regex=True)

    #remove 'page'
    text = text.str.replace('(?i:page.+)', ' ',regex=True)

    #remove 'wikipedia'
    text = text.str.replace('(?i:wikipedia.+)', ' ',regex=True)

    #remove 'talk'
    text = text.str.replace('(?i:talk.+)', ' ',regex=True)

    #remove punctuation
    text = text.apply(lambda x:"".join([i for i in x if i not in string.punctuation]))
    text = text.str.replace('’', ' ',regex=True)

    #remove numbers
    text = text.str.replace('\d+', ' ',regex=True)

    #lowercase
    text = text.apply(lambda x: x.lower())

    #remove multiple spaces
    text = text.str.replace('\s+', ' ',regex=True)

    #remove repeated letters
    text = text.str.replace(
        '''a{3,}|b{3,}|c{3,}|d{3,}|e{3,}|f{3,}|g{3,}|h{3,}|i{3,}|j{3,}|k{3,}|l{3,}|m{3,}|n{3,}|o{3,}|p{3,}|
        q{3,}|r{3,}|s{3,}|t{3,}|u{3,}|v{3,}|w{3,}|x{3,}|y{3,}|z{3,}'''
        , ' ',regex=True)

    #remove >20 character words
    text = text.str.replace('^[a-z]{20,}$', '',regex=True)

    #lemmatize
    text = text.apply(lambda x:[lemmatizer.lemmatize(w) for w in tokenizer.tokenize(x)])
    text = text.apply(lambda x:detokenizer.detokenize(x))

    return text

cleanedX = clean(X)
X = cleanedX


# #word frequency graph
# a = X.str.cat(sep=' ')
# words = nltk.tokenize.word_tokenize(a)
# word_dist = nltk.FreqDist(words)
# word_dist.plot(20);


#tokenize + vectorize
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(3,5), analyzer='char')
X = tfidf.fit_transform(X)


# #tfidf matrix
# feature_names = tfidf.get_feature_names_out()
# corpus_index = [n for n in cleanedX]
# tfidf_matrix = pd.DataFrame(X.todense(), index=corpus_index, columns=feature_names)
# print(tfidf_matrix)

#create and train model
classifier = RandomForestClassifier(n_estimators=100, random_state=0,
                                    n_jobs=-1, verbose=1,  criterion="entropy", class_weight=weights)
classifier.fit(X, y)
print("Model trained. Exporting...")
print("")



with open(r'.\model.pickle', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)
    print("Exported model.")
    print("")

with open(r'.\tfidf_vectorizer.pickle', 'wb') as picklefile:
    pickle.dump(tfidf,picklefile)
    print("Exported vectorizer.")
    print("")