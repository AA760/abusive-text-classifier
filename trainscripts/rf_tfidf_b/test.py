import pandas as pd
import sklearn
import nltk
import string
from sklearn.metrics import  confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
tokenizer = nltk.tokenize.WhitespaceTokenizer()
detokenizer = nltk.tokenize.treebank.TreebankWordDetokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

#import data
psgdb = psycopg2.connect(host="localhost", database="datasets", user="postgres", password="atc9310")

c = psgdb.cursor()
c.execute("SELECT comment_text, abusive FROM bTest")

table = c.fetchall()
c.close()

data = pd.DataFrame(table, columns =['comment_text', 'abusive'])
data['abusive'] = data['abusive'].replace(True,1)
data['abusive'] = data['abusive'].replace(False,0)

data.sample(frac=1)
test_X = data.comment_text
test_y = data.abusive


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

test_X = clean(test_X)


#vectorize
with open(r'tfidf_vectorizer.pickle', 'rb') as tfidf:
    tfidf = pickle.load(tfidf)
    print("Imported vectorizer.")
    print("")
test_X = tfidf.transform(test_X)


#predict with model
with open(r'model.pickle', 'rb') as model:
    model = pickle.load(model)
    print("Imported model. Testing...")
    print("")
y_preds = model.predict(test_X)
print("Finished predicting.")
print("")


#confusion matrix
cm = confusion_matrix(test_y, y_preds)
ax = sns.heatmap(cm, annot=True, cmap='Blues',fmt='g')

ax.set_title('Confusion Matrix - Random Forest w/ tfidf',weight='bold');
ax.set_xlabel('\nPredicted Values',weight='bold')
ax.set_ylabel('Actual Values\n',weight='bold');

ax.xaxis.set_ticklabels(['Not Abusive','Abusive'])
ax.yaxis.set_ticklabels(['Not Abusive','Abusive'])

plt.show()

#f1 scores
f10 = f1_score(test_y, y_preds, average='binary',pos_label=0)
f11 = f1_score(test_y, y_preds, average='binary',pos_label=1)
print('F1 score (Not Abusive): %f' % f10)
print('F1 score (Abusive): %f' % f11)
print('')

#other performance metrics
precision = precision_score(np.array(test_y), y_preds, average='binary')
print('Precision (Abusive): %f' % precision)

recall = recall_score(np.array(test_y), y_preds, average='binary')
print('Recall (Abusive): %f' % recall)

accuracy = accuracy_score(np.array(test_y), y_preds)
print('Overall accuracy: %f' % accuracy)