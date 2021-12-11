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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from imblearn import under_sampling
from imblearn import over_sampling
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv(r"C:\Users\user\Documents\Python\FYP\FYP\processed_train.csv",sep=',')
y = data.abusive


#class distribution
print("Original class distribution:")
print(y.value_counts())
print("")

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

#remove punctuation
data['nopunc'] = data['comment_text'].apply(lambda x:remove_punctuation(x))
data.head()

#all lowercase
data['lowcase'] = data['nopunc'].apply(lambda x: x.lower())

#remove stopwords
data['nostop'] = data['lowcase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#remove numbers
data['nonum'] = data['nostop'].str.replace('\d+', '',regex=True)

#tokenize + lemmatization
data['lemmatized'] = data.nonum.apply(lemmatize_text)

#untokenize
data['untokenized'] = data.lemmatized.apply(lambda x:TreebankWordDetokenizer().detokenize(x))

#vectorize - keep increasing features from 50k
tfidf = TfidfVectorizer(max_features=110000,max_df=0.5,min_df=1,ngram_range=(1,2))
features = tfidf.fit_transform(data.untokenized)
#c = pd.DataFrame(features.todense(),columns=tfidf.get_feature_names_out())
#print(c)

#undersample = under_sampling.RandomUnderSampler(sampling_strategy=1.0)
#features, y = undersample.fit_resample(features,y)
oversample = over_sampling.RandomOverSampler(sampling_strategy=1.0)
features, y = oversample.fit_resample(features,y)
print("Oversampled class distribution:")
print(y.value_counts())
print("")
print("Data prepared. Training...")
print("")


classifier = RandomForestClassifier(n_estimators=100, random_state=0,n_jobs=8)
classifier.fit(features, y.values.ravel())
print("Model trained. Exporting...")
print("")

#y_pred = classifier.predict(X_test)
#print("Testing complete.")

#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
#print(accuracy_score(y_test, y_pred))

with open(r'C:\Users\user\Documents\Python\FYP\FYP\Pickle\text_classifier', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)
    print("Exported model.")
    print("")

with open(r'C:\Users\user\Documents\Python\FYP\FYP\Pickle\text_vectorizer', 'wb') as picklefile:
    pickle.dump(tfidf,picklefile)
    print("Exported vectorizer.")

#with open('text_classifier', 'rb') as training_model:
#    model = pickle.load(training_model)
#    print("Imported model. Testing...")

#y_pred2 = model.predict(X_test)
#print("Testing complete...")

#print(confusion_matrix(y_test, y_pred2))
#print(classification_report(y_test, y_pred2))
#print(accuracy_score(y_test, y_pred2))