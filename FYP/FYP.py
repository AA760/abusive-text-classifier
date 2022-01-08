import pandas as pd
from pandas.io.parsers import TextFileReader
import sklearn
import nltk
nltk.download('wordnet')
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
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest

data = pd.read_csv(r"C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\processed_train.csv",sep=',')
y = data.abusive

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

#remove punctuation
data.comment_text = data.comment_text.apply(lambda x:remove_punctuation(x))
data.head()

#all lowercase
data.comment_text = data.comment_text.apply(lambda x: x.lower())

##remove stopwords
#data.comment_text = data.comment_text.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#remove numbers
data.comment_text = data.comment_text.str.replace('\d+', '',regex=True)

#tokenize + lemmatization
data.comment_text = data.comment_text.apply(lemmatize_text)

#untokenize
data.comment_text = data.comment_text.apply(lambda x:TreebankWordDetokenizer().detokenize(x))

#tokenize + vectorize
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
features = tfidf.fit_transform(data.comment_text)

#feature selection
selector = SelectKBest(k=min(20000, features.shape[1]))
selector.fit(features, y)
features = selector.transform(features)

#oversampling
oversample = over_sampling.RandomOverSampler(sampling_strategy=1.0)
features, y = oversample.fit_resample(features,y)
print("Data prepared. Training...")
print("")


classifier = RandomForestClassifier(n_estimators=300, random_state=0,n_jobs=6)
#classifier = MLPClassifier(hidden_layer_sizes=(128,64,32), random_state=0, max_iter=50, activation='logistic', solver='adam', verbose=True)
classifier.fit(features, y.values.ravel())
print("Model trained. Exporting...")
print("")

#y_pred = classifier.predict(X_test)
#print("Testing complete.")

#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
#print(accuracy_score(y_test, y_pred))

with open(r'C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\Pickle\text_classifier', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)
    print("Exported model.")
    print("")

with open(r'C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\Pickle\text_vectorizer', 'wb') as picklefile:
    pickle.dump(tfidf,picklefile)
    print("Exported vectorizer.")
    print("")

with open (r'C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\Pickle\feature_selector', 'wb') as picklefile:
    pickle.dump(selector,picklefile)
    print("Exported feature selector.")

#with open('text_classifier', 'rb') as training_model:
#    model = pickle.load(training_model)
#    print("Imported model. Testing...")

#y_pred2 = model.predict(X_test)
#print("Testing complete...")

#print(confusion_matrix(y_test, y_pred2))
#print(classification_report(y_test, y_pred2))
#print(accuracy_score(y_test, y_pred2))