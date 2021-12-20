import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.tokenize.treebank import TreebankWordDetokenizer
from keras.models import load_model


#d = {'comment_text': [input("Enter text:")]}
#input = pd.DataFrame(data=d)
input = input("Enter text:")

#def remove_punctuation(text):
#    punctuationfree="".join([i for i in text if i not in string.punctuation])
#    return punctuationfree

#def lemmatize_text(text):
#    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
#    lemmatizer = nltk.stem.WordNetLemmatizer()
#    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


##remove punctuation
#input = input.apply(lambda x:remove_punctuation(x))

##all lowercase
#input = input.apply(lambda x: x.lower())

#remove stopwords
#input = input.apply(lambda x: ' '.join([word for word in x.str.split() if word not in (stop)]))

##remove numbers
#input = input.str.replace('\d+', '',regex=True)

#tokenize + lemmatization
#input = input.apply(lemmatize_text)

#untokenize
#input = input.apply(lambda x:TreebankWordDetokenizer().detokenize(x))

##tokenize
#with open(r'C:\Users\abdul\OneDrive\Documents\Python\FYP\FYP\Pickle\keras_tokenizer', 'rb') as keras_tokenizer:
#    tokenizer = pickle.load(keras_tokenizer)

#features = tokenizer.texts_to_sequences(input)
#features_padded = pad_sequences(features, maxlen=200, padding="post", truncating="post")


#tokenize
input = nltk.tokenize.WhitespaceTokenizer().tokenize(input)

#remove stopwords
input = [w for w in input if not w.lower() in stop]

#lemmatize
input = ' '.join([nltk.stem.WordNetLemmatizer().lemmatize(words) for words in input])
print(input)

#convert to pd dataframe
input = pd.DataFrame({'comment_text': [input]})


##testing the model
model = load_model('nn_model/')

prediction = model.predict(input)
prediction = [1 if p > 0.5 else 0 for p in prediction]

print(prediction)


