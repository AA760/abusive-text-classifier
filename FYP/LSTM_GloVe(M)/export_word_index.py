import pickle
import json
import io


#tokenize
with open(r'.\keras_tokenizer', 'rb') as keras_tokenizer:
    tokenizer = pickle.load(keras_tokenizer)

tokenizer_json = tokenizer.word_index

with io.open('tokenizer.json', 'w', encoding='utf-8') as f:  

      f.write(json.dumps(tokenizer_json, ensure_ascii=False))