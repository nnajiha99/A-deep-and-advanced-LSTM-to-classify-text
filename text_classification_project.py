from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import json
import os

from modules import text_cleaning, lstm_model_creation

#1) Data Loading
CSV_PATH = os.path.join(os.getcwd(), 'dataset','True.csv')
df = pd.read_csv(CSV_PATH)

#2) Data inspection
df.describe()
df.info()
df.head()

# 206 duplicated data here
df.duplicated().sum()

df.drop_duplicates(keep=False, inplace=True)

# to check NaN
df.isna().sum()

print(df['text'][0])

#3) Data cleaning
#things to be removed

for index, temp in enumerate(df['text']):
  df['text'][index] = text_cleaning(temp)

#combined regex pattern
#out = re.sub('bit.ly/\d\w{1,10}|@[^\s]+|^.*?\)\s*-|\[.*?EST\]|[^a-zA-Z]',' ',temp)
#print(out)

#4) Features selection
X = df['text']
y = df['subject']

#5) Data preprocessing

#1. Tokenizer
num_words = 5000  #need to identify via checking the unique words in the sentences
oov_token = '<OOV>'  #out of vocab

tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(X)

#to transform the text using tokenizer --> mms.transform
X = tokenizer.texts_to_sequences(X)

#2. Padding
X = pad_sequences(X, maxlen=200, padding='post', truncating='post')

#3. One hot encoder
#to instantiate
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y[::,None])

#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True,test_size=0.2, random_state=123)

#6. Model development
model = lstm_model_creation(num_words, y.shape[1], dropout=0.4)

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=5)

#7) Model analysis
y_predicted = model.predict(X_test)
y_predicted = np.argmax(y_predicted,axis=1)
y_test = np.argmax(y_test,axis=1)

print(classification_report(y_test, y_predicted))
cm = confusion_matrix(y_test, y_predicted)
disp = ConfusionMatrixDisplay(cm)
disp.plot()

#Model saving

#to save trained model
model.save('model.h5')

#to save one hot encoder model
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe, f)

#tokenizer
token_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    json.dump(token_json, f)