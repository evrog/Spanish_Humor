#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import sklearn, tensorflow.keras
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, GRU, SimpleRNN, Conv1D,MaxPooling1D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import copy
import tensorflow.keras.backend as K


# In[ ]:


import numpy as  np
import random
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.stem import SnowballStemmer

word2vec = r"C:\Users\Annie\Documents\Working\Spanish jokes\SBW-vectors-300-min5.txt"
train=r"C:\Users\Annie\Documents\Working\Spanish jokes\data\haha_2019_train_preprocessed_lemmatized.csv"
test=r"C:\Users\Annie\Documents\Working\Spanish jokes\data\haha_2019_test_preprocessed_lemmatized.csv"

lda_train=r"C:\Users\Annie\Documents\Working\Spanish jokes\ALL_DATA\lda+pos_neg\train.csv"
lda_test=r"C:\Users\Annie\Documents\Working\Spanish jokes\ALL_DATA\lda+pos_neg\test.csv"
lda_ev=r"C:\Users\Annie\Documents\Working\Spanish jokes\ALL_DATA\lda+pos_neg\ev.csv"

emb_train=r"C:\Users\Annie\Documents\Working\Spanish jokes\ALL_DATA\embeddings\train.csv"
emb_test=r"C:\Users\Annie\Documents\Working\Spanish jokes\ALL_DATA\embeddings\test.csv"
emb_ev=r"C:\Users\Annie\Documents\Working\Spanish jokes\ALL_DATA\embeddings\ev.csv"

em_train=r"C:\Users\Annie\Documents\Working\Spanish jokes\ALL_DATA\emotions\train.csv"
em_test=r"C:\Users\Annie\Documents\Working\Spanish jokes\ALL_DATA\emotions\test.csv"
em_ev=r"C:\Users\Annie\Documents\Working\Spanish jokes\ALL_DATA\emotions\ev.csv"

other_train=r"C:\Users\Annie\Documents\Working\Spanish jokes\ALL_DATA\other_features\train.csv"
other_test=r"C:\Users\Annie\Documents\Working\Spanish jokes\ALL_DATA\other_features\test.csv"
other_ev=r"C:\Users\Annie\Documents\Working\Spanish jokes\ALL_DATA\other_features\ev.csv"


# <h2>text preprocessing

# In[ ]:


stemmer = SnowballStemmer('spanish')

def clean_text(text):
    text = text.replace("\\", " ").replace(u"╚", " ").replace(u"╩", " ")
    text = text.lower()
    text = re.sub('\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text) 
    text = re.sub('[¡¿.,:;_%©?*,!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)
    words = text.split()
    words = [w for w in words if len(w)>=3]
    stop_words = set(stopwords.words('spanish'))
    words = [w for w in words if not w in stop_words]
    text=' '.join(words)
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(i) for i in tokens]
    return stemmed


# <h2>data loading

# In[ ]:


#texts_train
values= pd.read_csv(train, sep=',', header=None, encoding = 'utf-8-sig').values
np.random.seed(42)
np.random.shuffle(values)
#df=pd.DataFrame(values)

m = len(values)

train_length = int(0.9 * m)
train_data, test_data = values[:train_length], values[train_length:]

df=pd.DataFrame(train_data)

texts_train=df[1].tolist()
scores_train=df[9].tolist()
categories_train_raw = [1 if str(s)!='nan' else 0 for s in scores_train]

df=pd.DataFrame(test_data)

texts_test=df[1].tolist()
texts_test_original=df[1].tolist()
scores_test=df[9].tolist()
categories_test_raw = [1 if str(s)!='nan' else 0 for s in scores_test]


df=pd.read_csv(test, sep=',', header=None, encoding = 'utf-8-sig')
texts_ev=df[1].tolist()
#texts_train= pd.read_csv(texts_ov, sep=';', header=None, encoding = 'utf-8-sig')[0].tolist()
#categories_train_row= pd.read_csv(texts_ov, sep=';', header=None, encoding = 'utf-8-sig')[0].tolist()


# In[ ]:


#lda
lda_scaler = MinMaxScaler()
lda_values_train= pd.read_csv(lda_train, sep=';', header=None, encoding = 'utf-8-sig').drop([0,1], axis=1).values
lda_values_train=lda_scaler.fit_transform(lda_values_train)
lda_values_test= pd.read_csv(lda_test, sep=';', header=None, encoding = 'utf-8-sig').drop([0,1], axis=1).values
lda_values_test=lda_scaler.transform(lda_values_test)
lda_values_ev= pd.read_csv(lda_ev, sep=';', header=None, encoding = 'utf-8-sig').drop([0,1], axis=1).values
lda_values_ev=lda_scaler.transform(lda_values_ev)


# In[ ]:


#embeddings
emb_scaler = MinMaxScaler()
emb_values_train= pd.read_csv(emb_train, sep=';', header=None, encoding = 'utf-8-sig').drop([0,1], axis=1).values
emb_values_train=emb_scaler.fit_transform(emb_values_train)
emb_values_test= pd.read_csv(emb_test, sep=';', header=None, encoding = 'utf-8-sig').drop([0,1], axis=1).values
emb_values_test=emb_scaler.transform(emb_values_test)
emb_values_ev= pd.read_csv(emb_ev, sep=';', header=None, encoding = 'utf-8-sig').drop([0,1], axis=1).values
emb_values_ev=emb_scaler.transform(emb_values_ev)


# In[ ]:


#emotions
em_scaler = MinMaxScaler()
em_values_train= pd.read_csv(em_train, sep=';', header=None, encoding = 'utf-8-sig').drop([0,1], axis=1).values
em_values_train=em_scaler.fit_transform(em_values_train)
em_values_test= pd.read_csv(em_test, sep=';', header=None, encoding = 'utf-8-sig').drop([0,1], axis=1).values
em_values_test=em_scaler.transform(em_values_test)
em_values_ev= pd.read_csv(em_ev, sep=';', header=None, encoding = 'utf-8-sig').drop([0,1], axis=1).values
em_values_ev=em_scaler.transform(em_values_ev)


# In[ ]:


#other
other_scaler = MinMaxScaler()
other_values_train= pd.read_csv(other_train, sep=';', header=None, encoding = 'utf-8-sig').drop([0,1], axis=1).values
other_values_train=other_scaler.fit_transform(other_values_train)
other_values_test= pd.read_csv(other_test, sep=';', header=None, encoding = 'utf-8-sig').drop([0,1], axis=1).values
other_values_test=other_scaler.transform(other_values_test)
other_values_ev= pd.read_csv(other_ev, sep=';', header=None, encoding = 'utf-8-sig').drop([0,1], axis=1).values
other_values_ev=other_scaler.transform(other_values_ev)


# <h2>tfidf

# In[ ]:


sw_list=stopwords.words('spanish')
vectorizer = TfidfVectorizer(max_features=5000)

for i in range(len(texts_train)):
    texts_train[i]=' '.join(clean_text(texts_train[i]))
for i in range(len(texts_test)):
    texts_test[i]=' '.join(clean_text(texts_test[i]))
for i in range(len(texts_ev)):
    texts_ev[i]=' '.join(clean_text(texts_ev[i]))
    

tfidf_train = vectorizer.fit_transform(texts_train).toarray()
tfidf_test = vectorizer.transform(texts_test).toarray()
tfidf_ev = vectorizer.transform(texts_ev).toarray()


# <h2> embedding matrix

# In[ ]:


words=set()#set of all words
for text in texts_train:
    words_text=text.split();
    words.update(words_text)
for text in texts_test:
    words_text=text.split();
    words.update(words_text)
for text in texts_ev:
    words_text=text.split();
    words.update(words_text)
print("number of words: {0}".format(len(words)))


# In[ ]:


embdict=dict()
index=0

with open(word2vec,'r',encoding = 'utf-8-sig')as f:
    header = f.readline()
    vocab_size, layer1_size = map(int, header.split())
    binary_len = np.dtype('float32').itemsize * layer1_size
    for line in range(vocab_size):
        word=str(f.readline()).replace('b','').replace('\'','').replace('\\n','').lower().split()
        w = stemmer.stem(word[0])
        if w in words:
            word.remove(word[0])
            emb = [float(x) for x in word]
            embdict[str(w)]=emb
        index+=1
        if index%100000==0:
            print("iteration "+str(index))

print("size of dictionary: {0}".format(len(embdict)))
del(words)


# In[ ]:


MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 300


# In[ ]:


tokenizer=Tokenizer()
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(texts_train+texts_test+texts_ev)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[ ]:


embedding_matrix = np.zeros((len(word_index), EMBEDDING_DIM))

for word, i in tokenizer.word_index.items():
    try:
        embedding_vector = embdict[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    except:
        pass
del(embdict)


# In[ ]:


texts_train = tokenizer.texts_to_sequences(texts_train)
texts_train = sequence.pad_sequences(texts_train, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', texts_train.shape)
texts_test = tokenizer.texts_to_sequences(texts_test)
texts_test = sequence.pad_sequences(texts_test, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', texts_test.shape)
texts_ev = tokenizer.texts_to_sequences(texts_ev)
texts_ev = sequence.pad_sequences(texts_ev, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', texts_ev.shape)


# <h2>model

# In[ ]:


num_classes=2
categories_train = tf.keras.utils.to_categorical(categories_train_raw, num_classes)
categories_test = tf.keras.utils.to_categorical(categories_test_raw, num_classes)


# In[ ]:


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[ ]:


input_cnn=Input(shape=(MAX_SEQUENCE_LENGTH,))
input_tfidf=Input(shape=(len(tfidf_train[0]),))
input_em=Input(shape=(len(em_train[0]),))
input_lda=Input(shape=(len(lda_train[0]),))
input_other=Input(shape=(len(other_train[0]),))

#cnn
x = Embedding(29531, 300, input_length=250, trainable=True, name='300')(input_cnn)
x=Conv1D(64,5,padding='same')(x)
x=MaxPooling1D(pool_size = (20), strides=(10))(x)
x=Conv1D(64,5,padding='same', name='64_1D')(x)
x=MaxPooling1D(pool_size = (20), strides=(10), name='20')(x)
x=Flatten(name='flatten')(x)
x = Model(inputs=input_cnn, outputs=x)#64

#tfidf 5000
y = Dense(1024, activation='relu', name='1024')(input_tfidf)
#y = Dropout(0.1)(y)
y = Dense(256, activation='relu')(y)
#y = Dropout(0.1)(y)
y = Dense(64, activation='relu', name='64')(y)
#y = Dropout(0.1)(y)
y = Model(inputs=input_tfidf, outputs=y)#64

#emotions 63
y2 = Dense(32, activation='relu', name='32')(input_em)
#y2 = Dropout(0.1)(y2)
y2 = Dense(16, activation='relu', name='16')(y2)
#y2 = Dropout(0.1)(y2)
y2 = Model(inputs=input_em, outputs=y2)#16

#lda 12
y3 = Dense(8, activation='relu')(input_lda)
#y3 = Dropout(0.1)(y3)
y3 = Dense(8, activation='relu')(y3)
#y3 = Dropout(0.1)(y3)
y3 = Model(inputs=input_lda, outputs=y3)#8

combined=concatenate([x.output, y.output, y2.output, input_other], name='concat')#216
z=Dense(128, activation='relu', name='128')(combined)
#z = Dropout(0.1)(z)
z=Dense(64, activation='relu')(z)
z = Dropout(0.8)(z)
z=Dense(2, activation='softmax', name='2')(z)

model = tensorflow.keras.models.Model(inputs=[input_cnn, input_tfidf, input_em, input_other], outputs=z)

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  )

model.summary()


# In[ ]:


model.fit([texts_train, np.array(tfidf_train), np.array(emb_values_train), np.array(em_values_train), np.array(lda_values_train), np.array(other_values_train)],
          categories_train, epochs=1, 
          verbose=1, 
          validation_data=([texts_test, np.array(tfidf_test), np.array(emb_values_test), np.array(em_values_test), np.array(lda_values_test), np.array(other_values_test)], categories_test)
         )


# In[ ]:


predict = np.argmax(model.predict([np.array(texts_test),np.array(values2), np.array(tfidf_test)]), axis=1)
answer = np.argmax(categories_test, axis=1)
print('F1-score: %f' % (f1_score(predict, answer, average="macro")*100))


# In[ ]:


predict = np.argmax(model.predict([np.array(texts_ev),np.array(values3),np.array(tfidf_ev)]), axis=1)
print(predict)
with open('prediction_cnn1.txt', 'w', encoding='utf-8') as file:
    for p in predict:
        print(str(p),file=file)


# In[ ]:


from tensorflow.keras.models import load_model
model.save('cnn.h5')


# In[ ]:


from tensorflow.keras.models import load_model
model = load_model('cnn.h5')

