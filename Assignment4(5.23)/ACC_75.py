#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install Korpora')


# In[2]:


from Korpora import Korpora


# In[3]:


stop_words = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']


# In[4]:


# 문장 1개를 활용할 딕셔너리와 함께 주면, 단어 인덱스 리스트로 변환해 주는 함수를 만들어 봅시다.
# 단, 모든 문장은 <BOS>로 시작하는 것으로 합니다. 
def get_encoded_sentence(sentence, word_to_index):
    return [word_to_index['<BOS>']]+[word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in sentence.split()]
# 여러 개의 문장 리스트를 한꺼번에 숫자 텐서로 encode해 주는 함수입니다. 
def get_encoded_sentences(sentences, word_to_index):
    return [get_encoded_sentence(sentence, word_to_index) for sentence in sentences]
# 숫자 벡터로 encode된 문장을 원래대로 decode하는 함수입니다. 
def get_decoded_sentence(encoded_sentence, index_to_word):
    index_to_word= dict(map(reversed,index_to_word.items()))
    return ' '.join(index_to_word[index] if index in index_to_word else '<UNK>' for index in encoded_sentence[1:])  #[1:]를 통해 <BOS>를 제외
# 여러 개의 숫자 벡터로 encode된 문장을 한꺼번에 원래대로 decode하는 함수입니다. 
def get_decoded_sentences(encoded_sentences, index_to_word):
    return [get_decoded_sentence(encoded_sentence, index_to_word) for encoded_sentence in encoded_sentences]


# In[5]:


class NLP_basic():
  def __init__(self,stopwords ,corpus = 'korea', data = False):
    self.corpus = 'korea'
    self.data = data
    self.stopwords = stopwords

  def data_loader(self):
    if self.corpus == 'korea':
      corpus_kr = Korpora.load("nsmc")
      df_train = pd.DataFrame(corpus_kr.train)
      df_test = pd.DataFrame(corpus_kr.test)
      df_train.drop_duplicates('text',inplace = True)
      df_test.drop_duplicates('text',inplace = True)
    elif self.data == True:
      pass
    return pd.DataFrame(df_train['text']),pd.DataFrame(df_train['label']), pd.DataFrame(df_test['text']),pd.DataFrame(df_test['label']) 

  def token_generator(self, data, stop_words = stop_words):
    token_list = []
    for x in data:
      word_tokens = okt.morphs(x)
      result = [word for word in word_tokens if not word in stop_words]
      for x in result:
        if x in token_list:
          continue
        else:
          token_list.append(x)
    word_index = {'<PAD>': 0, '<BOS>': 1, '<UNK>': 2}
    for (x,y) in enumerate(token_list):
      word_index[y] = x+3
    return word_index

  def get_encoded_sentence(self, sentence, word_to_index):
    return [word_to_index['<BOS>']]+[word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in sentence.split()]

  def get_encoded_sentences(self, sentences, word_to_index):
    return [get_encoded_sentence(sentence, word_to_index) for sentence in sentences]

  def get_decoded_sentence(self, encoded_sentence, index_to_word):
    index_to_word= dict(map(reversed,index_to_word.items()))
    return ' '.join(index_to_word[index] if index in index_to_word else '<UNK>' for index in encoded_sentence[1:])

  def get_decoded_sentences(self, encoded_sentences, index_to_word):
    return [get_decoded_sentence(encoded_sentence, index_to_word) for encoded_sentence in encoded_sentences]


# In[6]:


sample = NLP_basic(stop_words,'korea')


# In[7]:


import pandas as pd
import gensim as gn
import numpy as np
from collections import Counter
import nltk
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.initializers import Constant


# In[8]:


train_x, train_y, test_x,test_y = sample.data_loader()


# In[9]:


total_data_text = list(train_x['text']) + list(test_x['text'])
# 텍스트데이터 문장길이의 리스트를 생성한 후
num_tokens = [len(tokens) for tokens in total_data_text]
num_tokens = np.array(num_tokens)
# 문장길이의 평균값, 최대값, 표준편차를 계산해 본다. 
print('문장길이 평균 : ', np.mean(num_tokens))
print('문장길이 최대 : ', np.max(num_tokens))
print('문장길이 표준편차 : ', np.std(num_tokens))

# 예를들어, 최대 길이를 (평균 + 2*표준편차)로 한다면,  
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
maxlen = int(max_tokens)
print('pad_sequences maxlen : ', maxlen)
print(f'전체 문장의 {np.sum(num_tokens < max_tokens) / len(num_tokens)}%가 maxlen 설정값 이내에 포함됩니다. ')


# In[10]:


import pickle
with open('word_index.pickle', 'rb') as f:
    word_index = pickle.load(f)


# In[11]:


import os
word2vec_file_path = os.getenv('HOME')+'/data/word2vec_ko.model'
from gensim.models.keyedvectors import Word2VecKeyedVectors
word2vec = Word2VecKeyedVectors.load(word2vec_file_path)
vector = word2vec.wv['끝']


# In[12]:


vocab_size = len(word_index)  # 위 예시에서 딕셔너리에 포함된 단어 개수는 10
word_vector_dim = 100    # 그림과 같이 4차원의 워드 벡터를 가정합니다.

embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=word_vector_dim, mask_zero=True)

# tf.keras.preprocessing.sequence.pad_sequences를 통해 word vector를 모두 일정 길이로 맞춰주어야 
# embedding 레이어의 input이 될 수 있음에 주의해 주세요. 
raw_inputs = np.array(sample.get_encoded_sentences(train_x['text'], word_index), dtype=object)
raw_inputs = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs,
                                                       value=word_index['<PAD>'],
                                                       padding='post',
                                                       maxlen=95)
output = embedding(raw_inputs)


# In[13]:


index_to_word = word_index


# In[14]:


train_x_input = sample.get_encoded_sentences(train_x['text'], word_index)


# In[15]:


x_test = sample.get_encoded_sentences(test_x['text'], word_index)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,
                                                       value=word_index["<PAD>"],
                                                       padding='pre', # 혹은 'pre'
                                                       maxlen=maxlen)
y_test = test_y['label']


# In[16]:


x_train = tf.keras.preprocessing.sequence.pad_sequences(train_x_input,
                                                       value=word_index["<PAD>"],
                                                       padding='pre', # 혹은 'pre'
                                                       maxlen=maxlen)


# In[17]:


# validation set 10000건 분리
x_val = x_train[:10000]   
y_val = train_y[:10000]

# validation set을 제외한 나머지 15000건
partial_x_train = x_train[10000:]  
partial_y_train = train_y[10000:]

print(partial_x_train.shape)
print(partial_y_train.shape)


# In[18]:


vocab_size = 104663    # 어휘 사전의 크기입니다(10,000개의 단어)
word_vector_dim = 100  # 워드 벡터의 차원 수 
embedding_matrix = np.random.rand(vocab_size, word_vector_dim)

# embedding_matrix에 Word2Vec 워드 벡터를 단어 하나씩마다 차례차례 카피한다.
index_to_word2= dict(map(reversed,index_to_word.items()))

for i in range(4,vocab_size):
    if index_to_word2[i] in word2vec.wv:
        embedding_matrix[i] = word2vec.wv[index_to_word2[i]]


# In[19]:


index_to_word2[36]in word2vec.wv


# In[20]:


embedding_matrix.shape


# In[48]:


# # 모델 구성
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Embedding(vocab_size, 
#                                  word_vector_dim, 
#                                  embeddings_initializer=Constant(embedding_matrix),  # 카피한 임베딩을 여기서 활용
#                                  input_length=maxlen, 
#                                  trainable=True))   # trainable을 True로 주면 Fine-tuning
# model.add(tf.keras.layers.Conv1D(512, 25, activation='relu'))
# model.add(tf.keras.layers.MaxPooling1D(2))
# model.add(tf.keras.layers.Conv1D(512, 25, activation='relu'))
# model.add(tf.keras.layers.GlobalMaxPooling1D())
# model.add(tf.keras.layers.Dense(256, activation='relu'))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid')) 

# model.summary()


# In[54]:


# model 설계 - 딥러닝 모델 코드를 직접 작성해 주세요.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 
                                 word_vector_dim, 
                                 embeddings_initializer=Constant(embedding_matrix),  # 카피한 임베딩을 여기서 활용
                                 input_length=maxlen, 
                                 trainable=True))   # trainable을 True로 주면 Fine-tuning
model.add(tf.keras.layers.LSTM(1024))   # 가장 널리 쓰이는 RNN인 LSTM 레이어를 사용하였습니다. 이때 LSTM state 벡터의 차원수는 8로 하였습니다. (변경 가능)
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # 최종 출력은 긍정/부정을 나타내는 1dim 입니다.

model.summary()


# In[23]:


embedding_matrix.shape


# In[55]:


# 학습의 진행
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
epochs=25  # 몇 epoch를 훈련하면 좋을지 결과를 보면서 바꾸어 봅시다. 

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=epochs,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# In[56]:


# 테스트셋을 통한 모델 평가
results = model.evaluate(x_test,  y_test, verbose=2)

print(results)


# In[57]:


history_dict = history.history
print(history_dict.keys()) # epoch에 따른 그래프를 그려볼 수 있는 항목들


# In[58]:


import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo"는 "파란색 점"입니다
plt.plot(epochs, loss, 'bo', label='Training loss')
# b는 "파란 실선"입니다
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[59]:


plt.clf()   # 그림을 초기화합니다

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[ ]:




