# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 김신성
- 리뷰어 : 심재형

PRT(PeerReviewTemplate)
----------------------------------------------



### 코드가 정상적으로 동작하고 주어진 문제를 해결했나요? (O)
전부 정상적으로 진행됩니다.<br>
![image](https://github.com/GrainSack/Assignments/assets/65104209/28f8ca83-80a5-46fe-b399-03db16538373)

<br>다양한 모델을 만들어 loss와 accuracy 그래프를 각각 비교 확인함으로써 어떤 방식이 더 나은 결과인지 잘 확인했습니다~!<br>
![image](https://github.com/GrainSack/Assignments/assets/65104209/60939c8d-4264-4895-b2a5-8e1c659c281e)<br>
padding값에 pre를 주면서 post보다는 더 정확도가 높은 방법을 채용한 부분도 아주 잘하신거같아요
### 주석을 보고 작성자의 코드가 이해되었나요? (O)
```python
vocab_size = len(word_index)    # 어휘 사전의 크기입니다(10,000개의 단어)
word_vector_dim = 16  # 워드 벡터의 차원 수 (변경 가능한 하이퍼파라미터)

# model 설계 - 딥러닝 모델 코드를 직접 작성해 주세요.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
model.add(tf.keras.layers.LSTM(8))   # 가장 널리 쓰이는 RNN인 LSTM 레이어를 사용하였습니다. 이때 LSTM state 벡터의 차원수는 8로 하였습니다. (변경 가능)
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # 최종 출력은 긍정/부정을 나타내는 1dim 입니다.

model.summary()
     
```
코드를 읽을 때 각각의 항목별로 자세한 주석을통해 조금 더 간편하게 이해가 빨라서 좋았습니다

<br>

### 코드가 에러를 유발할 가능성이 있나요? (X)
코드 자체에는 에러를 유발할 가능성이 보이지 않습니다.
```python
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
    word_index = {'': 0, '': 1, '': 2}
    for (x,y) in enumerate(token_list):
      word_index[y] = x+3
    return word_index

  def get_encoded_sentence(self, sentence, word_to_index):
    return [word_to_index['']]+[word_to_index[word] if word in word_to_index else word_to_index[''] for word in sentence.split()]

  def get_encoded_sentences(self, sentences, word_to_index):
    return [get_encoded_sentence(sentence, word_to_index) for sentence in sentences]

  def get_decoded_sentence(self, encoded_sentence, index_to_word):
    index_to_word= dict(map(reversed,index_to_word.items()))
    return ' '.join(index_to_word[index] if index in index_to_word else '' for index in encoded_sentence[1:])

  def get_decoded_sentences(self, encoded_sentences, index_to_word):
    return [get_decoded_sentence(encoded_sentence, index_to_word) for encoded_sentence in encoded_sentences]
```

다양하게 쓰일 수 있는 함수들을 클래스화 하면서 사용성, 오류를 빠르게 확인가능하기에 저도 배우고갑니다!<br>
다만 다른 부분에선 주석이 자세히 적혀있어 이해가 빨랐는데 이 부분에서도 한 줄 주석으로 함수를 설명해 주신다면 더 좋을 거 같습니다

### 코드 작성자가 코드를 제대로 이해하고 작성했나요? (O)
```python
vocab_size = len(word_index)    # 어휘 사전의 크기입니다(10,000개의 단어)
word_vector_dim = 16  # 워드 벡터의 차원 수 (변경 가능한 하이퍼파라미터)

# model 설계 - 딥러닝 모델 코드를 직접 작성해 주세요.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
model.add(tf.keras.layers.LSTM(8))   # 가장 널리 쓰이는 RNN인 LSTM 레이어를 사용하였습니다. 이때 LSTM state 벡터의 차원수는 8로 하였습니다. (변경 가능)
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # 최종 출력은 긍정/부정을 나타내는 1dim 입니다.

model.summary()
```
어떤 레이어를 써야할지를 정확히 알고있으셔서 모델을 완벽히 이해중이십니다! 
### 코드가 간결한가요? (O)

![image](https://github.com/GrainSack/Assignments/assets/65104209/3e5c00cf-c3a1-47db-a51e-cd352934798c)
<br> 직관상 그래프를 sns.subplots을 이용하면서 보는 입장에서 더 나은 가독성을 제공한부분 멋집니다

----------------------------------------------

## 참고 링크 및 코드 개선
