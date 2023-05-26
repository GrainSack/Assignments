# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 김신성
- 리뷰어 : 심재형

PRT(PeerReviewTemplate)
----------------------------------------------

### 코드가 정상적으로 동작하고 주어진 문제를 해결했나요? (세모)
전부 정상적으로 진행됩니다.<br>
```python
data = data[['text','headlines']]
data.head()
```
<br>text와 headline의 순서를 바꾼 방식도 개발 직관성때문에 좋은선택인것 같습니다!
![image](https://github.com/GrainSack/Assignments/assets/65104209/4fd9da74-9b5a-4e50-8270-dbbf4b4300b1)
<br>train test epoch별 loss그래프도 잘 나왔습니다!

**조금 아쉬운점**
```python
import requests
from summa.summarizer import summarize

print('Summary:')
for i in text:
  print(summarize(i, words=10, split = True))
```
![image](https://github.com/GrainSack/Assignments/assets/65104209/3c52cee7-fbfb-4ae3-991a-deb4fa969cac)
<br> 조금 아쉬운점은 summary함수를 사용했을 때 빈값을 리턴하였는데 이부분에 대해서는 아마 아래원인 때문에 빈값을 리턴한것 같습니다!<br>
![image](https://github.com/GrainSack/Assignments/assets/65104209/667a9d9c-3287-4aa1-9ba9-cb35611a22cb)<br>
![image](https://github.com/GrainSack/Assignments/assets/65104209/53aeff40-1c69-449d-aa7d-9a0880d7a789)
<br>한번 내용을 위 사진에 맞게 수정을 해보시는걸 추천드립니다!
```python
# ex
# 정규화 처리가 진행되지 않은 데이터를 한번 이용해보세요!

# word값을 한번 올려보시는것도 추천드립니다!
for i in text:
  print(summarize(i, words=50, split = True))
```
### 주석을 보고 작성자의 코드가 이해되었나요? (O)
```python
# 데이터 전처리 함수
def preprocess_sentence(sentence, remove_stopwords=True):
    sentence = sentence.lower() # 텍스트 소문자화
    sentence = BeautifulSoup(sentence, "lxml").text # ,  등의 html 태그 제거
    sentence = re.sub(r'
', '', sentence) # 괄호로 닫힌 문자열 (...) 제거 Ex) my husband (and myself!) for => my husband for
    sentence = re.sub('"','', sentence) # 쌍따옴표 " 제거
    sentence = ' '.join([contractions[t] if t in contractions else t for t in sentence.split(" ")]) # 약어 정규화
    sentence = re.sub(r"'s\b","", sentence) # 소유격 제거. Ex) roland's -> roland
    sentence = re.sub("[^a-zA-Z]", " ", sentence) # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
    sentence = re.sub('[m]{2,}', 'mm', sentence) # m이 3개 이상이면 2개로 변경. Ex) ummmmmmm yeah -> umm yeah
    
    # 불용어 제거 (Text)
    if remove_stopwords:
        tokens = ' '.join(word for word in sentence.split() if not word in stopwords.words('english') if len(word) > 1)
    # 불용어 미제거 (Summary)
    else:
        tokens = ' '.join(word for word in sentence.split() if len(word) > 1)
    return tokens
```
<br>일반적으로 정규식방식은 처음보는 사람의 경우 이해하기 힘들 수 있지만 각각의 줄마다 주석을 자세하게 적음으로 써 사용자 입장에서
정말 이해하기 편합니다!

### 코드가 에러를 유발할 가능성이 있나요? (X)
코드 자체에는 에러를 유발할 가능성이 보이지 않습니다.
```python
# 원문의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq2text(input_seq):
    temp=''
    for i in input_seq:
        if (i!=0):
            temp = temp + src_index_to_word[i]+' '
    return temp

# 요약문의 정수 시퀀스를 텍스트 시퀀스로 변환
def seq2summary(input_seq):
    stop_condition = False
    temp=''
    for i in input_seq:
        if (i!=0) and (i !='eostoken') :
            temp = temp + src_index_to_word[i]+' '
        if (i == 'sostoken'  or len(temp.split()) >= (summary_max_len-1)):
          stop_condition = True
    return temp
```

<br> 위에 작성한 코드처럼 각 함수 내에 예외처리를 깔끔하게 예외처리 되었습니다!

### 코드 작성자가 코드를 제대로 이해하고 작성했나요? (O)
```python
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
history = model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train, \
          validation_data=([encoder_input_test, decoder_input_test], decoder_target_test), \
          batch_size=256, callbacks=[es], epochs=50)
```
<br>EarlyStopping patience 부분에대해서 2로 설정한 부분에대해서 loss업데이트 스택, 업데이트가 2번 진행되지않으면 학습을 중단하는 부분을 설명을 매우 잘해주셨습니다..! **ꉂꉂ(ᵔᗜᵔ*)**
### 코드가 간결한가요? (O)
```python
encoder_input_train = pad_sequences(encoder_input_train, maxlen=text_max_len, padding='post')
encoder_input_test = pad_sequences(encoder_input_test, maxlen=text_max_len, padding='post')
decoder_input_train = pad_sequences(decoder_input_train, maxlen=summary_max_len, padding='post')
decoder_target_train = pad_sequences(decoder_target_train, maxlen=summary_max_len, padding='post')
decoder_input_test = pad_sequences(decoder_input_test, maxlen=summary_max_len, padding='post')
decoder_target_test = pad_sequences(decoder_target_test, maxlen=summary_max_len, padding='post')
```
전체적인 함수화가 진행되어있기 때문에 재사용성이 높고 훨씬 간결해 보입니다.<br>
**매우매우 조금 아쉬운점**
padding값에 "post" 보다는 "pre"를 사용하는 점이 더 좋은 결과를 뽑을 수 있습니다!<br>
![image](https://github.com/GrainSack/Assignments/assets/65104209/ae6a90c9-cd07-4dea-8cef-5f1306ba9a01)
<br>무조건 정답은 아니지만 이러한경우로 사용된다는 점도 알아시면 좋을거같아요!

----------------------------------------------

## 참고 링크 및 코드 개선
