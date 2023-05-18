# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 김신성
- 리뷰어 : 심재형

PRT(PeerReviewTemplate)
----------------------------------------------



### 코드가 정상적으로 동작하고 주어진 문제를 해결했나요? (O)
군더더기 없는 깔끔한 코드들의 구성입니다.<br>
mse가 mae라는 이름의 변수로 설정되어 계산적으로는 문제가 전혀없지만!<br>혼동을 일으킬 수 있기때문에 이거를 수정하는 것도 좋을거같아요!<br>
**mae : 평균 절대 오차, mse: 평균 제곱 오차**
```python
from sklearn.metrics import mean_squared_error
>>> mae = mean_squared_error(df_y_test, y_test_prediction) 
>>> rmse = mae**0.5
>>> print(mae, rmse)
```
<br>

### 주석을 보고 작성자의 코드가 이해되었나요? (O)
```python
#Loss
def MSE(a, b):
    mse = ((a - b) ** 2).mean()  # 두 값의 차이의 제곱의 평균
    return mse
def loss(X, W, b, y):
    predictions = model(X, W, b)
    L = MSE(predictions, y)
    return L
#Gradient
def gradient(X, W, b, y):
    # N은 데이터 포인트의 개수
    N = len(y)
    
    # y_pred 준비
    y_pred = model(X, W, b)
    
    # 공식에 맞게 gradient 계산
    dW = 1/N * 2 * X.T.dot(y_pred - y)
        
    # b의 gradient 계산
    db = 2 * (y_pred - y).mean()
    return dW, db
```
함수 gradient는 수학적으로 이해해야하는 부분을 각각의 알맞은 주석을 입혀 이해가 더 쉽게 됐어요.
<br>

### 코드가 에러를 유발할 가능성이 있나요? (X)
각 태스크 별로 분류 하여 직관적인 코드 사용으로 오류가 발생하더라도 빠른 대응이 가능하기에 가능성이없다.
```python
df_test = df_test.drop('datetime', axis = 'columns')
     
df_test = df_test.drop('holiday',axis = 'columns')
     
df_test = df_test.drop('workingday',axis = 'columns')
```
<br>

### 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
```python
def MSE(a, b):
    mse = ((a - b) ** 2).mean()  # 두 값의 차이의 제곱의 평균
    return mse
def loss(X, W, b, y):
    predictions = model(X, W, b)
    L = MSE(predictions, y)
    return L
#Gradient
def gradient(X, W, b, y):
    # N은 데이터 포인트의 개수
    N = len(y)
    
    # y_pred 준비
    y_pred = model(X, W, b)
    
    # 공식에 맞게 gradient 계산
    dW = 1/N * 2 * X.T.dot(y_pred - y)
        
    # b의 gradient 계산
    db = 2 * (y_pred - y).mean()
    return dW, db
```
각각의 함수들의 동작방식을 정확하게 설명해주셔서 이해가 더 빨랐습니다!

### 코드가 간결한가요? (O)
```python
# Create a figure with multiple subplots
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
sns.countplot(x='month', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Countplot - Month')
sns.countplot(x='day', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Countplot - Day')
```
위 형태에서 subplots를 이용해서 사이즈와 범위를 깔끔하게 설정해주면서 중복되는 코드를 줄이는 좋은 코드를 사용한거같아요.<br>
위처럼 간결하게 표현해서 수정할 때도 용이합니다!

```python
# long code
plt.subplot(3, 2, 1)
sns.countplot(x=df["month"])
plt.title("Countplot - Month")
plt.subplot(3, 2, 2)
sns.countplot(x=df["day"])
plt.title("ountplot - Day")
# short code
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
sns.countplot(x='month', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Countplot - Month')
sns.countplot(x='day', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Countplot - Day')
```

----------------------------------------------

## 참고 링크 및 코드 개선
### 수정된 부분 1
```python
from sklearn.metrics import mean_squared_error
mae = mean_squared_error(df_y_test, y_test_prediction) 
rmse = mae**0.5
print(mae, rmse)
>>>
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df_y_test, y_test_prediction) 
rmse = mse**0.5
print(mse, rmse)
```