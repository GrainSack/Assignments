# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 김신성
- 리뷰어 : 심재형

PRT(PeerReviewTemplate)
----------------------------------------------



### 코드가 정상적으로 동작하고 주어진 문제를 해결했나요? (O)
전부 정상적으로 진행됩니다.<br>
```python
img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_bgr[refined_y:refined_y +img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = np.where(img_sticker!=0,sticker_area,img_add).astype(np.uint8)
plt.imshow(img_bgr) # rgb만 적용해놓은 원본 이미지에 왕관 이미지를 덮어 씌운 이미지가 나오게 된다.
plt.show()
```
<br>알파 값을 찾는 과정을통해 이미지 필터가 불투명도를 조절하는 효과를 구현하기위한 방법들이 정말 좋은거같습니다!
### 주석을 보고 작성자의 코드가 이해되었나요? (O)
```python
# 우리는 현재 이마 자리에 왕관을 두고 싶은건데, 이마위치 - 왕관 높이를 했더니 이미지의 범위를 초과하여 음수가 나오는 것
# opencv는 ndarray데이터를 사용하는데, ndarray는 음수인덱스에 접근 불가하므로 스티커 이미지를 잘라 줘야 한다.
# 왕관 이미지가 이미지 밖에서 시작하지 않도록 조정이 필요함
# 좌표 순서가 y,x임에 유의한다. (y,x,rgb channel)
# 현재 상황에서는 -y 크기만큼 스티커를 crop 하고, top 의 x좌표와 y 좌표를 각각의 경우에 맞춰 원본 이미지의 경계 값으로 수정하면 아래와 같은 형식으로 나옵니다.
# 음수값 만큼 왕관 이미지(혹은 추후 적용할 스티커 이미지)를 자른다.
if refined_x < 0: 
    img_sticker = img_sticker[:, -refined_x:]
    refined_x = 0
# 왕관 이미지를 씌우기 위해 왕관 이미지가 시작할 y좌표 값 조정
if refined_y < 0:
    img_sticker = img_sticker[-refined_y:, :] # refined_y가 -98이므로, img_sticker[98: , :]가 된다. (187, 187, 3)에서 (89, 187, 3)이 됨 (187개 중에서 98개가 잘려나감)
    refined_y = 0
```
어려운코드위에 상당히 많은 주석으로 실행 flow대로 설명하는 방법도 차례차례 읽으니 이해가 잘됐습니다.

<br>

### 코드가 에러를 유발할 가능성이 있나요? (세모)
코드 자체에는 에러가 없고 존윅이 상당히 커엽습니다<br>
```python
img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_bgr[refined_y:refined_y +img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = np.where(img_sticker!=0,sticker_area,img_add).astype(np.uint8)
plt.imshow(img_bgr) # rgb만 적용해놓은 원본 이미지에 왕관 이미지를 덮어 씌운 이미지가 나오게 된다.
plt.show()
```
jpeg는 알파채널이 존재하지않기때문에 확장자 관련 예외처리도 해놓으면 대응하기 쉬울거같아요!
### 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기) (O)
![image](https://github.com/GrainSack/Assignments/assets/65104209/476ecb52-72b0-4ee3-98f1-8fd801292503)<br>
직접 작성한 코드에 대해 확실히 이해하고있어서 여러 문제점들에 대해 분석방법이 좋았습니다.<br>
문제점 뿐만아니라 개선방법까지 나열함으로 써 확실한 방향성또한 잘 이해되었습니다!

### 코드가 간결한가요? (X)
추후에 다른 여러 이미지를 적용시키게 되면 계속 다시불러와야하는 생산성 측면에서 안좋을 수 있어서<br>
함수 형태로 만들어 두시면 편하게 사용할 수 있을거같아요
```python
# 예시
my_camera_pilter()
```

----------------------------------------------

## 참고 링크 및 코드 개선
