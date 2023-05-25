# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 김신성
- 리뷰어 : 심재형

PRT(PeerReviewTemplate)
----------------------------------------------

### 코드가 정상적으로 동작하고 주어진 문제를 해결했나요? (O)
전부 정상적으로 진행됩니다.<br>
```python
 if focus == 'foreground' and background_image is None:
        img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)
        img_concat = np.where(img_mask_color==255, img_orig, img_bg_blur)
    elif focus != 'foreground' and background_image is None :
        img_bg_blur = cv2.bitwise_or(img_orig_blur, img_bg_mask)
        img_concat = np.where(img_mask_color!=255, img_orig, img_bg_blur)      
```
```python
if background_image is not None:
        if backgound_blur == False:
            img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)
            img_concat = np.where(img_mask_color==255, img_orig, img_back)
        else :
            img_back = cv2.blur(img_back, (13,13))
            img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)
            img_concat = np.where(img_mask_color==255, img_orig, img_back)
```
<br>함수를 제작할 때 focus옵션 그리고 배경이미지가 있을경우 다르게 반응하는부분을 추가함으로 써 사용성이 좋은거같아요!<br>
![image](https://github.com/GrainSack/Assignments/assets/65104209/4fb56256-4647-45e3-a965-9f2599af332d)
<br>신과 짐승사진을 적절하게 사용해서 좀더 어울리는 사진이 된거같습니다
### 주석을 보고 작성자의 코드가 이해되었나요? (O)
```python
# np.where(조건, 참일때, 거짓일때)
# 세그멘테이션 마스크가 255인 부분만 원본 이미지 값을 가지고 오고 
# 아닌 영역은 블러된 이미지 값을 사용합니다.
img_concat = np.where(img_mask_color==255, img_orig, img_bg_blur)
# plt.imshow(): 저장된 데이터를 이미지의 형식으로 표시한다.
# cv2.cvtColor(입력 이미지, 색상 변환 코드): 입력 이미지의 색상 채널을 변경
# cv2.COLOR_BGR2RGB: 원본이 BGR 순서로 픽셀을 읽다보니 
# 이미지 색상 채널을 변경해야함 (BGR 형식을 RGB 형식으로 변경)
plt.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
plt.show()
```
<br>각 항목별로 주석을 자세하게 달아주셔서 아래에서 이해하는데 어려움이 없었습니다!

### 코드가 에러를 유발할 가능성이 있나요? (X)
코드 자체에는 에러를 유발할 가능성이 보이지 않습니다.
```python
if focus == 'foreground' and background_image is None:
        img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)
        img_concat = np.where(img_mask_color==255, img_orig, img_bg_blur)
    elif focus != 'foreground' and background_image is None :
        img_bg_blur = cv2.bitwise_or(img_orig_blur, img_bg_mask)
        img_concat = np.where(img_mask_color!=255, img_orig, img_bg_blur)
        
    if background_image is not None:
        if backgound_blur == False:
            img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)
            img_concat = np.where(img_mask_color==255, img_orig, img_back)
        else :
            img_back = cv2.blur(img_back, (13,13))
            img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)
            img_concat = np.where(img_mask_color==255, img_orig, img_back)
```

<br> 위에 작성한 코드처럼 함수 내에 예외처리를 깔끔하게 해놓았기 때문에 오류가 일어날 여지는 보이지 않습니다.

### 코드 작성자가 코드를 제대로 이해하고 작성했나요? (O)
![image](https://github.com/GrainSack/Assignments/assets/65104209/864e4b2a-61ad-4485-9b3e-c06be4026ed1)
<br>완벽히 이해했기 때문에 크로마키를 입히고 그 크로마키 자체도 블러처리를 하는 기능들을 추가함으로써
<br>더 나은 사용성을 제공한 부분에 대해 배우고 갑니다 ^^7
### 코드가 간결한가요? (O)
```python
a , b= Img_seg(input_image = 'musk.jpg' ,background_image='gogogo!!.jpg', focus = 'foreground',backgound_blur = True)
a , b= Img_seg(input_image = 'not_saying_his_name.jpg' ,background_image='musk.jpg', focus = 'foreground',backgound_blur = False)
a , b= Img_seg(background_image = 'back.jpg', backgound_blur = True)
a , b= Img_seg(focus = 'background')
```
자주 사용하는 코드의 재사용을 하기위해 함수를 제작하고 여러가지 옵션을 추가해서 훨씬 가독성 있습니다!
<br> 멋진 코드 배우고갑니다 ꉂꉂ(ᵔᗜᵔ*)

----------------------------------------------

## 참고 링크 및 코드 개선
