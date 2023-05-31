# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 김신성
- 리뷰어 : 심재형

PRT(PeerReviewTemplate)
----------------------------------------------

### 텐서플로우와 다른 프레임워크 사용(파이토치)
```python
import torch
from transformers import GPT2LMHeadModel
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import re
#from pytorch_lightning.core.lightning import LightningModule


import urllib.request
from transformers import PreTrainedTokenizerFast
```
제가 파이토치에 대해선 잘 모르지만 한번 열심히 리뷰해보겠습니다!<br>

```python
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
```
pretrained된 kogpt2모델을 사용하신 부분도 훨씬 더 나은 결과를 도출할수 있을거 같습니다!<br>

```python
# 챗봇 데이터를 처리하는 클래스를 만든다.
class ChatbotDataset(Dataset):
    def __init__(self, chats, max_len=40):  # 데이터셋의 전처리를 해주는 부분
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.mask = MASK
        self.tokenizer = koGPT2_TOKENIZER

    def __len__(self):  # chatbotdata 의 길이를 리턴한다.
        return len(self._data)

    def __getitem__(self, idx):  # 로드한 챗봇 데이터를 차례차례 DataLoader로 넘겨주는 메서드
        turn = self._data.iloc[idx]
        q = turn["Q"]  # 질문을 가져온다.
        q = re.sub(r"([?.!,])", r" ", q)  # 구둣점들을 제거한다.

        a = turn["A"]  # 답변을 가져온다.
        a = re.sub(r"([?.!,])", r" ", a)  # 구둣점들을 제거한다.

        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)

        #질문의 길이가 최대길이보다 크면
        if q_len > self.max_len:
            a_len = self.max_len - q_len        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:       #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_toked = q_toked[-(int(self.max_len / 2)) :]   #질문길이를 최대길이의 반으로 
                q_len = len(q_toked)
                a_len = self.max_len - q_len              #답변의 길이를 최대길이 - 질문길이
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        #질문의 길이 + 답변의 길이가 최대길이보다 크면
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:       #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_toked = q_toked[-(int(self.max_len / 2)) :]   #질문길이를 최대길이의 반으로 
                q_len = len(q_toked)
                a_len = self.max_len - q_len              #답변의 길이를 최대길이 - 질문길이
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        # 답변 labels = [mask, mask, ...., mask, ..., ,..답변.. , ....]
        labels = [self.mask,] * q_len + a_toked[1:]

        # mask = 질문길이 0 + 답변길이 1 + 나머지 0
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        # 답변 labels을 index 로 만든다.
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        # 최대길이만큼 PADDING
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        # 질문 + 답변을 index 로 만든다.    
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        # 최대길이만큼 PADDING
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        #질문+답변, 마스크, 답변
        return (token_ids, np.array(mask), labels_ids)
```
수많은 데이터 전처리 과정을 클래스로 처리해줌으로 써 훨씬더 간결하게 표현할 수 있고 좋은것 같습니다,,!<br>
주석 자체에서 계산을 진행함으로써 눈으로 따라가면 직관적인 이해가 가능해서 처음보는데도 이해가 잘 됩니다.<br>

```python
train_set = ChatbotDataset(Chatbot_Data, max_len=40)
```
클래스 처리를통해 복잡한 전처리 과정을 간편하게 진행했습니다!<br>
여기서 **max_len=40**의 값을 찾는 과정이 있으면 조금 더 최적의 값을 찾을 수 있을거같아요!<br>

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("start")
for epoch in range(epochs):
    for batch_idx, samples in enumerate(train_dataloader):
        optimizer.zero_grad()
        token_ids, mask, label = samples

        # Move tensors to the same device as the model
        token_ids = token_ids.to(device)
        mask = mask.to(device)
        label = label.to(device)

        out = model(token_ids)
        out = out.logits

        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))
        loss = criterion(mask_out.transpose(2, 1), label)

        avg_loss = loss.sum() / mask.sum()
        avg_loss.backward()
        optimizer.step()

print("end")
```
GPU를 사용하는 과정에서 시작과 끝을 알려주는 코멘트를 넣는 방법이 확실히 진행되고 있는 사안에대해서 확인할 수 있어서 좋은거같습니다~!
<br>

```python
with torch.no_grad():
    while True:
        q = input("user > ").strip()
        if q == "quit":
            break
        a = ""
        while True:
            input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + A_TKN + a)).unsqueeze(dim=0)

            # Move the input tensor to the same device as the model
            input_ids = input_ids.to(device)

            pred = model(input_ids)
            pred = pred.logits

            # Move the tensor from CUDA device to CPU and convert to NumPy array
            gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]

            if gen == EOS:
                break
            a += gen.replace("▁", " ")
        print("Chatbot > {}".format(a.strip()))
```
일반적으로 질문 - 답변 딱 한번만 진행하는 함수형태가아닌 계속 대화를 진행하는 과정이 정말 GPT형태처럼 멋있습니다..!!
<br> 위 방법도 엄청 좋지만 statefull하게 하기위해 stateless처럼 전 대화를 더하는 방식도 괜찮은거같아요!<br>
![image](https://github.com/GrainSack/Assignments/assets/65104209/e0391740-832d-4ad3-af5f-b98bd39fdc8b)


![image](https://github.com/GrainSack/Assignments/assets/65104209/64207244-a3c5-4065-a3b6-352afafcf736)
<br> 상남자의 대화법까지 감명받았습니다

### 코드가 정상적으로 동작하고 주어진 문제를 해결했나요? (O)
### 주석을 보고 작성자의 코드가 이해되었나요? (O)
### 코드가 에러를 유발할 가능성이 있나요? (X)
### 코드 작성자가 코드를 제대로 이해하고 작성했나요? (O)
### 코드가 간결한가요? (O)

----------------------------------------------

## 참고 링크 및 코드 개선
- https://www.slideshare.net/xguru/ss-16106464
