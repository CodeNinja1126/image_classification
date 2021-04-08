# image_classification

## dataset.py
- 학습 및 추론에 사용할 dataset 클래스의 구현이 정의되어 있습니다. 

### train set
-  `__getitem__` 메소드의 경우 `train.csv`의 `idx//7` 번째의 row를 읽습니다. 
 이미지는 각 row 당 7개가 존재합니다. 
 이 중 어느 이미지를 선택할 지는 `idx%7`을 통해 결정합니다.

```python3
# 이미지 이름 목록
self.img_name = ['normal', 'mask1', 'mask2', 
                'mask3', 'mask4', 'mask5', 'incorrect_mask']
# 이미지 포맷 목록
self.img_type_list = ['.png', '.jpg', '.jpeg']

# csv의 idx//7번째 경로 불러오기
img_path = os.path.join(self.data_path, 
                        self.mask_image_frame.loc[idx//7,'path'])

# 불러온 경로에서 idx%7번째 이미지 이름을 가져옴
for img_type in self.img_type_list:
    if os.path.isfile(os.path.join(img_path, self.img_name[idx%7] + img_type)):
        image = Image.open(os.path.join(img_path, self.img_name[idx%7] + img_type))
        break
```
## model.py
- 딥러닝 모델이 구현되어 있습니다.

### resModule
- 반복되는 residual connection을 구현하기 위해 resModule을 만들고 함수를 이용해 반복횟수와 채널 수를 조정할 수 있도록 하였습니다.
```python3
class resModule(nn.Module):
    def __init__(self, channel):
        super(resModule, self).__init__()
        
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(channel, channel//2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channel//2)
        self.conv2 = nn.Conv2d(channel//2, channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.bn3 = nn.BatchNorm2d(channel)
    
    def forward(self, x):
        origin = x.clone()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.bn3(x + origin)
    
        return x


def resLayer(channel, iter_num):
    resList = [resModule(channel) for _ in range(iter_num)]
    return nn.Sequential(*resList)
```

## train.py
- 모델의 학습이 구현되어 있습니다.
- argparse를 통해 코드의 수정없이 batch size나 epoch num등의 대략의 설정값을 지정할 수 있도록 하였습니다.
- 100 배치마다 평균 loss를 측정하고 랜덤값 10개를 집어넣어 대략적인 Accuracy를 볼 수 있도록 하였습니다.

## inference.py 
- eval dataset의 추론이 구현되어 있습니다.
- inference 역시 argparse를 이용해 설정값을 줘 편리성을 높였습니다.

## Notebook
- Notebook 폴더에는 위의 각 구현물들의 프로토타입이 노트북으로 구현되어 있습니다.
