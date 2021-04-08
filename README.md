# image_classification

## dataset.py
- 학습 및 추론에 사용할 dataset 클래스의 구현이 정의되어 있습니다. 

### train set
-  `__getitem__` 메소드의 경우 `train.csv`의 `idx//7` 번째의 row를 읽습니다. 
 이미지는 각 row 당 7개가 존재합니다. 
 이 중 어느 이미지를 선택할 지는 `idx%7`을 통해 결정합니다.

```python3
self.img_name = ['normal', 'mask1', 'mask2', 
                'mask3', 'mask4', 'mask5', 'incorrect_mask']
self.img_type_list = ['.png', '.jpg', '.jpeg']

img_path = os.path.join(self.data_path, 
                        self.mask_image_frame.loc[idx//7,'path'])
        
for img_type in self.img_type_list:
    if os.path.isfile(os.path.join(img_path, self.img_name[idx%7] + img_type)):
        image = Image.open(os.path.join(img_path, self.img_name[idx%7] + img_type))
        break
```
## model.py
- 딥러닝 모델이 구현되어 있습니다.

### resModule
