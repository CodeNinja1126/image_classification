# image_classification

## dataset.py
- 학습 및 추론에 사용할 dataset 클래스의 구현이 정의되어 있습니다. 

### train set
-  `__getitem__` 메소드의 경우 `train.csv`의 `idx//7` 번째의 row를 읽습니다. 
 이미지는 각 row 당 7개가 존재합니다. 
 이 중 어느 이미지를 선택할 지는 `idx%7`을 통해 결정합니다.

