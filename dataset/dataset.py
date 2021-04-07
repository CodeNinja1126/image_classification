import os
import tqdm
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 데이터셋 위치 및 파일 명
img_name = ['normal', 'mask1', 'mask2', 
            'mask3', 'mask4', 'mask5', 'incorrect_mask']
img_type_list = ['.png', '.jpg', '.jpeg']

csv_path = '/opt/ml/input/data/train/train.csv'
data_path = '/opt/ml/input/data/train/images'

test_csv_path = '/opt/ml/input/data/eval/info.csv'
test_data_path = '/opt/ml/input/data/eval/images'

# 데이터 전처리
data_transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

class MaskImageDataset(Dataset):
    
    def __init__(self, csv_file, data_path, transform=None):
        """
        Args:
            csv_file (string): csv_file 경로
            data_path (string): data_path 경로
            transform (string): 샘플에 적용될 transform(전처리)
        """
        self.mask_image_frame = pd.read_csv(csv_file)
        self.data_path = data_path
        self.transform = transform
    
    
    def __len__(self):
        return len(self.mask_image_frame) * 7
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = os.path.join(self.data_path, 
                                self.mask_image_frame.loc[idx//7,'path'])
        
        for img_type in img_type_list:
            if os.path.isfile(os.path.join(img_path, img_name[idx%7] + img_type)):
                image = Image.open(os.path.join(img_path, img_name[idx%7] + img_type))
                break
        
        if self.transform:
            image = self.transform(image)

        img_class = self.mask_image_frame.loc[idx//7,'age'] // 30

        if self.mask_image_frame.loc[idx//7,'gender'] == 'female':
            img_class += 3
        
        if idx%7 == 0:
            img_class += 12
        elif idx%7 == 6:
            img_class += 6
        
        label = torch.zeros((1, 18), dtype = torch.float)
        label[0, img_class] += 1
            
        return image, label

class ValidationSet(Dataset):
    
    def __init__(self, csv_file, data_path, transform=None):
        self.mask_image_frame = pd.read_csv(csv_file)
        self.data_path = data_path
        self.transform = transform


    def __len__(self):
        return len(self.mask_image_frame)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = os.path.join(self.data_path, 
                               self.mask_image_frame.loc[idx, 'ImageID'])
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, idx