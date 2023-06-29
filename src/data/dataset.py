from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2
import os
import torch
import albumentations as A



class SegmentationDataset(Dataset[any]):
    def __init__(self, df:pd.DataFrame, train:bool, resize: tuple = None,transform =None, augment:A.Compose = None, device:str = 'cuda'):
        self.df = df
        self.train = train
        self.resize = resize
        self.transform = transform
        self.augment = augment
        self.device = device

        if self.train:
            self.df = df[df['split'] == 'train']
        else:
            self.df = df[df['split'] == 'test']
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx:int):
        image = self.get_image(idx)
        mask = self.get_mask(idx)
        image , mask  = self.preprocess(image, mask)
        return image.to(self.device), mask.to(self.device)
    
    def get_image(self, idx:int):
        image_path = self.df.iloc[idx, 3]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = image /255.0
        if self.resize: 
            image =  cv2.resize(image, self.resize)
        return image
    
    def get_mask(self, idx:int):
        mask_path = self.df.iloc[idx, 4]
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if self.resize: 
            mask = cv2.resize(mask, self.resize)
        return mask
    
    def preprocess(self, image:np.array,    mask:np.array):

        image = np.expand_dims(image, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        if self.augment:
            data =self.augment()(image = image,mask = mask)
            image = data['image']
            mask = data['mask']
        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)
        mask = torch.tensor(mask, dtype=torch.float32).permute(2,0,1)
       
        return image, mask


def train_augmentation():

    return  A.Compose([ 
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                            border_mode=cv2.BORDER_REFLECT),
        
    ])



def create_dataloader(df:pd.DataFrame, train:bool, resize: tuple = None,
                      transform:transforms.Normalize = None, augment:A.Compose = None, 
                      batch_size:int = 1, shuffle:bool = True, drop_last:bool = False , device:str = 'cuda'):
  
    dataset = SegmentationDataset(df, train,resize, transform, augment , device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle , drop_last=drop_last)

    return dataloader