from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2
import os
import torch
import albumentations as A



class SegmentationDataset(Dataset[any]):
    """
    Custom dataset for segmentation tasks.

    Args:
        df (pd.DataFrame): The input DataFrame containing image and mask file paths.
        train (bool, optional): Specifies whether to use the training split of the dataset. 
            Defaults to True.
        resize (tuple, optional): The desired size for resizing images and masks. 
            Defaults to None (no resizing).
        transform (Any, optional): Any additional data transformation to be applied to the images. 
            Defaults to None.
        augment (A.Compose, optional): Augmentation pipeline for data augmentation. 
            Defaults to None.
        device (str, optional): The device to move the data to (e.g., 'cuda' or 'cpu'). 
            Defaults to 'cuda'.
        dem (bool, optional): Specifies whether to include digital elevation model (DEM) data. 
            Defaults to False.
    
    Returns:
        SegmentationDataset: A custom dataset object.

    """
    def __init__(self, df:pd.DataFrame,  train:bool =True, resize: tuple = None,
                 transform =None, augment:A.Compose = None, device:str = 'cuda' , dem:bool = False):
        
        # Initialization code goes here
        super(SegmentationDataset, self).__init__()
        self.df = df
        self.train = train
        self.resize = resize
        self.transform = transform
        self.augment = augment
        self.device = device
        self.dem = dem

        if self.train:
            #Sets the dataset to use the training images.
            self.df = df[df['split'] == 'train']
        else:
            #Sets the dataset to use the test images.
            self.df = df[df['split'] == 'test']
    
    def __len__(self):
        # Returns the length of the dataset
        return len(self.df)
    
    def __getitem__(self, idx:int):
        # Returns the data and label pair at index `idx`
        image = self.get_image(idx)
        mask = self.get_mask(idx)
        image , mask  = self.preprocess(image, mask)
        return image.to(self.device), mask.to(self.device)
    
    def get_image(self, idx:int):
        # Retrive the image at index `idx` and return it as a numpy array
        image_path = self.df.iloc[idx,1]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # in case of rgb images convert to from bgr to rgb
        if len(image.shape) ==3: image = image[:,:,::-1]
        # normalize the image
        image = image /255.0
        # resize the image
        if self.resize: 
            image =  cv2.resize(image, self.resize)
        return image
    
    def get_mask(self, idx:int):
        # Retrive the mask at index `idx` and return it as a numpy array
        mask_path = self.df.iloc[idx, 2]
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # Resize the mask
        if self.resize: 
            mask = cv2.resize(mask, self.resize)
        return mask
    
    def preprocess(self, image:np.array,    mask:np.array):

        # Preprocessing steps go here
        
        # Add channel dimension in case of grayscale images
        if len(image.shape) ==2 : image = np.expand_dims(image, axis=-1)

        # Add channel dimension to grayscale masks
        mask = np.expand_dims(mask, axis=-1)

        # Perform the augmentations pipeline
        if self.augment:
            data =self.augment()(image = image,mask = mask)
            image = data['image']
            mask = data['mask']
        
        # Read the digita; elevation model (DEM) data
        if self.dem:
            dem = cv2.imread('data/dem/dem.tif', cv2.IMREAD_UNCHANGED)
            dem = cv2.resize(dem, self.resize)
            # Normalize the DEM data
            dem = dem/dem.max()
            # Add the DEM data as a channel to the image
            image = np.dstack((image, dem))
        
        # Convert the image and mask to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)
        mask = torch.tensor(mask, dtype=torch.float32).permute(2,0,1)
       
        return image, mask


def train_augmentation():
    """
        Augmentation pipeline for training data.

        Returns:
            A.Compose: Augmentation pipeline object.
    """

    return  A.Compose([ 
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                            border_mode=cv2.BORDER_REFLECT),
        
    ])



def create_dataloader(df:pd.DataFrame, train:bool, resize: tuple = None,
                      transform:transforms.Normalize = None, augment:A.Compose = None, 
                      batch_size:int = 1, shuffle:bool = True, drop_last:bool = False , device:str = 'cuda' , dem:bool = False):
    """
    Creates a data loader for the segmentation dataset.

    Args:
        df (pd.DataFrame): The input DataFrame containing image and mask file paths.
        train (bool): Specifies whether to use the training split of the dataset.
        resize (tuple, optional): The desired size for resizing images and masks.
        transform (transforms.Normalize, optional): Additional data transformation to be applied to the images.
        augment (A.Compose, optional): Augmentation pipeline for data augmentation.
        batch_size (int, optional): The batch size. Defaults to 1.
        shuffle (bool, optional): Specifies whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): Specifies whether to drop the last incomplete batch. Defaults to False.
        device (str, optional): The device to move the data to (e.g., 'cuda' or 'cpu'). Defaults to 'cuda'.
        dem (bool, optional): Specifies whether to include digital elevation map (DEM) data. Defaults to False.
    
    Returns:
        torch.utils.data.DataLoader: A data loader object for the segmentation dataset.

    """
    dataset = SegmentationDataset(df, train,resize, transform, augment , device , dem)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle , drop_last=drop_last)

    return dataloader