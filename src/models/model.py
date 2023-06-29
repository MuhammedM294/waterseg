import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import torch.nn as nn


class UNet(nn.Module):
    
    def __init__(self, encoder_name:str, encoder_weights:str,in_channels:int = 1, classes:int = 1, activation:str = None):
        
        
        super(UNet,self).__init__()

        self.model = smp.Unet( encoder_name=encoder_name,
                               encoder_weights=encoder_weights, 
                               in_channels=in_channels,
                               classes=classes,
                               activation=activation)
    
    def forward(self, images, masks = None):


        logits = self.model(images)
        if masks != None:
            loss = DiceLoss(mode='binary')(logits, masks)+ nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss
        return logits