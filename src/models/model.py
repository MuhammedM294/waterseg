import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import torch.nn as nn


class UNet(nn.Module):
    """
    UNet model for image segmentation.

    Args:
        encoder_name (str): The name of the encoder backbone architecture.
        encoder_weights (str): Path to the pretrained encoder weights file or 'imagenet' for ImageNet pretraining.
        in_channels (int, optional): Number of input channels. Defaults to 1.
        classes (int, optional): Number of classes to predict. Defaults to 1.
        activation (str, optional): Activation function to use. Defaults to None.

    Attributes:
        model: The UNet model from the segmentation_models.pytorch library.

    """
    
    def __init__(self, encoder_name:str, encoder_weights:str,in_channels:int = 1, classes:int = 1, activation:str = None):
        
        
        super(UNet,self).__init__()

        self.model = smp.Unet( encoder_name=encoder_name,
                               encoder_weights=encoder_weights, 
                               in_channels=in_channels,
                               classes=classes,
                               activation=activation)
    
    def forward(self, images, masks = None):
        """
        Forward pass of the UNet model.

        Args:
            images: The input images.
            masks: The ground truth masks for training.

        Returns:
            logits: The predicted logits from the UNet model.
            loss: The calculated loss (if ground truth masks are provided).

        """

        logits = self.model(images)
        if masks != None:
            loss = DiceLoss(mode='binary')(logits, masks)+ nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss
        return logits