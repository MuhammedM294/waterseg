import torch
import numpy as np
from collections import namedtuple

class Metrics(object):
    
   def __init__(self ,output = None, target = None):
        self.smooth = 1e-6
        self.output = output
        self.target = target 
        self.CM = None
        self.batchs_accurcy, self.epochs_accurcy = [] , []
        self.batchs_precision , self.epochs_precision = [] , []
        self.batchs_recall , self.epochs_recall= [] , []
        self.batchs_f1_score , self.epochs_f1_score = [] , []
        self.batchs_specificity, self.epochs_specificity= [] , []
        self.batchs_iou , self.epochs_iou = [] , []
       

       

   def update(self, output, target):
        self.CM = self.confusion_matric(output, target)
        self.save_batch(self.CM)

   def get_batchs_values(self):
        return {"accuracy": np.mean(self.batchs_accurcy), \
                "precision": np.mean(self.batchs_precision), \
                "recall": np.mean(self.batchs_recall), \
                "f1_score": np.mean(self.batchs_f1_score), \
                "specificity": np.mean(self.batchs_specificity), \
                "iou": np.mean(self.batchs_iou)}
   def get_epochs_values(self):
        return {"accuracy": self.epochs_accurcy, \
                "precision": self.epochs_precision, \
                "recall": self.epochs_recall, \
                "f1_score": self.epochs_f1_score, \
                "specificity": self.epochs_specificity, \
                "iou": self.epochs_iou}

   def confusion_matric(self, output, target):
        
        assert output.shape == target.shape, \
        "output and target must have the same shape"
        assert output.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'
        
        TP = torch.sum((output == 1) & (target == 1)).item()
        TN = torch.sum((output == 0) & (target == 0)).item()
        FP = torch.sum((output == 1) & (target == 0)).item()
        FN = torch.sum((output == 0) & (target == 1)).item()
        accuracy = (TP + TN + self.smooth) / (TP + TN + FP + FN + self.smooth)
        precision = (TP + self.smooth) / (TP + FP + self.smooth)
        recall = (TP + self.smooth) / (TP + FN + self.smooth)
        f1_score = 2 * (precision * recall) / (precision + recall + self.smooth)
        specificity = (TN + self.smooth) / (TN + FP + self.smooth)
        iou = (TP + self.smooth) / (TP + FP + FN + self.smooth)
        self.CM = namedtuple("CM", ["TP", "TN", "FP", "FN" ,\
                                        "accuracy", "precision", "recall", "f1_score", \
                                        "specificity", "iou"])
        CM = self.CM(TP, TN, FP, FN , accuracy, precision, recall, f1_score, specificity, iou)

        return CM 
   
  
   def save_batch(self, CM):
        self.batchs_accurcy.append(CM.accuracy)
        self.batchs_precision.append(CM.precision)
        self.batchs_recall.append(CM.recall)
        self.batchs_f1_score.append(CM.f1_score)
        self.batchs_specificity.append(CM.specificity)
        self.batchs_iou.append(CM.iou)
    
   def save_epoch(self):
        self.epochs_accurcy.append(np.mean(self.batchs_accurcy))
        self.epochs_precision.append(np.mean(self.batchs_precision))
        self.epochs_recall.append(np.mean(self.batchs_recall))
        self.epochs_f1_score.append(np.mean(self.batchs_f1_score))
        self.epochs_specificity.append(np.mean(self.batchs_specificity))
        self.epochs_iou.append(np.mean(self.batchs_iou))
        print("Epoch Metrics Saved!")


       
  
       
  
       
