import torch
import numpy as np
from collections import namedtuple

class Metrics(object):
   """
    Class for tracking and computing evaluation metrics during model training.

    Args:
        output: The predicted output from the model.
        target: The target ground truth values.

    Attributes:
        smooth (float): A small value added for numerical stability.
        output: The predicted output from the model.
        target: The target ground truth values.
        CM: A named tuple representing the confusion matrix and evaluation metrics.
        batches_loss (list): List to store the losses computed for each batch.
        epochs_loss (list): List to store the average losses computed for each epoch.
        batches_accuracy (list): List to store the accuracies computed for each batch.
        epochs_accuracy (list): List to store the average accuracies computed for each epoch.
        batches_precision (list): List to store the precisions computed for each batch.
        epochs_precision (list): List to store the average precisions computed for each epoch.
        batches_recall (list): List to store the recalls computed for each batch.
        epochs_recall (list): List to store the average recalls computed for each epoch.
        batches_f1_score (list): List to store the F1 scores computed for each batch.
        epochs_f1_score (list): List to store the average F1 scores computed for each epoch.
        batches_specificity (list): List to store the specificities computed for each batch.
        epochs_specificity (list): List to store the average specificities computed for each epoch.
        batches_iou (list): List to store the intersection over union (IoU) values computed for each batch.
        epochs_iou (list): List to store the average IoU values computed for each epoch.

    """
    
   def __init__(self ,output = None, target = None):
        self.smooth = 1e-6
        self.output = output
        self.target = target 
        self.CM = None
        self.batches_loss , self.epochs_loss = [] , []
        self.batches_accurcy, self.epochs_accurcy = [] , []
        self.batches_precision , self.epochs_precision = [] , []
        self.batches_recall , self.epochs_recall= [] , []
        self.batches_f1_score , self.epochs_f1_score = [] , []
        self.batches_specificity, self.epochs_specificity= [] , []
        self.batches_iou , self.epochs_iou = [] , []
           

   def update(self, output, target, loss):
        """
        Update the metrics using the predicted output, target, and loss.

        Args:
            output: The predicted output from the model.
            target: The target ground truth values.
            loss: The loss value for the current batch.

        """

        self.CM = self.confusion_matric(output, target)
        self.save_batch(self.CM, loss)

   

   def confusion_matric(self, output, target):
        """
        Compute the confusion matrix and evaluation metrics.

        Args:
            output: The predicted output from the model.
            target: The target ground truth values.

        Returns:
            CM: A named tuple representing the confusion matrix and evaluation metrics.

        """
        
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
   
  
   def save_batch(self, CM , loss ):
        """
        Save the metrics computed for the current batch.

        Args:
            CM: A named tuple representing the confusion matrix and evaluation metrics.
            loss: The loss value for the current batch.

        """
        self.batches_loss.append(loss)
        self.batches_accurcy.append(CM.accuracy)
        self.batches_precision.append(CM.precision)
        self.batches_recall.append(CM.recall)
        self.batches_f1_score.append(CM.f1_score)
        self.batches_specificity.append(CM.specificity)
        self.batches_iou.append(CM.iou)
    
   def save_epoch(self):
        """
        Save the average metrics computed for the current epoch.

        """
        self.epochs_loss.append(np.mean(self.batches_loss))
        self.epochs_accurcy.append(np.mean(self.batches_accurcy))
        self.epochs_precision.append(np.mean(self.batches_precision))
        self.epochs_recall.append(np.mean(self.batches_recall))
        self.epochs_f1_score.append(np.mean(self.batches_f1_score))
        self.epochs_specificity.append(np.mean(self.batches_specificity))
        self.epochs_iou.append(np.mean(self.batches_iou))

        print("Epoch Metrics Saved!")
        self.reset()
   
   def reset(self):
        """
        Reset the metric lists.

        """
        self.batches_loss  = []
        self.batches_accurcy = []
        self.batches_precision = []
        self.batches_recall = []
        self.batches_f1_score = []
        self.batches_specificity = []
        self.batches_iou = []
     
        print("Metrics Reseted!")

   def get_batches_metrics(self):
        """
        Get the average metrics computed for each batch.

        Returns:
            metrics_batches_values: A dictionary containing the average metrics computed for each batch.

        """
        return {"loss": np.mean(self.batches_loss), \
                "accuracy": np.mean(self.batches_accurcy), \
                "precision": np.mean(self.batches_precision), \
                "recall": np.mean(self.batches_recall), \
                "f1_score": np.mean(self.batches_f1_score), \
                "specificity": np.mean(self.batches_specificity), \
                "iou": np.mean(self.batches_iou)}
   

   def get_epochs_metrics(self):
        """
        Get the average metrics computed for each epoch.

        Returns:
            metrics_epochs_values: A dictionary containing the average metrics computed for each epoch.

        """
        return {"loss": self.epochs_loss, \
                "accuracy": self.epochs_accurcy, \
                "precision": self.epochs_precision, \
                "recall": self.epochs_recall, \
                "f1_score": self.epochs_f1_score, \
                "specificity": self.epochs_specificity, \
                "iou": self.epochs_iou}


       
  
       
  
       
