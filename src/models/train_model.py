import torch.optim as optim
from tqdm import tqdm
import torch
from src.models.metrics import Metrics


def train_fn(train_dataloader, model, optimizer , metrics):
    """
    Performs training on the training dataset using the provided model, optimizer, and metrics.

    Args:
        train_dataloader (torch.utils.data.DataLoader): The data loader for the training dataset.
        model: The model used for training.
        optimizer: The optimizer used for updating the model parameters.
        metrics: The metrics object used for tracking training metrics.

    Returns:
        dict: A dictionary containing the metrics computed for each batch.

    """

    print("Training ...")
    model.train()
    for image, mask in tqdm(train_dataloader):
        optimizer.zero_grad()
        logits, loss = model(image, mask)
        pred_mask = (torch.sigmoid(logits) > 0.5)*1
        loss.backward()
        optimizer.step()
        metrics.update( pred_mask, mask, loss.item())
        
    
    metrics_batches_values = metrics.get_batches_metrics()
    print("Training Done!")
    metrics.save_epoch()
   
    return  metrics_batches_values