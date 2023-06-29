import torch.optim as optim
from tqdm import tqdm
from src.models.metrics import Metrics
import torch


def train_fn(train_dataloader, model, optimizer , metrics):
   
    model.train()
    train_loss = 0
    for image, mask in tqdm(train_dataloader):
        optimizer.zero_grad()
        logits, loss = model(image, mask)
        pred_mask = (torch.sigmoid(logits) > 0.5)*1
        metrics.update(pred_mask, mask)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    epoch_metrics = metrics.get_batchs_values()
    metrics.save_epoch()
    return train_loss / len(train_dataloader) , epoch_metrics