import torch
from tqdm import tqdm
from src.models.metrics import Metrics

def eval_fn(valid_dataloader, model , metrics):

    model.eval()
    valid_loss = 0
    valid_dataloader_len = len(valid_dataloader)
    for image, mask in tqdm(valid_dataloader):
        with torch.no_grad():
            logits, loss = model(image, mask)
            valid_loss += loss.item()
            pred_mask = (torch.sigmoid(logits) > 0.5)*1
            metrics.update(pred_mask, mask)
    epoch_metrics = metrics.get_batchs_values()
    metrics.save_epoch()

    return valid_loss / valid_dataloader_len , epoch_metrics