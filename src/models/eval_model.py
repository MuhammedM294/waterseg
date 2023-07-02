import torch
from tqdm import tqdm
from src.models.metrics import Metrics

def eval_fn(valid_dataloader, model , metrics):
    """
    Performs evaluation on the validation dataset using the provided model and metrics.

    Args:
        valid_dataloader (torch.utils.data.DataLoader): The data loader for the validation dataset.
        model: The model used for evaluation.
        metrics: The metrics object used for tracking evaluation metrics.

    Returns:
        dict: A dictionary containing the metrics computed for each batch.

    """
    print("Validating ...")
    model.eval()
    for image, mask in tqdm(valid_dataloader):
        with torch.no_grad():
            logits, loss = model(image, mask)
            pred_mask = (torch.sigmoid(logits) > 0.5)*1
            metrics.update( pred_mask, mask , loss.item() )
            
    
    metrics_batches_values = metrics.get_batches_metrics()
    print("Validation Done!")
    metrics.save_epoch()

    return metrics_batches_values 