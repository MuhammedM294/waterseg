import matplotlib.pyplot as plt

def visualize_over_epochs(train_metrics, valid_metrics ,
                           metric_name , figsize=(10, 7), 
                             ):
    
    """
    Visualize the specified metric over epochs for both train and validation sets.

    Args:
        train_metrics (Metrics): The Metrics object containing the metrics computed for the train set.
        valid_metrics (Metrics): The Metrics object containing the metrics computed for the validation set.
        metric_name (str): The name of the metric to visualize.
        figsize (tuple): The size of the figure (width, height). Default is (10, 7).

    """
    plt.figure(figsize=figsize)
    plt.plot(train_metrics.get_epochs_metrics()[metric_name], color='blue', label=f'Train {metric_name.title()}')
    plt.plot(valid_metrics.get_epochs_metrics()[metric_name], color='red', label=f'Valid {metric_name.title()}')
    plt.xlabel('Epochs')
    plt.ylabel(f"{metric_name.title()}")
    plt.title(f'{metric_name.title()} over epochs')
    plt.legend()
    plt.show()
