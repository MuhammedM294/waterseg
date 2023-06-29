import matplotlib.pyplot as plt

def visualize_over_epochs(train_metrics, valid_metrics ,
                           metric_name , figsize=(10, 7), 
                             ):
  
    plt.figure(figsize=figsize)
    plt.plot(train_metrics.get_epochs_values()[metric_name], color='blue', label=f'Train {metric_name.title()}')
    plt.plot(valid_metrics.get_epochs_values()[metric_name], color='red', label=f'Valid {metric_name.title()}')
    plt.xlabel('Epochs')
    plt.ylabel(f"{metric_name.title()}")
    plt.title(f'{metric_name.title()} over epochs')
    plt.legend()
    plt.show()
