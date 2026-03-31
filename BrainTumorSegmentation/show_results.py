import json
import math
import matplotlib.pyplot as plt
from helpers import show_prediction, show_prediction_diff
from brain_dataset import BrainDataSet
import random
from UNet import UNet

def main():
    with open("history.json") as f:
        history = json.load(f)
    
    metrics = ['acc', 'auc_pr', 'auc_roc', 'iou', 'loss', 'precision', 'recall']
    
    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 4 * nrows),
        sharex=True,
        constrained_layout=True
    )
    
    axs = axs.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axs[i]
        
        ax.plot(history[metric], label='train', linewidth=2)
        ax.plot(history[f'val_{metric}'], label='val', linewidth=2)
        
        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    for j in range(len(metrics), len(axs)):
        axs[j].axis("off")
    
    plt.savefig("graphics/results.png")
    plt.show()
    
    ds = BrainDataSet()
    val_ds = ds.val_ds # this is okay, since the seed is set in the dataset, otherwise one has to be careful to avoid data leakage
    
    model = UNet().model
    model.load_weights("brain.weights.h5")
    
    num_text_ims = 4

    batch_numbers = random.sample(range(len(val_ds)), num_text_ims)
    idx_numbers = random.sample(range(ds.batch_size), num_text_ims)
    
    for idx in range(len(idx_numbers)):
      show_prediction(val_ds, model, batch_numbers[idx], idx_numbers[idx])
      show_prediction_diff(val_ds, model, batch_numbers[idx], idx_numbers[idx])

if __name__ == '__main__':
    main()