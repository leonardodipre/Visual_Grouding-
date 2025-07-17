import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_metrics(csv_file):
    df = pd.read_csv(csv_file)
    epochs = df['epoch']

   
    min_idx = df['train_loss'].idxmin()
    min_value = df.loc[min_idx, 'train_loss']
    min_epoch = df.loc[min_idx, 'epoch']

    print(f"train_loss minimo: {min_value:.4f} all'epoch {min_epoch}")

    max_val_mean_iou = df['val_mean_iou'].idxmax()
    max_value_iou = df.loc[max_val_mean_iou, 'val_mean_iou']
    min_epoch_iou = df.loc[max_val_mean_iou, 'epoch']

    print(f"val_mean_iou max: {max_value_iou:.4f} all'epoch {min_epoch_iou}")

    max_val_mean_acc = df['val_accuracy'].idxmax()
    max_value_acc = df.loc[max_val_mean_acc, 'val_accuracy']
    min_epoch_acc = df.loc[max_val_mean_acc, 'epoch']

    print(f"val_accuracy max: {max_value_acc:.4f} all'epoch {min_epoch_acc}")



    
    # Plot generale delle loss principali
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df['train_loss'], label='Train Loss')
    plt.plot(epochs, df['bbox_loss'], label='BBox Loss')
    plt.title("Train & BBox Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Metriche di performance
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df['val_mean_iou'], label='Val Mean IoU')
    plt.plot(epochs, df['val_accuracy'], label='Val Accuracy')
    plt.title("Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Altre loss: RAC, MRC, ATT
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df['rac_loss'], label='RAC Loss')
    plt.plot(epochs, df['mrc_loss'], label='MRC Loss')
    plt.plot(epochs, df['att_reg_loss'], label='Attention Reg Loss')
    plt.title("Regularization Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Weighted loss components
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df['w_adw'], label='w_adw')
    plt.plot(epochs, df['w_odw'], label='w_odw')
    plt.title("Weighted Loss Factors")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot metrics from a training log CSV.')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file')
    args = parser.parse_args()
    plot_metrics(args.csv_file)
