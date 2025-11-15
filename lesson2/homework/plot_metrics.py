import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

def plot_metrics_from_csv(csv_path):
    """Plot training and validation metrics from CSV file"""
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get the directory name for title
    dir_name = os.path.basename(os.path.dirname(csv_path))
    
    # Create plots directory if it doesn't exist
    plots_dir = "../../plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    if 'train_loss_epoch' in df.columns:
        plt.plot(df['epoch'], df['train_loss_epoch'], label='Training Loss', marker='o')
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss - {dir_name}')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Accuracy
    plt.subplot(1, 2, 2)
    if 'train_acc_epoch' in df.columns:
        plt.plot(df['epoch'], df['train_acc_epoch'], label='Training Accuracy', marker='o')
    if 'val_acc' in df.columns:
        plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy - {dir_name}')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()

    plt.show()

def plot_all_metrics():
    """Plot metrics from all CSV files in lightning_logs"""
    
    # Find all metrics.csv files
    csv_files = glob.glob("./lightning_logs/*/metrics.csv")
    
    if not csv_files:
        print("No metrics.csv files found!")
        return
    
    print(f"Found {len(csv_files)} metrics files:")
    for i, file in enumerate(csv_files):
        print(f"{i+1}. {file}")
    
    # Plot metrics for each file
    for csv_file in csv_files:
        try:
            plot_metrics_from_csv(csv_file)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

if __name__ == "__main__":
    plot_all_metrics()