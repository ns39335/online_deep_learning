"""
Training script for Part 2: Road Detection (Segmentation + Depth)

Implements the training code from scratch including:
* Creating an optimizer
* Creating a model, loss, metrics (task dependent)
* Loading the data (task dependent)
* Running the optimizer for several epochs
* Logging + saving your model (use the provided `save_model`)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from homework.models import Detector, save_model
from homework.metrics import DetectionMetric
from homework.datasets.road_dataset import load_data


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    seg_weight: float = 1.0,
    depth_weight: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Training loop for the detector

    Args:
        model: The detector model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for the optimizer
        seg_weight: Weight for segmentation loss
        depth_weight: Weight for depth loss
        device: Device to train on (cuda or cpu)
    """
    model = model.to(device)
    
    # Create optimizer (Adam with weight decay for regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Create loss functions
    # For segmentation: CrossEntropyLoss (handles class imbalance)
    seg_criterion = nn.CrossEntropyLoss()
    
    # For depth: L1Loss (Mean Absolute Error)
    depth_criterion = nn.L1Loss()
    
    # Metrics
    train_metric = DetectionMetric(num_classes=3)
    val_metric = DetectionMetric(num_classes=3)
    
    best_val_iou = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_metric.reset()
        total_train_loss = 0.0
        train_seg_loss_sum = 0.0
        train_depth_loss_sum = 0.0
        num_batches = 0
        
        for batch in train_loader:
            images = batch["image"].to(device)
            track_labels = batch["track"].to(device).long()
            depth_labels = batch["depth"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, depth_pred = model(images)
            
            # Compute losses
            seg_loss = seg_criterion(logits, track_labels)
            depth_loss = depth_criterion(depth_pred, depth_labels)
            
            # Combine losses with weights
            total_loss = seg_weight * seg_loss + depth_weight * depth_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            total_train_loss += total_loss.item()
            train_seg_loss_sum += seg_loss.item()
            train_depth_loss_sum += depth_loss.item()
            num_batches += 1
            
            # Update metrics
            preds, depth = model.predict(images)
            train_metric.add(preds, track_labels, depth, depth_labels)
        
        # Validation phase
        model.eval()
        val_metric.reset()
        total_val_loss = 0.0
        val_seg_loss_sum = 0.0
        val_depth_loss_sum = 0.0
        val_batches = 0
        
        with torch.inference_mode():
            for batch in val_loader:
                images = batch["image"].to(device)
                track_labels = batch["track"].to(device).long()
                depth_labels = batch["depth"].to(device)
                
                logits, depth_pred = model(images)
                
                # Compute losses
                seg_loss = seg_criterion(logits, track_labels)
                depth_loss = depth_criterion(depth_pred, depth_labels)
                total_loss = seg_weight * seg_loss + depth_weight * depth_loss
                
                total_val_loss += total_loss.item()
                val_seg_loss_sum += seg_loss.item()
                val_depth_loss_sum += depth_loss.item()
                val_batches += 1
                
                preds, depth = model.predict(images)
                val_metric.add(preds, track_labels, depth, depth_labels)
        
        # Compute metrics
        train_results = train_metric.compute()
        val_results = val_metric.compute()
        
        avg_train_loss = total_train_loss / num_batches
        avg_val_loss = total_val_loss / val_batches
        
        # Update learning rate based on validation IoU
        scheduler.step(val_results['iou'])
        
        # Log results
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train - Loss: {avg_train_loss:.4f} "
              f"(Seg: {train_seg_loss_sum/num_batches:.4f}, "
              f"Depth: {train_depth_loss_sum/num_batches:.4f})")
        print(f"          IoU: {train_results['iou']:.4f}, "
              f"Depth Error: {train_results['abs_depth_error']:.4f}, "
              f"TP Depth Error: {train_results['tp_depth_error']:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f} "
              f"(Seg: {val_seg_loss_sum/val_batches:.4f}, "
              f"Depth: {val_depth_loss_sum/val_batches:.4f})")
        print(f"          IoU: {val_results['iou']:.4f}, "
              f"Depth Error: {val_results['abs_depth_error']:.4f}, "
              f"TP Depth Error: {val_results['tp_depth_error']:.4f}")
        
        # Save best model based on IoU
        if val_results['iou'] > best_val_iou:
            best_val_iou = val_results['iou']
            save_path = save_model(model)
            print(f"  ✓ New best model saved! IoU: {best_val_iou:.4f}")
    
    print(f"\nTraining completed! Best validation IoU: {best_val_iou:.4f}")


def main():
    """
    Main function to set up data and start training
    """
    # Set your hyperparameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-3
    seg_weight = 1.0
    depth_weight = 1.0
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load training data
    print("\nLoading training data...")
    train_data = load_data(
        dataset_path="drive_data/train",
        transform_pipeline="default",
        return_dataloader=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    
    # Load validation data (no augmentation!)
    print("Loading validation data...")
    val_data = load_data(
        dataset_path="drive_data/val",
        transform_pipeline="default",
        return_dataloader=True,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    
    # Create model
    print("\nCreating model...")
    model = Detector(in_channels=3, num_classes=3)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Start training
    print("\nStarting training...")
    print("=" * 80)
    train(model, train_data, val_data, num_epochs, learning_rate, 
          seg_weight, depth_weight, device)
    print("=" * 80)


if __name__ == "__main__":
    main()
