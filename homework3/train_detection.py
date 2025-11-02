"""
Training script for the Detector model
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from homework import models
from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    log_dir: Path = None,
    seg_weight: float = 1.0,
    depth_weight: float = 1.0,
):
    """
    Train the detection model

    Args:
        model: the Detector model
        train_loader: training data loader
        val_loader: validation data loader
        num_epochs: number of training epochs
        learning_rate: learning rate for optimizer
        device: device to train on
        log_dir: directory for tensorboard logs
        seg_weight: weight for segmentation loss
        depth_weight: weight for depth loss
    """
    model = model.to(device)
    model.train()

    # Loss functions
    seg_criterion = nn.CrossEntropyLoss()
    depth_criterion = nn.L1Loss()  # Mean Absolute Error
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Tensorboard writer
    if log_dir is not None:
        writer = SummaryWriter(log_dir)
    
    global_step = 0
    best_iou = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_metric = DetectionMetric(num_classes=3)
        train_seg_loss = 0.0
        train_depth_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            track_labels = batch["track"].to(device)
            depth_labels = batch["depth"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            seg_logits, depth_pred = model(images)
            
            # Compute losses
            seg_loss = seg_criterion(seg_logits, track_labels)
            depth_loss = depth_criterion(depth_pred, depth_labels)
            total_loss = seg_weight * seg_loss + depth_weight * depth_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Track metrics
            train_seg_loss += seg_loss.item()
            train_depth_loss += depth_loss.item()
            
            with torch.no_grad():
                seg_preds = seg_logits.argmax(dim=1)
                train_metric.add(seg_preds, track_labels, depth_pred, depth_labels)
            
            if log_dir is not None and batch_idx % 10 == 0:
                writer.add_scalar("train/seg_loss", seg_loss.item(), global_step)
                writer.add_scalar("train/depth_loss", depth_loss.item(), global_step)
                writer.add_scalar("train/total_loss", total_loss.item(), global_step)
            
            global_step += 1
        
        # Compute training metrics
        train_metrics = train_metric.compute()
        avg_train_seg_loss = train_seg_loss / len(train_loader)
        avg_train_depth_loss = train_depth_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_metric = DetectionMetric(num_classes=3)
        val_seg_loss = 0.0
        val_depth_loss = 0.0
        
        with torch.inference_mode():
            for batch in val_loader:
                images = batch["image"].to(device)
                track_labels = batch["track"].to(device)
                depth_labels = batch["depth"].to(device)
                
                seg_logits, depth_pred = model(images)
                
                seg_loss = seg_criterion(seg_logits, track_labels)
                depth_loss = depth_criterion(depth_pred, depth_labels)
                
                val_seg_loss += seg_loss.item()
                val_depth_loss += depth_loss.item()
                
                seg_preds = seg_logits.argmax(dim=1)
                val_metric.add(seg_preds, track_labels, depth_pred, depth_labels)
        
        val_metrics = val_metric.compute()
        avg_val_seg_loss = val_seg_loss / len(val_loader)
        avg_val_depth_loss = val_depth_loss / len(val_loader)
        
        # Update learning rate scheduler based on IoU
        scheduler.step(val_metrics["iou"])
        
        # Log to tensorboard
        if log_dir is not None:
            writer.add_scalar("train/seg_loss_epoch", avg_train_seg_loss, epoch)
            writer.add_scalar("train/depth_loss_epoch", avg_train_depth_loss, epoch)
            writer.add_scalar("train/iou", train_metrics["iou"], epoch)
            writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
            writer.add_scalar("train/abs_depth_error", train_metrics["abs_depth_error"], epoch)
            writer.add_scalar("train/tp_depth_error", train_metrics["tp_depth_error"], epoch)
            
            writer.add_scalar("val/seg_loss", avg_val_seg_loss, epoch)
            writer.add_scalar("val/depth_loss", avg_val_depth_loss, epoch)
            writer.add_scalar("val/iou", val_metrics["iou"], epoch)
            writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
            writer.add_scalar("val/abs_depth_error", val_metrics["abs_depth_error"], epoch)
            writer.add_scalar("val/tp_depth_error", val_metrics["tp_depth_error"], epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Seg Loss: {avg_train_seg_loss:.4f}, Depth Loss: {avg_train_depth_loss:.4f}")
        print(f"  Train - IoU: {train_metrics['iou']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Train - Depth Error: {train_metrics['abs_depth_error']:.4f}, TP Depth Error: {train_metrics['tp_depth_error']:.4f}")
        print(f"  Val - Seg Loss: {avg_val_seg_loss:.4f}, Depth Loss: {avg_val_depth_loss:.4f}")
        print(f"  Val - IoU: {val_metrics['iou']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Val - Depth Error: {val_metrics['abs_depth_error']:.4f}, TP Depth Error: {val_metrics['tp_depth_error']:.4f}")
        
        # Save best model based on IoU
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            models.save_model(model)
            print(f"  Saved model with validation IoU: {best_iou:.4f}")
    
    if log_dir is not None:
        writer.close()
    
    print(f"\nTraining completed! Best validation IoU: {best_iou:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train detection model")
    parser.add_argument("--train_path", type=str, default="drive_data/train",
                        help="Path to training data")
    parser.add_argument("--val_path", type=str, default="drive_data/val",
                        help="Path to validation data")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory for tensorboard logs")
    parser.add_argument("--seg_weight", type=float, default=1.0,
                        help="Weight for segmentation loss")
    parser.add_argument("--depth_weight", type=float, default=1.0,
                        help="Weight for depth loss")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA training")
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available() and not args.no_cuda:
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Create log directory
    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = Path("logs") / "detection" / timestamp
    
    args.log_dir = Path(args.log_dir)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading training data...")
    train_loader = load_data(
        args.train_path,
        transform_pipeline="default",
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    print("Loading validation data...")
    val_loader = load_data(
        args.val_path,
        transform_pipeline="default",
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    # Create model
    print("Creating model...")
    model = models.Detector(in_channels=3, num_classes=3)
    
    # Train model
    print("Starting training...")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        log_dir=args.log_dir,
        seg_weight=args.seg_weight,
        depth_weight=args.depth_weight,
    )


if __name__ == "__main__":
    main()
