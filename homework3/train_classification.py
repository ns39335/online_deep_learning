"""
Training script for the Classifier model
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter

from homework import models
from homework.datasets.classification_dataset import load_data
from homework.metrics import AccuracyMetric


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    log_dir: Path = None,
):
    """
    Train the classification model

    Args:
        model: the Classifier model
        train_loader: training data loader
        val_loader: validation data loader
        num_epochs: number of training epochs
        learning_rate: learning rate for optimizer
        device: device to train on
        log_dir: directory for tensorboard logs
    """
    model = model.to(device)
    model.train()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Tensorboard writer
    if log_dir is not None:
        writer = SummaryWriter(log_dir)
    
    global_step = 0
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_metric = AccuracyMetric()
        train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            preds = model.predict(images)
            train_metric.add(preds, labels)
            
            if log_dir is not None and batch_idx % 10 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
            
            global_step += 1
        
        # Compute training metrics
        train_metrics = train_metric.compute()
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_metric = AccuracyMetric()
        val_loss = 0.0
        
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                logits = model(images)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                preds = model.predict(images)
                val_metric.add(preds, labels)
        
        val_metrics = val_metric.compute()
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate scheduler
        scheduler.step(val_metrics["accuracy"])
        
        # Log to tensorboard
        if log_dir is not None:
            writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
            writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)
            writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
            writer.add_scalar("val/loss", avg_val_loss, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            models.save_model(model)
            print(f"  Saved model with validation accuracy: {best_accuracy:.4f}")
    
    if log_dir is not None:
        writer.close()
    
    print(f"\nTraining completed! Best validation accuracy: {best_accuracy:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train classification model")
    parser.add_argument("--train_path", type=str, default="classification_data/train",
                        help="Path to training data")
    parser.add_argument("--val_path", type=str, default="classification_data/val",
                        help="Path to validation data")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory for tensorboard logs")
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
        args.log_dir = Path("logs") / "classification" / timestamp
    
    args.log_dir = Path(args.log_dir)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data with augmentation for training
    print("Loading training data with augmentation...")
    train_loader = load_data(
        args.train_path,
        transform_pipeline="aug",
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
    model = models.Classifier(in_channels=3, num_classes=6)
    
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
    )


if __name__ == "__main__":
    main()
