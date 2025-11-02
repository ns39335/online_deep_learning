"""
Training script for Part 1: Classification

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

from homework.models import Classifier, save_model
from homework.metrics import AccuracyMetric
from homework.datasets.classification_dataset import load_data


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Training loop for the classifier

    Args:
        model: The classifier model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for the optimizer
        device: Device to train on (cuda or cpu)
    """
    model = model.to(device)
    
    # Create optimizer (Adam with weight decay for regularization)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Create loss function (CrossEntropyLoss for multi-class classification)
    criterion = nn.CrossEntropyLoss()
    
    # Metrics
    train_metric = AccuracyMetric()
    val_metric = AccuracyMetric()
    
    best_val_accuracy = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_metric.reset()
        train_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # Update metrics
            preds = model.predict(images)
            train_metric.add(preds, labels)
        
        # Validation phase
        model.eval()
        val_metric.reset()
        val_loss = 0.0
        val_batches = 0
        
        with torch.inference_mode():
            for batch in val_loader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_batches += 1
                
                preds = model.predict(images)
                val_metric.add(preds, labels)
        
        # Compute metrics
        train_results = train_metric.compute()
        val_results = val_metric.compute()
        
        avg_train_loss = train_loss / num_batches
        avg_val_loss = val_loss / val_batches
        
        # Update learning rate based on validation accuracy
        scheduler.step(val_results['accuracy'])
        
        # Log results
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Accuracy: {train_results['accuracy']:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Accuracy: {val_results['accuracy']:.4f}")
        
        # Save best model
        if val_results['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_results['accuracy']
            save_path = save_model(model)
            print(f"  ✓ New best model saved! Accuracy: {best_val_accuracy:.4f}")
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_accuracy:.4f}")


def main():
    """
    Main function to set up data and start training
    """
    # Set your hyperparameters
    batch_size = 128
    num_epochs = 50
    learning_rate = 1e-3
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load training data with augmentation
    print("\nLoading training data...")
    train_data = load_data(
        dataset_path="classification_data/train",
        transform_pipeline="aug",  # Use augmentation for training
        return_dataloader=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    
    # Load validation data (no augmentation!)
    print("Loading validation data...")
    val_data = load_data(
        dataset_path="classification_data/val",
        transform_pipeline="default",  # No augmentation for validation
        return_dataloader=True,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    
    # Create model
    print("\nCreating model...")
    model = Classifier(in_channels=3, num_classes=6)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Start training
    print("\nStarting training...")
    print("=" * 60)
    train(model, train_data, val_data, num_epochs, learning_rate, device)
    print("=" * 60)


if __name__ == "__main__":
    main()
