"""
Usage:
    python3 -m homework.train_planner --model mlp_planner --epochs 50
    python3 -m homework.train_planner --model transformer_planner --epochs 50
    python3 -m homework.train_planner --model cnn_planner --epochs 50 --transform default
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .datasets.road_dataset import load_data
from .metrics import PlannerMetric
from .models import load_model, save_model


def train(
    model_name: str = "mlp_planner",
    transform: str = "state_only",
    num_epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    num_workers: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train a planner model.

    Args:
        model_name: Name of model to train (mlp_planner, transformer_planner, cnn_planner)
        transform: Transform pipeline to use (state_only for MLP/Transformer, default for CNN)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        num_workers: Number of data loading workers
        device: Device to train on
    """
    # Setup
    device = torch.device(device)
    print(f"Training {model_name} on {device}")

    # Create log directory
    log_dir = Path("logs") / model_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Load model
    model = load_model(model_name)
    model = model.to(device)
    model.train()

    # Loss function (MSE for continuous waypoint prediction)
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Load data
    train_data = load_data(
        "drive_data/train",
        transform_pipeline=transform,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )

    val_data = load_data(
        "drive_data/val",
        transform_pipeline=transform,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"Training samples: {len(train_data.dataset)}")
    print(f"Validation samples: {len(val_data.dataset)}")

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_metric = PlannerMetric()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_data):
            # Move data to device
            if model_name == "cnn_planner":
                inputs = batch["image"].to(device)
            else:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                inputs = {"track_left": track_left, "track_right": track_right}

            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            # Forward pass
            optimizer.zero_grad()
            if model_name == "cnn_planner":
                pred_waypoints = model(inputs)
            else:
                pred_waypoints = model(**inputs)

            # Compute loss (only on valid waypoints)
            loss = criterion(
                pred_waypoints * waypoints_mask.unsqueeze(-1),
                waypoints * waypoints_mask.unsqueeze(-1),
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += loss.item()
            train_metric.add(pred_waypoints, waypoints, waypoints_mask)

            # Log
            if batch_idx % 20 == 0:
                writer.add_scalar("train/batch_loss", loss.item(), global_step)

            global_step += 1

        # Compute training metrics
        train_loss /= len(train_data)
        train_metrics = train_metric.compute()

        # Validation
        model.eval()
        val_metric = PlannerMetric()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_data:
                # Move data to device
                if model_name == "cnn_planner":
                    inputs = batch["image"].to(device)
                else:
                    track_left = batch["track_left"].to(device)
                    track_right = batch["track_right"].to(device)
                    inputs = {"track_left": track_left, "track_right": track_right}

                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                # Forward pass
                if model_name == "cnn_planner":
                    pred_waypoints = model(inputs)
                else:
                    pred_waypoints = model(**inputs)

                # Compute loss
                loss = criterion(
                    pred_waypoints * waypoints_mask.unsqueeze(-1),
                    waypoints * waypoints_mask.unsqueeze(-1),
                )

                val_loss += loss.item()
                val_metric.add(pred_waypoints, waypoints, waypoints_mask)

        val_loss /= len(val_data)
        val_metrics = val_metric.compute()

        # Update learning rate
        scheduler.step(val_loss)

        # Log epoch metrics
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/longitudinal_error", train_metrics["longitudinal_error"], epoch)
        writer.add_scalar("train/lateral_error", train_metrics["lateral_error"], epoch)

        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/longitudinal_error", val_metrics["longitudinal_error"], epoch)
        writer.add_scalar("val/lateral_error", val_metrics["lateral_error"], epoch)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Long: {val_metrics['longitudinal_error']:.4f} | "
            f"Val Lat: {val_metrics['lateral_error']:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = save_model(model)
            print(f"Saved best model to {model_path}")

    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train planner models")
    parser.add_argument(
        "--model",
        type=str,
        default="mlp_planner",
        choices=["mlp_planner", "transformer_planner", "cnn_planner"],
        help="Model to train",
    )
    parser.add_argument(
        "--transform",
        type=str,
        default=None,
        help="Transform pipeline (state_only for MLP/Transformer, default for CNN)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data workers")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )

    args = parser.parse_args()

    # Auto-select transform if not specified
    if args.transform is None:
        args.transform = "default" if args.model == "cnn_planner" else "state_only"

    train(
        model_name=args.model,
        transform=args.transform,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        device=args.device,
    )
