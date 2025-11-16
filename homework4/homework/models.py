from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Input: concatenate left and right track (n_track * 2 * 2 = 40 features)
        input_dim = n_track * 2 * 2
        hidden_dim = 512  # Increased capacity

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, n_waypoints * 2),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]

        # Concatenate and flatten: (b, n_track, 2) + (b, n_track, 2) -> (b, n_track*2*2)
        x = torch.cat([track_left, track_right], dim=1)  # (b, n_track*2, 2)
        x = x.reshape(batch_size, -1)  # (b, n_track*2*2)

        # Pass through MLP
        x = self.mlp(x)  # (b, n_waypoints*2)

        # Reshape to waypoints
        waypoints = x.reshape(batch_size, self.n_waypoints, 2)  # (b, n_waypoints, 2)

        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 128,  # Increased from 64
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Learned query embeddings for waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Encode track points to d_model dimensions with more capacity
        self.track_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Transformer decoder layers (cross-attention) - increased capacity
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,  # Increased from 4
            dim_feedforward=512,  # Increased from 256
            dropout=0.15,  # Slightly increased
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)  # Increased from 3

        # Output projection to 2D waypoints with intermediate layer
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 2),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]

        # Concatenate left and right track boundaries
        track = torch.cat([track_left, track_right], dim=1)  # (b, n_track*2, 2)

        # Encode track points to d_model dimensions
        memory = self.track_encoder(track)  # (b, n_track*2, d_model)

        # Get query embeddings and expand for batch
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)  # (b, n_waypoints, d_model)

        # Apply transformer decoder (queries attend to track features)
        output = self.transformer_decoder(queries, memory)  # (b, n_waypoints, d_model)

        # Project to 2D waypoints
        waypoints = self.output_proj(output)  # (b, n_waypoints, 2)

        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # CNN backbone for feature extraction - optimized for size
        self.conv_layers = nn.Sequential(
            # Input: (b, 3, 96, 128)
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),  # (b, 24, 48, 64)
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=5, stride=2, padding=2),  # (b, 48, 24, 32)
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),  # (b, 64, 12, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),  # (b, 96, 6, 8)
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

        # Calculate flattened size: 96 * 6 * 8 = 4608
        self.fc_layers = nn.Sequential(
            nn.Linear(96 * 6 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        # Normalize
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # CNN feature extraction
        x = self.conv_layers(x)  # (b, 96, 6, 8)

        # Flatten
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # (b, 96*6*8)

        # Fully connected layers
        x = self.fc_layers(x)  # (b, n_waypoints*2)

        # Reshape to waypoints
        waypoints = x.reshape(batch_size, self.n_waypoints, 2)  # (b, n_waypoints, 2)

        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024